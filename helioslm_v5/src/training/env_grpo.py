"""MultiEnv GRPO end-to-end loop (v5.43; v5.44 adds the async
`AsyncEnvGRPO` companion on the AsyncGRPO skeleton).

The v5.42 decision-engine wave delivered the two halves of multi-env GRPO
as separately oracle-tested pieces — ``envs.py`` (environment registry:
math / code / alignment-audit, constructed ground truth, no reference LLM)
and ``grpo.py`` (the group-relative update math) — and explicitly left the
wiring between them open. This module is that wiring:

    MultiEnvBatch.tasks -> GRPOTrainer._sample_response (G per task)
    -> MultiEnvBatch.reward (env-routed) -> _learn_from_samples(rewards=...)

Honest scope notes:
  - Rewards come from the ENVS, never from ``GRPOTrainer.compute_rewards``
    (which is math-answer-specific). The update math is byte-identical to
    the single-env trainer: per-task group normalization, clipped
    surrogate, k3 KL penalty. We inject rewards; we do not reimplement.
  - Per-env advantage normalization and cross-env reward scaling remain
    the CALLER's policy — the same disclaimer ``MultiEnvBatch`` carries.
    This loop normalizes within each task's own group of G samples only.
  - Sampling runs in eval mode under no_grad; the caller's training mode
    is restored on exit (M-T1 discipline, same as ``train_step``).
  - The alignment-audit env consumes REAL audit records
    (``MultiEnvBatch`` enforces this); nothing here weakens that rule.
"""
import queue
import threading

import torch

from .async_grpo import AsyncGRPO
from .envs import MultiEnvBatch
from .grpo import GRPOTrainer


class EnvGRPO:
    """End-to-end multi-env GRPO loop around a GRPOTrainer.

    Args:
        trainer: a GRPOTrainer (policy/ref/optimizer live on it).
        batch: a MultiEnvBatch providing tasks and env-routed scoring.

    Usage:
        loop = EnvGRPO(trainer, MultiEnvBatch({"math": 2, "code": 1}))
        stats = loop.train_step(seed=0)
        # stats carries the GRPO fields (loss/policy_loss/kl_penalty/
        # mean_reward) plus "env_rewards" and per-env means for audit.
    """

    def __init__(self, trainer: GRPOTrainer, batch: MultiEnvBatch):
        if not isinstance(trainer, GRPOTrainer):
            raise ValueError("EnvGRPO needs a GRPOTrainer instance")
        if not isinstance(batch, MultiEnvBatch):
            raise ValueError("EnvGRPO needs a MultiEnvBatch instance")
        self.trainer = trainer
        self.batch = batch

    def rollout(self, seed: int = 0) -> dict:
        """Sample tasks and G responses per task; score with the envs.

        Returns a dict with:
          tasks:   the task list from MultiEnvBatch (order preserved)
          samples: flat repeat-interleaved list, one dict per (task, group
                   member), each carrying the GRPO fields plus "env"
          rewards: float32 tensor [len(samples)], env-scored, same order
        """
        trainer = self.trainer
        tasks = self.batch.tasks(seed=seed)
        g = trainer.group_size
        prev_training = trainer.model.training
        samples = []
        try:
            trainer.model.eval()
            for t in tasks:
                for _ in range(g):
                    resp, old_lp, full_ids, plen = \
                        trainer._sample_response(t["prompt"])
                    samples.append({
                        "response": resp,
                        "old_logprob": old_lp,  # detached
                        "full_ids": full_ids,
                        "prompt_len": plen,
                        "env": t["env"],
                    })
        finally:
            trainer.model.train(prev_training)

        # Env-scored rewards, same flat repeat-interleaved order.
        rewards = []
        i = 0
        for t in tasks:
            for _ in range(g):
                rewards.append(
                    self.batch.reward(t, samples[i]["response"]))
                i += 1
        return {"tasks": tasks, "samples": samples,
                "rewards": torch.tensor(rewards, dtype=torch.float32)}

    def train_step(self, seed: int = 0) -> dict:
        """One full multi-env GRPO step: rollout + shared update math."""
        out = self.rollout(seed=seed)
        stats = self.trainer._learn_from_samples(
            out["samples"], answers=None, rewards=out["rewards"])
        envs = [s["env"] for s in out["samples"]]
        stats["env_rewards"] = out["rewards"].tolist()
        stats["mean_reward_by_env"] = {}
        for name in dict.fromkeys(envs):
            rs = [r for r, e in zip(out["rewards"].tolist(), envs)
                  if e == name]
            stats["mean_reward_by_env"][name] = sum(rs) / len(rs)
        return stats


class AsyncEnvGRPO(AsyncGRPO):
    """Asynchronous multi-env GRPO: EnvGRPO's env-routed scoring on the
    AsyncGRPO producer/consumer skeleton (v5.44).

    Same honesty contract as both parents:
      - Rewards come from the MultiEnvBatch ENVS, never from
        ``GRPOTrainer.compute_rewards`` — the update path is
        ``_learn_from_samples(samples, None, rewards=...)``, byte-identical
        to the synchronous ``EnvGRPO`` loop. Asynchrony changes WHO
        produces samples and WHEN, never the mathematics.
      - Per-env advantage normalization and cross-env reward scaling
        remain the CALLER's policy (MultiEnvBatch's own disclaimer);
        each group is normalized within its own G samples only.
      - The alignment-audit env still consumes REAL audit records
        (MultiEnvBatch enforces this upstream of this class).
      - Workers sample with the trainer's current weights, no
        weight-version sync (AsyncGRPO's documented toy-scale caveat).

    One mechanical difference vs. AsyncGRPO: each queue item is
    (task, samples) — tasks carry their own "env" routing, so there is
    no parallel answers list; the learner scores every sample with
    ``batch.reward(task, response)`` at consume time.
    """

    def __init__(self, trainer: GRPOTrainer, batch: MultiEnvBatch,
                 n_workers: int = 1, max_pending: int = 8, seed: int = 0):
        if not isinstance(trainer, GRPOTrainer):
            raise ValueError("AsyncEnvGRPO needs a GRPOTrainer instance")
        if not isinstance(batch, MultiEnvBatch):
            raise ValueError("AsyncEnvGRPO needs a MultiEnvBatch instance")
        super().__init__(trainer, n_workers=n_workers,
                         max_pending=max_pending, seed=seed)
        self.batch = batch

    # --- producer -----------------------------------------------------------

    def rollout_worker(self, worker_id: int, tasks: list) -> None:
        """Producer thread body: G samples per task, one (task, samples)
        group per queue item. Sampling discipline mirrors EnvGRPO.rollout
        (model.eval + no_grad inside _sample_response, training mode
        restored on exit); each sample is tagged with its task's "env"."""
        t = self.trainer
        was_training = t.model.training
        t.model.eval()
        try:
            for task in tasks:
                if self._stop.is_set():
                    return
                torch.manual_seed(self.seed + worker_id)
                samples = []
                for _ in range(self.G):
                    resp, old_lp, full_ids, plen = \
                        t._sample_response(task["prompt"])
                    samples.append({"response": resp,
                                    "old_logprob": old_lp,
                                    "full_ids": full_ids,
                                    "prompt_len": plen,
                                    "env": task["env"]})
                # Same back-pressure-aware put as AsyncGRPO: poll _stop so
                # stop() unblocks a worker parked on a full queue.
                while not self._stop.is_set():
                    try:
                        self.q.put((task, samples), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
        finally:
            t.model.train(was_training)

    # --- consumer -----------------------------------------------------------

    def learn(self, timeout=None):
        """Drain ONE (task, samples) group, score it with the envs, apply
        the shared update. Returns the metrics dict (with "env" tagged),
        or None on empty queue."""
        try:
            task, samples = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        rewards = torch.tensor(
            [self.batch.reward(task, s["response"]) for s in samples],
            dtype=torch.float32)
        m = self.trainer._learn_from_samples(samples, None, rewards=rewards)
        m["env"] = task["env"]
        self.metrics.append(m)
        self.q.task_done()
        return m

    def run(self, tasks: list = None, seed: int = 0) -> list:
        """Convenience driver: sample tasks from the MultiEnvBatch
        (unless an explicit task list is given), split them round-robin
        across workers, learn until every group is consumed, join
        threads. Returns the per-group metrics list
        (len == number of tasks)."""
        if tasks is None:
            tasks = self.batch.tasks(seed=seed)
        per_worker = [tasks[i::self.n_workers]
                      for i in range(self.n_workers)]
        self._threads = [
            threading.Thread(target=self.rollout_worker,
                             args=(wid, pt), daemon=True)
            for wid, pt in enumerate(per_worker)]
        for t in self._threads:
            t.start()
        done = 0
        while done < len(tasks):
            m = self.learn(timeout=30)
            if m is not None:
                done += 1
        for t in self._threads:
            t.join()
        return self.metrics
