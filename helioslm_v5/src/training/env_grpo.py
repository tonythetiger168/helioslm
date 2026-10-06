"""MultiEnv GRPO end-to-end loop (v5.43).

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
import torch

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
