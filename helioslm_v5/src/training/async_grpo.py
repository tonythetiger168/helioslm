"""HeliosLM v5.31 — asynchronous GRPO skeleton (GLM-5 direction).

GLM-5's key post-training idea: decouple rollout generation from learning
so slow environment interaction never stalls the optimizer. This module
provides the decoupled interface with an exactness oracle:

- RolloutWorker threads produce complete groups (question -> G samples +
  detached old log-probs, exactly train_step's sampling discipline) into a
  bounded queue (back-pressure via max_pending).
- The learner drains groups and calls GRPOTrainer._learn_from_samples —
  the SAME method the synchronous train_step uses. The update math exists
  exactly once; asynchrony changes WHO produces samples and WHEN, never
  the mathematics (verified bitwise by T26 parity oracle).

Honest scope (recorded, not hidden): thread-level skeleton, not GLM-5's
process/GPU-actor fleet; FIFO fairness, no preemption, no weight-version
sync (workers sample with the trainer's current weights — acceptable only
because learner updates are far slower than worker drains at toy scale).
Production hardening is milestone-13 infra work.
"""
import queue
import threading

import torch


class AsyncGRPO:
    def __init__(self, trainer, n_workers: int = 1, max_pending: int = 8,
                 seed: int = 0):
        assert n_workers >= 1
        self.trainer = trainer
        self.n_workers = n_workers
        self.G = trainer.group_size
        self.seed = seed
        self.q: queue.Queue = queue.Queue(maxsize=max_pending)
        self._stop = threading.Event()
        self._threads: list = []
        self.metrics: list = []

    # --- producer -----------------------------------------------------------

    def rollout_worker(self, worker_id: int, questions: list,
                       answers: list) -> None:
        """Producer thread body. Sampling discipline mirrors train_step:
        model.eval + torch.no_grad, G samples per question, one group per
        queue item (atomic, FIFO)."""
        t = self.trainer
        was_training = t.model.training
        t.model.eval()
        try:
            for question, answer in zip(questions, answers):
                if self._stop.is_set():
                    return
                torch.manual_seed(self.seed + worker_id)
                samples = []
                for _ in range(self.G):
                    resp, old_lp, full_ids, plen = t._sample_response(question)
                    samples.append({"response": resp,
                                    "old_logprob": old_lp,
                                    "full_ids": full_ids,
                                    "prompt_len": plen})
                # Back-pressure-aware put: a full queue must not pin the
                # worker forever — poll _stop so stop() unblocks it. A group
                # abandoned mid-stop is discarded (documented cooperative
                # shutdown semantics; the learner simply never sees it).
                while not self._stop.is_set():
                    try:
                        self.q.put((question, answer, samples), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
        finally:
            t.model.train(was_training)

    # --- consumer -----------------------------------------------------------

    def learn(self, timeout=None):
        """Drain ONE group and apply its update. Returns the metrics dict,
        or None on empty queue."""
        try:
            question, answer, samples = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        m = self.trainer._learn_from_samples(samples, [answer])
        self.metrics.append(m)
        self.q.task_done()
        return m

    def run(self, questions: list, answers: list) -> list:
        """Convenience driver: split questions round-robin across workers,
        learn until every group is consumed, join threads. Returns the
        per-group metrics list (len == len(questions))."""
        per_q = [questions[i::self.n_workers] for i in range(self.n_workers)]
        per_a = [answers[i::self.n_workers] for i in range(self.n_workers)]
        self._threads = [
            threading.Thread(target=self.rollout_worker,
                             args=(wid, pq, pa), daemon=True)
            for wid, (pq, pa) in enumerate(zip(per_q, per_a))]
        for t in self._threads:
            t.start()
        done = 0
        while done < len(questions):
            m = self.learn(timeout=30)
            if m is not None:
                done += 1
        for t in self._threads:
            t.join()
        return self.metrics

    def stop(self) -> None:
        """Cooperative shutdown: workers check _stop between questions."""
        self._stop.set()
