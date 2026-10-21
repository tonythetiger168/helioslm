"""Multi-teacher on-policy distillation, MOPD (v5.76, roadmap B2-6).

MiMo-V2-Flash (Xiaomi, arXiv 2601.02780) post-trains one student from
several domain-specialised teachers. The student samples its own
responses (on-policy) and every sampled token is scored by the teacher of
that sample's domain. Definition as implemented by NVIDIA NeMo RL
(``adv_estimator: opd``, docs/about/algorithms/mopd.md), which cites the
report; the report itself was not reachable from this container:

  advantage   A_t = sg[ log pi_teacher(y_t) - log pi_student(y_t) ]
              on the student's own sampled tokens, policy tokens only
              (environment / tool tokens masked out);
  update      REINFORCE form (no PPO ratio clipping), no reference KL:
              loss = -sum_t A_t * pi_new(y_t) / pi_old(y_t) / T, whose
              gradient at the sampling policy is -sum_t A_t grad log pi;
              maximising A is reverse-KL(student || teacher) minimisation,
              using only the teacher's log-prob of the sampled token;
  routing     each sample is scored by exactly ONE teacher, chosen by its
              domain (``env``); unmapped domains fall back to a default
              teacher, or raise in strict mode; no ensemble averaging;
  TROPD       optional proximal teacher pi_prox = a*pi_T + (1-a)*pi_S,
              A_t = sg[log pi_prox - log pi_S] >= log(1 - a); a = 1 is
              plain MOPD; optional subtraction of the mean advantage over
              every trained token of the step;
  full vocab  optional exact per-position reverse KL
              sum_v p_S(v) (log p_S(v) - log p_T(v)) at the sampled
              positions instead of the sampled-token estimator.

Wiring: samples are the GRPO sample dicts (``full_ids``, ``prompt_len``,
optional ``action_mask``, optional ``old_token_logprobs``) plus ``env``;
log-probs come from ``GRPOTrainer._token_logprobs``, the policy and its
optimizer from the wrapped ``GRPOTrainer``. Not implemented: combining
the distillation advantage with outcome rewards, async collection.
"""
from typing import Dict, List, Optional

import torch

from .grpo import GRPOTrainer


class MOPDTrainer:
    """On-policy distillation from per-domain teachers.

    Args:
        trainer: a GRPOTrainer; its ``model`` is the student and its
            optimizer takes the steps. Its ref model, KL and group size
            are unused (MOPD has no group baseline).
        teachers: {alias: frozen model}; every teacher must share the
            student's vocabulary.
        teacher_by_env: {env name: alias}. A sample's ``env`` picks its
            teacher; envs not in the map use ``default_teacher``.
        default_teacher: fallback alias (must be a key of ``teachers``);
            None means every env must be mapped.
        strict: raise on an unmapped env even when a default exists.
        alpha: TROPD mixing weight in (0, 1]; 1.0 is plain MOPD.
        subtract_global_baseline: subtract the mean advantage over all
            trained tokens of the step.
        full_vocab: use the exact full-vocabulary reverse KL loss.
    """

    def __init__(self, trainer: GRPOTrainer, teachers: Dict[str, object],
                 teacher_by_env: Optional[Dict[str, str]] = None,
                 default_teacher: Optional[str] = None, strict: bool = False,
                 alpha: float = 1.0, subtract_global_baseline: bool = False,
                 full_vocab: bool = False):
        if not isinstance(trainer, GRPOTrainer):
            raise ValueError("MOPDTrainer needs a GRPOTrainer instance")
        if not teachers:
            raise ValueError("MOPDTrainer needs at least one teacher")
        teacher_by_env = dict(teacher_by_env or {})
        for env, alias in teacher_by_env.items():
            if alias not in teachers:
                raise ValueError(f"env {env!r} maps to unknown teacher {alias!r}")
        if default_teacher is not None and default_teacher not in teachers:
            raise ValueError(f"default_teacher {default_teacher!r} is not a teacher")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if full_vocab and (alpha != 1.0 or subtract_global_baseline):
            raise ValueError("TROPD alpha / global baseline only reshape the "
                             "sampled-token advantage; full_vocab ignores it")
        self.trainer = trainer
        self.teachers = teachers
        self.teacher_by_env = teacher_by_env
        self.default_teacher = default_teacher
        self.strict = strict
        self.alpha = alpha
        self.subtract_global_baseline = subtract_global_baseline
        self.full_vocab = full_vocab
        for t in teachers.values():
            t.eval()
            for p in t.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    def teacher_for(self, sample: dict) -> str:
        env = sample.get("env")
        if env in self.teacher_by_env:
            return self.teacher_by_env[env]
        if self.strict or self.default_teacher is None:
            raise KeyError(f"no teacher for env {env!r}")
        return self.default_teacher

    @torch.no_grad()
    def rollout(self, prompt_ids: torch.Tensor, envs: List[str],
                max_new_tokens: int, temperature: float = 1.0) -> List[dict]:
        """Sample one response per prompt row from the student (eval mode,
        plain softmax so the stored log-probs score the sampling
        distribution). A response ends at its first EOS (kept)."""
        model = self.trainer.model
        prev = model.training
        model.eval()
        try:
            gen = model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_p=1.0)
        finally:
            model.train(prev)
        P = prompt_ids.shape[1]
        eos = model.config.eos_token_id
        samples = []
        for row, env in zip(gen, envs):
            resp = row[P:].tolist()
            if eos in resp:
                resp = resp[:resp.index(eos) + 1]
            full = torch.cat([row[:P], torch.tensor(resp, dtype=row.dtype,
                                                    device=row.device)])
            samples.append({"full_ids": full.unsqueeze(0), "prompt_len": P,
                            "env": env})
        return samples

    @torch.no_grad()
    def advantages(self, samples: List[dict]):
        """Per-sample token advantages (after TROPD and the optional global
        baseline), the old student log-probs, and the raw teacher-minus-
        student gaps. Fills ``old_token_logprobs`` on samples lacking it."""
        tr = self.trainer
        prev = tr.model.training
        tr.model.eval()
        try:
            olds, gaps, advs = [], [], []
            for s in samples:
                if "old_token_logprobs" not in s:
                    s["old_token_logprobs"] = tr._token_logprobs(
                        tr.model, s["full_ids"], s["prompt_len"],
                        s.get("action_mask")).detach()
                old = s["old_token_logprobs"]
                teach = tr._token_logprobs(
                    self.teachers[self.teacher_for(s)], s["full_ids"],
                    s["prompt_len"], s.get("action_mask"))
                gap = teach - old
                if self.alpha < 1.0:
                    prox = torch.logaddexp(
                        teach + torch.log(torch.tensor(self.alpha)),
                        old + torch.log(torch.tensor(1.0 - self.alpha)))
                    adv = prox - old
                else:
                    adv = gap
                olds.append(old)
                gaps.append(gap)
                advs.append(adv)
        finally:
            tr.model.train(prev)
        if self.subtract_global_baseline:
            flat = torch.cat(advs)
            if flat.numel():
                mean = flat.mean()
                advs = [a - mean for a in advs]
        return advs, olds, gaps

    def _full_vocab_kl(self, s: dict) -> torch.Tensor:
        """Summed exact reverse KL over the sample's policy positions."""
        P = s["prompt_len"]
        ids = s["full_ids"]
        logits = self.trainer.model(ids)[0][:, P - 1:-1, :].float()
        with torch.no_grad():
            t_logits = self.teachers[self.teacher_for(s)](ids)[0][
                :, P - 1:-1, :].float()
        if logits.shape[-1] != t_logits.shape[-1]:
            raise ValueError("teacher and student vocabularies differ")
        ls = torch.log_softmax(logits, -1)[0]
        lt = torch.log_softmax(t_logits, -1)[0]
        kl = (ls.exp() * (ls - lt)).sum(-1)
        mask = s.get("action_mask")
        if mask is not None:
            kl = kl[mask.to(kl.device)]
        return kl.sum()

    def sample_loss(self, s: dict, adv: torch.Tensor,
                    old: torch.Tensor) -> torch.Tensor:
        """Grad-carrying summed loss of one sample (before the 1/T
        normalisation): -sum_t A_t * ratio_t, or the summed exact reverse
        KL with ``full_vocab``."""
        if self.full_vocab:
            return self._full_vocab_kl(s)
        tr = self.trainer
        new = tr._token_logprobs(tr.model, s["full_ids"], s["prompt_len"],
                                 s.get("action_mask"))
        ratio = torch.exp((new - old).clamp(-20.0, 20.0))
        return -(adv * ratio).sum()

    def learn(self, samples: List[dict]) -> dict:
        """One optimizer step on ``samples``; loss normalised by the total
        number of trained tokens."""
        if not samples:
            raise ValueError("MOPDTrainer.learn needs at least one sample")
        tr = self.trainer
        advs, olds, gaps = self.advantages(samples)
        total = sum(int(o.numel()) for o in olds)
        stats = {"tokens": total,
                 "gap_mean": (torch.cat(gaps).mean().item() if total else 0.0),
                 "adv_mean": (torch.cat(advs).mean().item() if total else 0.0)}
        if total == 0:
            stats["loss"] = 0.0
            return stats
        prev = tr.model.training
        tr.model.train()
        tr.optimizer.zero_grad()
        loss_total = 0.0
        try:
            for s, adv, old in zip(samples, advs, olds):
                if old.numel() == 0:
                    continue
                loss_i = self.sample_loss(s, adv, old) / total
                loss_i.backward()
                loss_total += loss_i.item()
            torch.nn.utils.clip_grad_norm_(tr.model.parameters(), 1.0)
            tr.optimizer.step()
        finally:
            tr.model.train(prev)
        stats["loss"] = loss_total
        return stats

    def train_step(self, prompt_ids: torch.Tensor, envs: List[str],
                   max_new_tokens: int) -> dict:
        return self.learn(self.rollout(prompt_ids, envs, max_new_tokens))
