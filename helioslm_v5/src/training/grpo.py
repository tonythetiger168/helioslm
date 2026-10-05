"""GRPO - Group Relative Policy Optimization (DeepSeek-R1 / DeepSeekMath Style)

Key innovation: No value model needed. Uses group mean as baseline.
Simpler than PPO, more stable than REINFORCE.

Deviations from the DeepSeekMath reference, noted honestly:
  - The ratio/clip objective here is **sequence-level** (one ratio per
    response, from the summed per-token log-probs), whereas DeepSeekMath
    uses **per-token** ratios with per-token clipping and length
    normalization. The sequence-level variant is a accepted simplification
    but is not numerically identical to the reference objective.

Tokenizer: the model consumes token ids while prompts here are text. Pass a
``tokenizer`` with ``encode``/``decode`` if the model has one; otherwise a
built-in byte-level fallback (UTF-8 bytes <-> ids) is used.
"""
import re
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class _ByteLevelTokenizer:
    """Minimal byte-level fallback tokenizer (UTF-8 bytes <-> token ids)."""

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        # Safe for arbitrary model token ids (vocab > 256): ids are reduced
        # mod 256 (also handles negative ids) so they always form valid bytes.
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class GRPOTrainer:
    """
    GRPO trainer for reasoning models.

    Algorithm:
      1. Sample G responses from the policy for each question (no_grad),
         storing the OLD policy's per-sequence log-prob sums (detached).
      2. Compute reward for each response (e.g., correctness, format).
      3. Compute advantage: A_i = (r_i - mean(r_group)) / std(r_group).
      4. Recompute the NEW policy's log-probs with a grad-carrying forward
         over the full prompt+response sequence.
      5. Update policy: maximize clip(pi_new/pi_old * A, 1-eps, 1+eps)
         minus a non-negative k3 KL penalty against the frozen ref model.
    """

    def __init__(self, model, ref_model, config, tokenizer: Optional[object] = None):
        self.model = model  # Policy model
        self.ref_model = ref_model  # Reference model (frozen)
        self.config = config

        self.group_size = config.grpo.group_size  # G, typically 8-16
        self.epsilon = config.grpo.epsilon  # Clip parameter
        self.kl_coef = config.grpo.kl_coef  # KL penalty coefficient
        self.max_new_tokens = getattr(config.grpo, "max_new_tokens", 16)

        self.device = next(model.parameters()).device
        self.tokenizer = tokenizer if tokenizer is not None else _ByteLevelTokenizer()

        # Reference model is frozen and always in eval mode.
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.grpo.lr,
            weight_decay=0.01
        )

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------
    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        return torch.tensor([list(ids)], dtype=torch.long, device=self.device)

    def _decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids))

    # ------------------------------------------------------------------
    # Log-prob helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sequence_logprob(model, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Summed log-prob of the response tokens of one sequence.

        ``full_ids`` is (1, T) = prompt + response; response tokens are
        positions ``prompt_len..T-1``, predicted by logits at positions
        ``prompt_len-1..T-2``. Gradients flow when called outside no_grad.
        """
        logits, _, _ = model(full_ids)
        logp = torch.log_softmax(logits[:, prompt_len - 1:-1, :].float(), dim=-1)
        tgt = full_ids[:, prompt_len:]
        token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return token_logp.sum()

    def compute_rewards(self, responses: List[str], answers: List[str]) -> torch.Tensor:
        """
        Compute rewards for responses.

        Reward components:
          - Accuracy: 1.0 if correct, 0.0 if wrong
          - Format: +0.5 if follows reasoning format (e.g., <think>...</think>)
          - Length: -0.1 * (length - optimal) / optimal
        """
        rewards = []
        for resp, ans in zip(responses, answers):
            reward = 0.0

            # Accuracy reward
            if self._check_correctness(resp, ans):
                reward += 1.0

            # Format reward
            if "<think>" in resp and "</think>" in resp:
                reward += 0.5

            # Length penalty
            optimal_len = len(ans) * 3  # Heuristic
            length_penalty = -0.1 * abs(len(resp) - optimal_len) / max(optimal_len, 1)
            reward += length_penalty

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _check_correctness(self, response: str, answer: str) -> bool:
        """Check correctness by extracting the FINAL answer and comparing for
        equality — no substring matching (which produced false positives such
        as response "25" matching answer "2").

        Extraction (in priority order): last ``\\boxed{...}``; otherwise the
        last number in the post-``</think>`` segment; otherwise the last
        non-empty line. Both sides are normalized (whitespace/punctuation/
        thousand separators stripped). This is still a heuristic — a
        production setup should use symbolic verification — but it has no
        substring false positives.
        """
        if "</think>" in response:
            final = response.split("</think>")[-1]
        else:
            final = response

        pred = self._extract_final_answer(final)
        gold = self._normalize_answer(answer)
        if pred is None or not gold:
            return False
        if pred == gold:
            return True
        # Numeric tolerance: "42" vs "42.0".
        try:
            return float(pred) == float(gold)
        except ValueError:
            return False

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize an answer string for equality comparison."""
        t = text.strip()
        # Strip surrounding math-mode / currency decorations and thousand
        # separators, then a trailing period.
        t = t.strip("$€£ ").replace(",", "").replace(" ", "")
        return t.rstrip(".")

    @classmethod
    def _extract_final_answer(cls, text: str) -> Optional[str]:
        """Extract the final answer candidate from free-form text."""
        # 1) LaTeX \boxed{...} (last occurrence wins).
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
        if boxed:
            return cls._normalize_answer(boxed[-1])
        # 2) Last number (integer or decimal, optionally signed).
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if numbers:
            return cls._normalize_answer(numbers[-1])
        # 3) Last non-empty line.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            return cls._normalize_answer(lines[-1])
        return None

    def train_step(self, questions: List[str], answers: List[str]):
        """
        Single GRPO training step.

        Training-mode discipline (M-T1): ``model.generate`` switches the
        policy to eval mode and never restores it, which would silently
        disable dropout etc. during the training forward. We therefore
        snapshot the caller's mode, sample in eval mode under no_grad,
        switch back to train mode before the grad-carrying forwards, and
        restore the caller's original mode on exit (even on exceptions).

        Memory discipline (m1): instead of stacking B*G grad-carrying
        forwards into one graph before a single backward (all graphs alive
        at once), each sample's loss is backwarded individually and
        gradients accumulate on the parameters — only one graph is alive
        at a time. Dividing each per-sample loss by N = B*G makes the
        accumulated gradients identical to the batched-mean version.

        Args:
            questions: list of question strings (non-empty; each question
                must be a non-blank string so it tokenizes to >= 1 token)
            answers: list of ground truth answers (same length as questions)
        """
        # Input guards: an empty batch (or an empty/blank prompt) used to
        # fail deep inside generation/logprob code with an opaque error
        # (empty tensor ops, 0-length prompt_len). Fail fast with a clear
        # message instead.
        if not questions:
            raise ValueError(
                "GRPOTrainer.train_step requires at least one question; "
                "got an empty prompts list")
        if len(questions) != len(answers):
            raise ValueError(
                f"questions/answers length mismatch: {len(questions)} "
                f"questions vs {len(answers)} answers")
        for i, q in enumerate(questions):
            if not isinstance(q, str) or not q.strip():
                raise ValueError(
                    f"GRPOTrainer.train_step: question at index {i} is "
                    "empty or blank; prompts must contain at least one "
                    "non-whitespace character")
        batch_size = len(questions)
        n_samples = batch_size * self.group_size
        prev_training = self.model.training
        try:
            # 1. Sample G responses per question (old policy, eval mode,
            #    no grad).
            self.model.eval()
            samples = []  # one dict per (question, group member)
            for question in questions:
                for _ in range(self.group_size):
                    resp, old_logprob, full_ids, prompt_len = \
                        self._sample_response(question)
                    samples.append({
                        "response": resp,
                        "old_logprob": old_logprob,  # detached
                        "full_ids": full_ids,
                        "prompt_len": prompt_len,
                    })

            return self._learn_from_samples(samples, answers)
        finally:
            # Restore the caller's original training mode (M-T1).
            self.model.train(prev_training)

    def _learn_from_samples(self, samples: list, answers: list) -> dict:
        """Steps 2-8 of GRPO: rewards, group advantages, reference log-probs,
        PPO-style clipped update with the k3 KL estimator, optimizer step.

        Extracted from train_step so the math exists EXACTLY ONCE: the
        synchronous trainer and AsyncGRPO (v5.31, GLM-5 direction) share
        this method — async only changes WHO produces the samples and WHEN,
        never the update. `samples` is the flat, repeat-interleaved list
        (batch_size * group_size entries) in the same shape train_step
        builds; `answers` is per-question (len == len(samples)/group_size).
        """
        batch_size = len(samples) // self.group_size
        n_samples = len(samples)

        all_responses = [s["response"] for s in samples]
        # Rewards — each of the G responses of question i is scored against
        # answers[i] (repeat-interleaved alignment).
        all_answers = [a for a in answers for _ in range(self.group_size)]
        rewards = self.compute_rewards(all_responses, all_answers)

        # Group-normalized advantages.
        rewards = rewards.view(batch_size, self.group_size)
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, unbiased=False, keepdim=True) \
            .clamp(min=1e-8)
        advantages = (rewards - mean_rewards) / std_rewards
        # NOTE: with group_size == 1 every advantage is exactly 0 (each
        # sample is its own baseline), so the policy-gradient term
        # vanishes; use group_size >= 2 in practice.
        advantages_flat = advantages.reshape(-1).detach()

        # Reference model log-probs (frozen, no grad, flat order).
        ref_logprobs = self._get_ref_logprobs(samples).detach()

        # Grad-carrying training forwards MUST run in train mode so
        # dropout and other train-only behavior are active.
        self.model.train()

        self.optimizer.zero_grad()
        policy_loss_total = 0.0
        kl_total = 0.0
        for i, s in enumerate(samples):
            new_lp = self._sequence_logprob(
                self.model, s["full_ids"], s["prompt_len"])

            # Ratio against the OLD policy (the ref model only enters the
            # KL term). Sequence-level ratio — see module docstring for the
            # difference vs DeepSeekMath's per-token objective. Clamp the
            # log-ratio before exp() so a diverged policy can never produce
            # inf/overflow ratios (m3).
            log_ratio = (new_lp - s["old_logprob"]).clamp(-20.0, 20.0)
            ratio = torch.exp(log_ratio)
            adv = advantages_flat[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.epsilon,
                                1 + self.epsilon) * adv
            policy_i = -torch.min(surr1, surr2)

            # KL penalty with DeepSeekMath's k3 estimator:
            # exp(logr) - logr - 1 >= 0 pointwise for
            # logr = log pi_ref - log pi_new. expm1 avoids the
            # catastrophic cancellation of exp(logr) - 1 near logr == 0
            # (m4); the clamp only guards against inf at absurdly large
            # logr.
            logr = (ref_logprobs[i] - new_lp).clamp(max=60.0)
            kl_i = self.kl_coef * (torch.expm1(logr) - logr)

            # 1/N scaling: accumulated grads == single batched mean.
            loss_i = (policy_i + kl_i) / n_samples
            loss_i.backward()
            policy_loss_total += policy_i.item() / n_samples
            kl_total += kl_i.item() / n_samples

        # Update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {
            "loss": policy_loss_total + kl_total,
            "policy_loss": policy_loss_total,
            "kl_penalty": kl_total,
            "mean_reward": mean_rewards.mean().item(),
        }

    def _sample_response(self, question: str) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        """Sample a response from the policy model (no grad).

        Returns (response_text, old_logprob_detached, full_ids, prompt_len).
        """
        prompt_ids = self._encode(question)
        prompt_len = prompt_ids.shape[1]
        with torch.no_grad():
            # HeliosLMv5.generate's contract has no eos_token_id argument;
            # it stops on config.eos_token_id and freezes finished rows.
            gen = self.model.generate(
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=1.0,
                # top_p=1.0 (no nucleus filter): old_logprob and the
                # importance ratio score the plain softmax distribution, so
                # the behavior policy must sample from exactly that
                # distribution — the default top_p=0.9 filter would make
                # sampling and logprobs disagree (real off-policy bias
                # whenever the filter binds).
                top_p=1.0,
            )
            if gen.dim() == 1:
                gen = gen.unsqueeze(0)
            gen = gen.to(self.device)
            # Contract (verified in model_v5.py: ``generated`` starts as
            # ``input_ids.clone()`` and new tokens are appended): generate
            # returns the FULL sequence, prompt prefix included, with at
            # least one new token. Require it explicitly instead of guessing
            # between full-sequence and continuation conventions — the old
            # length heuristic could misclassify either one (m2).
            if gen.shape[1] > prompt_len and torch.equal(
                    gen[:, :prompt_len], prompt_ids):
                full_ids = gen
            else:
                raise ValueError(
                    "model.generate must return the full sequence "
                    "(prompt + >=1 new tokens) per the HeliosLMv5.generate "
                    f"contract; got output shape {tuple(gen.shape)} for "
                    f"prompt_len={prompt_len} with a non-matching prefix"
                )
            # OLD policy per-sequence log-prob sum (detached).
            old_logprob = self._sequence_logprob(
                self.model, full_ids, prompt_len).detach()

        response = self._decode(full_ids[0, prompt_len:].tolist())
        response = self._clean_response_text(response)
        return response, old_logprob, full_ids, prompt_len

    def _clean_response_text(self, text: str) -> str:
        """Strip EOS/control characters from decoded text before reward
        computation (m6).

        With the byte-level fallback tokenizer, special ids (EOS=2, PAD=0,
        BOS=1) decode to raw control bytes; with a real tokenizer they may
        decode to special-token strings. Neither should leak into the
        reward heuristics.
        """
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if isinstance(eos_token, str) and eos_token:
            text = text.replace(eos_token, "")
        # Drop C0/C1 control chars (covers EOS/PAD/BOS bytes in the
        # byte-level fallback), keeping tab/newline for line-based parsing.
        return "".join(
            ch for ch in text
            if ch in "\n\t" or (ord(ch) >= 32 and ord(ch) != 127)
        )

    def _get_ref_logprobs(self, samples) -> torch.Tensor:
        """Per-sequence response log-prob sums from the frozen ref model."""
        logprobs = []
        with torch.no_grad():
            for s in samples:
                logprobs.append(self._sequence_logprob(
                    self.ref_model, s["full_ids"], s["prompt_len"]))
        return torch.stack(logprobs).detach()
