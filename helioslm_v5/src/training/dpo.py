"""DPO — Direct Preference Optimization (Rafailov et al. 2023).

The closed-form alternative to RLHF: skip the reward model and PPO, and
optimize preferences directly against a frozen reference policy with the
sigmoid-margin loss

    L = -E log σ( β[ (logπ_θ(y_w|x) - logπ_ref(y_w|x))
                     - (logπ_θ(y_l|x) - logπ_ref(y_l|x)) ] )

Deviations from the reference paper, noted honestly (same simplification
family as ``grpo.py`` in this directory):
  - Log-probs are **sequence-level** (summed per-token log-probs, one
    margin per pair). The original uses per-token parameterizations with
    length normalization; sequence-level is the accepted simplification
    and is NOT numerically identical to the reference objective.
  - ``beta`` defaults to 0.1 (paper default) but is exposed — the KL
    trade-off knob is a claim about the deployment, not a constant.
  - The reference model is required (pass ``ref_model``). A reference-free
    variant exists in the literature (ORPO/simPO family) but is a
    DIFFERENT objective; we do not silently substitute it.

Contract: ``model``/``ref_model`` follow the HeliosLMv5 call convention —
``model(full_ids)`` returns a tuple whose first element is the logits
tensor of shape (1, T, V). Pairs are dicts:
    {"prompt": str or token list, "chosen": str or token list,
     "rejected": str or token list}
Text is tokenized with the provided ``tokenizer`` (encode/decode) or a
byte-level fallback, same convention as GRPOTrainer.
"""
from typing import List, Optional

import torch

from helioslm_v5.src.training.grpo import _ByteLevelTokenizer


class DPOTrainer:
    """DPO trainer over preference pairs.

    Args:
        model: policy model (trained in place by ``fit``).
        ref_model: frozen reference policy; must share the tokenizer and
            output contract with ``model``.
        beta: KL-ish temperature on the log-prob margin (paper default 0.1).
        lr: Adam learning rate for the policy.
        tokenizer: object with encode/decode; byte-level fallback if None.

    Metrics returned by ``loss``/``fit`` steps: ``loss``, ``acc``
    (fraction of pairs where the margin favors chosen), ``margin``
    (mean logπ_θ(y_w) - logπ_θ(y_l)).
    """

    def __init__(self, model, ref_model, beta: float = 0.1, lr: float = 1e-6,
                 tokenizer=None):
        if beta <= 0:
            raise ValueError(f"DPOTrainer: beta must be positive, got {beta}")
        self.model = model
        self.ref_model = ref_model
        self.beta = float(beta)
        self.tokenizer = tokenizer if tokenizer is not None \
            else _ByteLevelTokenizer()
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    # ------------------------------------------------------------------
    # Log-prob helpers (sequence-level, mirrors grpo.GRPOTrainer)
    # ------------------------------------------------------------------
    def _encode(self, x) -> List[int]:
        if isinstance(x, str):
            return list(self.tokenizer.encode(x))
        return list(x)

    @staticmethod
    def _sequence_logprob(model, full_ids: torch.Tensor,
                          prompt_len: int) -> torch.Tensor:
        """Summed log-prob of the response tokens (gradients flow when
        called outside no_grad on the policy model)."""
        out = model(full_ids)
        logits = out[0] if isinstance(out, tuple) else out
        logp = torch.log_softmax(
            logits[:, prompt_len - 1:-1, :].float(), dim=-1)
        tgt = full_ids[:, prompt_len:]
        return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------
    def loss(self, pairs: List[dict]) -> tuple:
        """DPO loss on a batch of preference pairs + honest metrics."""
        if not pairs:
            raise ValueError("DPOTrainer.loss: empty pair list")
        margins, accs = [], []
        losses = []
        for pr in pairs:
            p_ids = self._encode(pr["prompt"])
            c_ids = self._encode(pr["chosen"])
            r_ids = self._encode(pr["rejected"])
            c_full = torch.tensor([p_ids + c_ids])
            r_full = torch.tensor([p_ids + r_ids])
            pl = len(p_ids)

            pi_c = self._sequence_logprob(self.model, c_full, pl)
            pi_r = self._sequence_logprob(self.model, r_full, pl)
            with torch.no_grad():
                ref_c = self._sequence_logprob(self.ref_model, c_full, pl)
                ref_r = self._sequence_logprob(self.ref_model, r_full, pl)

            margin = self.beta * ((pi_c - ref_c) - (pi_r - ref_r))
            losses.append(-torch.nn.functional.logsigmoid(margin))
            margins.append(float(margin.detach()) / self.beta)
            accs.append(float(margin > 0))

        loss = torch.stack(losses).mean()
        metrics = {
            "loss": float(loss.detach()),
            "acc": sum(accs) / len(accs),
            "margin": sum(margins) / len(margins),
        }
        return loss, metrics

    def fit(self, pairs: List[dict], epochs: int = 1,
            verbose: bool = False) -> List[dict]:
        """Full-batch optimization over ``pairs`` for ``epochs`` passes.
        Returns the per-step metrics list."""
        if not pairs:
            raise ValueError("DPOTrainer.fit: empty pair list")
        history = []
        for ep in range(epochs):
            self.opt.zero_grad()
            loss, metrics = self.loss(pairs)
            loss.backward()
            self.opt.step()
            history.append(metrics)
            if verbose:
                print(f"  dpo ep{ep} loss {metrics['loss']:.4f} "
                      f"acc {metrics['acc']:.2f} "
                      f"margin {metrics['margin']:.3f}", flush=True)
        return history
