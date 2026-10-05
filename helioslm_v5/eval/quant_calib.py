"""Calibration × quantization cross experiment (Phase 2.3).

The unique positioning no frontier lab occupies: measure what NVFP4-style
quantization does to CALIBRATION, not just accuracy. The runner scores an
N-way multiple-choice task by choice loglikelihood (no generation, no
reference LLM — constructed ground truth), turns choice softmax into a
confidence, and reports ECE/accuracy for a bf16 model and an NVFP4
fake-quantized twin of the SAME weights. The drift between the two is the
quantified claim.

Honesty constraints:
  - ECE on an UNTRAINED model is mechanics, not science. The oracle suite
    validates the measurement with controlled stubs (a perfectly
    calibrated stub must give ECE ~= 0; an overconfident stub must give
    ECE > 0). Real drift claims require a trained checkpoint.
  - Fake-quant (QAT-style QDQ) stands in for true NVFP4 kernels; the
    deviation is recorded in every report (weights on the E2M1x16 grid,
    block E4M3 scales — finer than spec in two documented ways).
"""
from typing import Dict, List, Optional

import torch


def choice_logprob(model, prompt_ids: List[int],
                   choice_ids) -> float:
    """log P(choice | prompt) for a choice given as a token SEQUENCE
    (a single int is treated as a one-token sequence). Position-faithful:
    each choice token is scored conditioned on the prompt AND the choice
    tokens before it.
    """
    ch = [choice_ids] if isinstance(choice_ids, int) \
        else list(choice_ids)
    full = list(prompt_ids) + ch
    with torch.no_grad():
        out = model(torch.tensor([full]))
        logits = out[0] if isinstance(out, tuple) else out
        seg = torch.log_softmax(
            logits[0, len(prompt_ids) - 1:len(full) - 1].float(), dim=-1)
        tgt = torch.tensor(full[len(prompt_ids):])
    return float(seg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())


def choice_confidence(model, prompt_ids: List[int],
                      choice_ids: List) -> tuple:
    """(predicted choice index, confidence, correct?) is the caller's;
    here: softmax over per-choice sequence log-probs -> (probs, logps)."""
    logps = torch.tensor([choice_logprob(model, prompt_ids, c)
                          for c in choice_ids])
    probs = torch.softmax(logps, dim=-1)
    return probs, logps


def ece(confidences: List[float], correct: List[int],
        n_bins: int = 10) -> float:
    """Standard equal-width ECE. Lists align: confidences[i] is the
    model's confidence in its PREDICTED choice; correct[i] is 0/1."""
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("ece: empty or mismatched inputs")
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        e += (len(idx) / len(confidences)) * abs(acc - conf)
    return e


def run_quant_calib_probe(model, tasks, n_bins: int = 10,
                          tag: str = "unknown") -> Dict:
    """tasks: [{prompt_ids, choice_ids, answer_idx}]. Returns accuracy,
    ECE, and provenance. Read-only on the model."""
    confs, corrects = [], []
    for t in tasks:
        probs, _ = choice_confidence(model, t["prompt_ids"],
                                     t["choice_ids"])
        pick = int(probs.argmax())
        confs.append(float(probs[pick]))
        corrects.append(int(pick == t["answer_idx"]))
    acc = sum(corrects) / len(corrects)
    return {"tag": tag, "accuracy": acc,
            "ece": ece(confs, corrects, n_bins),
            "n": len(tasks)}


def compare_bf16_vs_nvfp4(model, tasks, n_bins: int = 10,
                          model_tag: str = "unknown") -> Dict:
    """Drift report: same weights, bf16 vs NVFP4 fake-quant inference.
    The model is cloned first — the caller's model is never mutated."""
    import copy
    from helioslm_v5.src.quantization.qat import apply_qat

    base = run_quant_calib_probe(model, tasks, n_bins,
                                 tag=f"{model_tag}/bf16")
    twin = copy.deepcopy(model)
    applied = apply_qat(twin, method="nvfp4")
    if not applied:
        raise ValueError("apply_qat wrapped zero Linear modules — "
                         "nothing to quantize, the comparison is void")
    quant = run_quant_calib_probe(twin, tasks, n_bins,
                                  tag=f"{model_tag}/nvfp4-fakequant")
    return {
        "model": model_tag,
        "method": "nvfp4-fakequant (E2M1x16 + E4M3 block scales; "
                  "QAT-style QDQ, not a true FP4 kernel)",
        "bf16": base, "nvfp4": quant,
        "drift": {"accuracy": quant["accuracy"] - base["accuracy"],
                  "ece": quant["ece"] - base["ece"]},
    }
