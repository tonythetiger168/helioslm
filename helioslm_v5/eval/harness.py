"""HeliosLM evaluation harness (v5.11).

A minimal, dependency-free log-likelihood harness in the spirit of
lm-evaluation-harness, operating directly on token ids (HeliosLM ships no
tokenizer — tasks provide token-id sequences).

Core primitives:
  - ``loglikelihood(model, context_ids, continuation_ids)`` — sum logprob of
    the continuation under the model, teacher-forced in one forward pass.
  - ``multiple_choice`` — score each choice and pick the argmax.
  - ``run_harness(model, tasks)`` — run a dict of built-in tasks and return
    ``{task_name: {"acc": float, "n": int}}``.

Interface contract (so external harnesses can adapt HeliosLM):
  ``model.forward(ids)`` must return ``(logits, _, _)`` with logits
  ``[B, L, vocab]`` — exactly ``HeliosLMv5.forward``'s contract. To plug
  HeliosLM into lm-evaluation-harness, wrap a forward call in a class with
  ``loglikelihood(context, continuation)`` delegating here.

CLI:
  python -m helioslm_v5.eval.harness --task copy_vs_reverse --size lite
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


# --------------------------------------------------------------------------
# Scoring primitives
# --------------------------------------------------------------------------

@torch.no_grad()
def loglikelihood(model, context_ids, continuation_ids) -> float:
    """Sum log-probability of ``continuation_ids`` given ``context_ids``.

    Single teacher-forced forward pass; position t of the continuation is
    scored against logits at the previous position (standard LM scoring).
    """
    ctx = list(context_ids)
    cont = list(continuation_ids)
    if len(ctx) == 0:
        raise ValueError("context must be non-empty (at least one token)")
    if len(cont) == 0:
        return 0.0
    ids = torch.tensor([ctx + cont], dtype=torch.long)
    logits, _, _ = model(ids)
    logprobs = torch.log_softmax(logits[0, :len(ctx) + len(cont) - 1].float(),
                                 dim=-1)
    total = 0.0
    for i, tok in enumerate(cont):
        total += logprobs[len(ctx) - 1 + i, tok].item()
    return total


@torch.no_grad()
def multiple_choice(model, context_ids, choice_ids_list) -> int:
    """Return the index of the highest-scoring choice."""
    scores = [loglikelihood(model, context_ids, ch) for ch in choice_ids_list]
    return max(range(len(scores)), key=lambda i: scores[i]), scores


# --------------------------------------------------------------------------
# Built-in tasks (token-id based; vocab is model-relative)
# --------------------------------------------------------------------------

def _copy_vs_reverse(n_samples=32, seq_len=6, vocab=1000, seed=0):
    """Synthetic probing task: given a random sequence, choose between the
    exact copy, the reversal, and a random distractor. An untrained model
    scores at chance (~1/3); the harness plumbing is what is under test.
    """
    g = torch.Generator().manual_seed(seed)
    docs = []
    for _ in range(n_samples):
        seq = torch.randint(3, vocab, (seq_len,), generator=g).tolist()
        distractor = torch.randint(3, vocab, (seq_len,), generator=g).tolist()
        if distractor == seq:
            distractor[0] = (distractor[0] + 1) % vocab
        docs.append({
            "context": seq,
            "choices": [seq, seq[::-1], distractor],
            "gold": 0,
        })
    return docs


TASKS = {
    "copy_vs_reverse": _copy_vs_reverse,
}


def run_harness(model, task_docs: dict) -> dict:
    """Evaluate ``model`` on ``{task_name: [docs]}``.

    Each doc: {"context": [ids], "choices": [[ids]...], "gold": int}.
    Returns ``{task_name: {"acc": float, "n": int}}``.
    """
    model.eval()
    report = {}
    for name, docs in task_docs.items():
        correct = 0
        for doc in docs:
            pred, _ = multiple_choice(model, doc["context"], doc["choices"])
            correct += int(pred == doc["gold"])
        report[name] = {"acc": correct / max(len(docs), 1), "n": len(docs)}
    return report


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="copy_vs_reverse", choices=sorted(TASKS))
    ap.add_argument("--size", default="lite", choices=["lite", "full"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    from helioslm_v5.src.model_v5 import HeliosLMv5

    cfg = HeliosLMv5Config(size=args.size)
    model = HeliosLMv5(cfg)
    model.eval()

    docs = TASKS[args.task](seed=args.seed)
    report = run_harness(model, {args.task: docs})
    for name, r in report.items():
        print(f"{name}: acc={r['acc']:.3f} ({r['n']} docs)")


if __name__ == "__main__":
    main()
