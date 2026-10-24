"""T121 — distillation from an API teacher (v5.82).

Pins:

1. alignment oracles: with top-k covering every teacher token, the
   first-character marginal and the next-character conditional recovered
   from 2-character teacher tokens equal the teacher's own distributions
   exactly; with a partial top-k they are the renormalised restriction;
   an uncovered prefix gives None;
2. sparse_kd_loss equals cross-entropy for one-hot targets and the
   hand-written soft cross-entropy (value and gradient) otherwise;
3. collect_sequences keeps only verified samples and reports the rate;
4. the simulated top-k API on a real model: with k = all tokens, the
   first-character targets equal the exact joint p(a) * P(second char in
   the alphabet | a), renormalised, and the listed mass equals that
   joint's total; token ids outside the alphabet round-trip;
5. the committed probe artifact (3 student seeds): a verifier beats
   training on every teacher sample in every seed; filtering top-k paths
   never hurts; soft top-k targets beat raw samples on average; the
   ground-truth arm's seed spread is recorded rather than generalised.

Run from repo root: python3 evastar/tests/test_api_distill.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.src.training.api_distill import (  # noqa: E402
    collect_sequences, first_char_targets, next_char_targets, sparse_kd_loss)

_ART = _REPO / "benchmarks" / "api_distill_probe_2026-10-20.json"


def _toy_teacher():
    """A char model over 'abc': p(first), p(second | first)."""
    p1 = {"a": 0.5, "b": 0.3, "c": 0.2}
    p2 = {"a": {"a": 0.1, "b": 0.6, "c": 0.3},
          "b": {"a": 0.7, "b": 0.2, "c": 0.1},
          "c": {"a": 0.25, "b": 0.25, "c": 0.5}}
    toks = [(x + y, math.log(p1[x] * p2[x][y])) for x in p1 for y in p2[x]]
    return p1, p2, sorted(toks, key=lambda t: -t[1])


def test_alignment_oracles():
    p1, p2, toks = _toy_teacher()
    first = first_char_targets(toks)
    assert all(abs(first[c] - p1[c]) < 1e-12 for c in p1)
    for x in p1:
        nxt = next_char_targets(toks, x)
        assert all(abs(nxt[y] - p2[x][y]) < 1e-12 for y in p2[x])
    k3 = toks[:3]                       # partial top-k: renormalised
    first3 = first_char_targets(k3)
    mass = {}
    for t, lp in k3:
        mass[t[0]] = mass.get(t[0], 0) + math.exp(lp)
    z = sum(mass.values())
    assert all(abs(first3[c] - mass[c] / z) < 1e-12 for c in mass)
    assert abs(sum(first3.values()) - 1) < 1e-12
    assert next_char_targets(k3, "z") is None
    try:
        first_char_targets([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty top-k accepted")
    print("PASS test_alignment_oracles")


def test_sparse_kd_loss():
    torch.manual_seed(0)
    logits = torch.randn(4, 7, requires_grad=True)
    hard = [{2: 1.0}, {0: 1.0}, {6: 1.0}, {3: 1.0}]
    a = sparse_kd_loss(logits, hard)
    b = F.cross_entropy(logits, torch.tensor([2, 0, 6, 3]))
    assert torch.allclose(a, b, atol=1e-6)
    soft = [{1: 0.25, 2: 0.75}, {0: 1.0}, {5: 0.5, 6: 0.5}, {3: 0.1, 4: 0.9}]
    q = torch.zeros(4, 7)
    for i, t in enumerate(soft):
        for k, v in t.items():
            q[i, k] = v
    ref = -(q * torch.log_softmax(logits, -1)).sum(-1).mean()
    got = sparse_kd_loss(logits, soft)
    assert torch.allclose(got, ref, atol=1e-6)
    g1 = torch.autograd.grad(got, logits)[0]
    g2 = torch.autograd.grad(ref, logits)[0]
    assert torch.allclose(g1, g2, atol=1e-6)
    try:
        sparse_kd_loss(logits, soft[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("row/target mismatch accepted")
    print("PASS test_sparse_kd_loss")


def test_collect_sequences():
    answers = iter(["1", "2", "3", "4"])
    pairs, st = collect_sequences([1, 2, 3, 4], lambda p: next(answers),
                                  lambda p, r: int(r) % 2 == 0)
    assert pairs == [(2, "2"), (4, "4")]
    assert st == {"queries": 4, "kept": 2, "accept_rate": 0.5}
    pairs, st = collect_sequences([1], lambda p: "x")
    assert pairs == [(1, "x")] and st["accept_rate"] == 1.0
    print("PASS test_collect_sequences")


def test_topk_api_on_model():
    from evastar.eval import api_distill_probe as ap
    from evastar.eval.mopd_probe import EvastarV5, config
    torch.manual_seed(3)
    model = EvastarV5(config()).eval()
    prompt = ap.prompt_pool(1, 0)[0][0]
    full = ap.TopKAPI(model, k=10_000)
    toks = full.topk(prompt, "")
    # every 1-char "$" token plus 10 x 11 two-char tokens
    assert len(toks) == 1 + 10 * 11
    # exact reference: a 2-char token "ab" has mass p(a) p(b | a) over the
    # alphabet ("$" ends a token alone), so the first-char target of a is
    # p(a) * P(second char in alphabet | a), renormalised
    alpha = torch.tensor([ap.char_id(c) for c in ap.ALPHABET])

    def probs(ids):
        with torch.no_grad():
            return torch.softmax(model(ids.unsqueeze(0))[0][0, -1].float(), -1)
    p1 = probs(prompt)
    ref = {}
    for c in ap.ALPHABET:
        pa = p1[ap.char_id(c)].item()
        if c == "$":
            ref[c] = pa
        else:
            nxt = torch.cat([prompt, torch.tensor([ap.char_id(c)])])
            ref[c] = pa * probs(nxt)[alpha].sum().item()
    z = sum(ref.values())
    first = ap.first_char_targets(toks)
    assert set(first) == set(ap.ALPHABET)
    assert all(abs(first[c] - ref[c] / z) < 1e-5 for c in ap.ALPHABET)
    assert abs(sum(math.exp(x) for _, x in toks) - z) < 1e-5
    for i in (0, 5, 17, 30, ap.EOS):
        assert ap.char_id(ap.id_char(i)) == i
    print("PASS test_topk_api_on_model")


def test_probe_artifact():
    art = json.loads(_ART.read_text())
    res = {r["arm"]: r for r in art["results"]}
    t = art["teacher"]
    assert 0.4 < t["greedy_exact_match"] < 0.8          # imperfect teacher
    assert t["sample_accept_rate"] < t["greedy_path_accept_rate"]
    seeds = art["train"]["student_seeds"]
    assert len(seeds) >= 3
    by = {a: r["exact_match_by_seed"] for a, r in res.items()}
    mean = {a: r["exact_match_mean"] for a, r in res.items()}
    # robust claims, in EVERY seed:
    for i in range(len(seeds)):
        # a verifier beats training on everything the teacher said
        assert by["seq_kd_verified"][i] > by["seq_kd"][i] + 0.3
        # filtering top-k paths never hurts
        assert by["topk_kd_verified"][i] >= by["topk_kd"][i]
    # on average soft top-k targets beat raw samples, and ground truth
    # beats raw teacher samples
    assert mean["topk_kd"] > mean["seq_kd"]
    assert mean["ground_truth"] > mean["seq_kd"]
    # recorded, not generalised: the from-scratch ground-truth arm is
    # unstable across seeds (it straddles the task's accuracy jump)
    assert max(by["ground_truth"]) - min(by["ground_truth"]) > 0.1
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(art["results"], sort_keys=True).encode()).hexdigest()
    print("PASS test_probe_artifact " + json.dumps(mean))


if __name__ == "__main__":
    test_alignment_oracles()
    test_sparse_kd_loss()
    test_collect_sequences()
    test_topk_api_on_model()
    test_probe_artifact()
