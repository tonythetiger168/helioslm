"""T113 — roadmap AC4: repo-scale long-context probe, v5.74.

Pins:

1. the task: every query's answer is the value of the file that defines
   that name, names are distinct within a repository, the value position
   is right after ``USE name``, the distance bound holds, and the streams
   are seed-deterministic;
2. architecture oracles on untrained models: every arm is causal
   (changing a later token never changes earlier logits); the all-SWA arm
   cannot see past its stacked receptive field (n_layers * (window - 1)
   tokens) — changing a token farther back leaves the query logits
   bit-identical — while dense and hybrid do see it (CSA's dense
   equivalence at ratio 1 is already pinned by T92);
3. query_loss_and_acc reads the logits at p - 1;
4. the committed artifact: the hybrid solves every size; all-SWA (the
   structural negative control) falls with the file count to near chance;
   no arm is below chance; the 4000-step dense supplement learns the task
   well beyond its 2000-step row (dense converges slower here); hashes.

Run from repo root: python3 evastar/tests/test_repo_longctx.py
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval.repo_longctx import (  # noqa: E402
    ARMS, DEF, EQ, FILE, N_VALUES, NAME0, USE, VAL0, RepoProbeLM, make_batch,
    make_repo, max_query_distance, query_loss_and_acc)

_ART = _REPO / "benchmarks" / "repo_longctx_2026-10-19.json"


def test_task_generator():
    rng = random.Random(3)
    for _ in range(50):
        ids, pos, ans = make_repo(8, 4, 6, rng)
        defs = {}
        for i, t in enumerate(ids):
            if t == DEF:
                assert ids[i + 2] == EQ
                assert ids[i + 1] not in defs, "duplicate name"
                defs[ids[i + 1]] = ids[i + 3]
        assert ids.count(FILE) == 8 and len(defs) == 8
        for p, a in zip(pos, ans):
            assert ids[p - 2] == USE and ids[p] == a
            assert defs[ids[p - 1]] == a
            first_def = ids.index(DEF)
            assert p - first_def <= max_query_distance(8, 6) + 3 * 4
        assert all(VAL0 <= a < VAL0 + N_VALUES for a in ans)
        assert all(t >= 0 for t in ids)
    a = make_batch(4, 8, 4, 6, random.Random(0))
    b = make_batch(4, 8, 4, 6, random.Random(0))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    try:
        make_repo(40, 4, 6, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("too many files accepted")
    print("PASS test_task_generator")


def _model(arm, **kw):
    torch.manual_seed(0)
    return RepoProbeLM(arm, dim=32, heads=2, n_layers=4, **kw).eval()


def test_causality_all_arms():
    ids = make_batch(2, 6, 2, 4, random.Random(1))[0]
    cut = ids.shape[1] // 2
    alt = ids.clone()
    alt[:, cut:] = (alt[:, cut:] + 7) % 120 + 1
    for arm in ARMS:
        m = _model(arm)
        with torch.no_grad():
            a, b = m(ids), m(alt)
        assert torch.allclose(a[:, :cut], b[:, :cut], atol=1e-5), arm
        assert torch.isfinite(a).all(), arm
    print("PASS test_causality_all_arms")


def test_receptive_field():
    window, layers = 8, 4
    reach = layers * (window - 1)          # 28 tokens back
    ids = make_batch(1, 8, 2, 6, random.Random(2))[0]
    q = ids.shape[1] - 1
    far = q - reach - 3                    # strictly outside the reach
    alt = ids.clone()
    alt[0, far] = (alt[0, far] + 11) % 120 + 1
    seen = {}
    for arm in ("all_swa", "dense", "hybrid"):
        m = _model(arm, window=window, global_every=2)
        with torch.no_grad():
            seen[arm] = (m(ids)[0, q] - m(alt)[0, q]).abs().max().item()
    assert seen["all_swa"] == 0.0, seen
    assert seen["dense"] > 1e-6 and seen["hybrid"] > 1e-6, seen
    print(f"PASS test_receptive_field {seen}")


def test_query_loss_reads_previous_logit():
    class Echo(torch.nn.Module):
        def forward(self, ids):
            # predicts token t+1 = ids[t] + 1 exactly
            out = torch.full((*ids.shape, 128), -1e4)
            out.scatter_(-1, ((ids + 1) % 128).unsqueeze(-1), 0.0)
            return out
    ids = torch.tensor([[5, 6, 7, 8]])
    pos = torch.tensor([[2, 3]])
    ans = torch.tensor([[7, 8]])
    loss, acc = query_loss_and_acc(Echo(), ids, pos, ans)
    assert acc == 1.0 and loss.item() < 1e-3
    print("PASS test_query_loss_reads_previous_logit")


def test_ab_artifact():
    art = json.loads(_ART.read_text())
    res = {r["arm"]: r["acc_by_files"] for r in art["results"]}
    assert set(res) == set(ARMS)
    n = art["task"]["train_files"]
    chance = art["task"]["chance"]
    sizes = [str(n * m) for m in (1, 2, 4)]
    assert all(list(v) == sizes for v in res.values())
    # the sliding/global hybrid finds the definition at every size
    assert min(res["hybrid"].values()) > 0.95, res["hybrid"]
    # all-SWA only sees the last few files: it falls with the file count
    # and is near chance at 4x
    swa = [res["all_swa"][k] for k in sizes]
    assert swa[0] > swa[1] > swa[2] and swa[2] < chance + 0.1, swa
    # every arm is at least at chance (no degenerate run)
    assert all(a >= chance - 0.03 for v in res.values() for a in v.values())
    # the longer dense run (same settings, 4000 steps) learns the task
    sup = art["supplement"]["dense_4000_steps"]
    assert sup["acc_by_files"][sizes[0]] > res["dense"][sizes[0]] + 0.3, sup
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(art["results"], sort_keys=True).encode()).hexdigest()
    assert art["supplement"]["sha256"] == hashlib.sha256(
        json.dumps(sup, sort_keys=True).encode()).hexdigest()
    print("PASS test_ab_artifact " + json.dumps(res))

if __name__ == "__main__":
    test_task_generator()
    test_causality_all_arms()
    test_receptive_field()
    test_query_loss_reads_previous_logit()
    test_ab_artifact()
