"""HeliosLM v5 test cases — model security: observability-driven probes.

Split out of the monolithic test_v5.py layout on 2026-10-11; the
test_v5.py runner imports every case module and executes the tests in
the historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.

Theme (skills/model-security-audit): the attack surface is defined by
observability, not by attack names. These probes pin down what each
HeliosLM surface actually reveals per query — full logits (extraction),
scalar NLL (membership), fingerprint vectors (model identity), and
cross-tenant prefix-cache state (nothing must flow).
"""

import torch

from evastar.tests.case_common import (_lite_model, _pass)


# ----------------------------------------------------------------------
# Model fingerprint: probe-set logits identify EXACT weights — bitwise
# stable across reloads, moved by any perturbation, far across seeds
# ----------------------------------------------------------------------
def test_model_fingerprint():
    model, config = _lite_model(seed=401)

    def fingerprint(m):
        g = torch.Generator().manual_seed(99173)  # fixed probe set
        probes = torch.randint(0, config.vocab_size, (4, 8), generator=g)
        with torch.no_grad():
            lg, _, _ = m(probes)
        return lg[:, -1, :].flatten()  # [4*V]

    fp1 = fingerprint(model)

    # Same weights in a fresh instance: bitwise-identical fingerprint.
    model2, _ = _lite_model(seed=402)
    model2.load_state_dict(model.state_dict())
    model2.eval()
    d_same = (fp1 - fingerprint(model2)).abs().max().item()
    assert d_same == 0.0, (
        f"identical weights must give a bitwise fingerprint: {d_same:.3e}")

    # Any weight perturbation moves the fingerprint (clone detection).
    # Perturb the LM head: it feeds the logits of EVERY probe position,
    # so the edit cannot be masked by probe-content chance.
    model3, _ = _lite_model(seed=403)
    model3.load_state_dict(model.state_dict())
    model3.eval()
    with torch.no_grad():
        model3.lm_head.weight.view(-1)[0].add_(1e-3)
    d_perturb = (fp1 - fingerprint(model3)).abs().max().item()
    assert d_perturb > 0.0, "a 1e-3 weight edit must move the fingerprint"

    # A different-seed model is far away in cosine distance (random
    # high-dimensional vectors sit near 0; 0.9 is a wide margin).
    model4, _ = _lite_model(seed=404)
    cos = torch.nn.functional.cosine_similarity(
        fp1, fingerprint(model4), dim=0).item()
    assert abs(cos) < 0.9, f"different-seed fingerprint too close: {cos:.4f}"
    _pass("test_model_fingerprint",
          f"reload bitwise-exact, 1e-3 edit moves {d_perturb:.2e}, "
          f"cross-seed cos {cos:.4f}")


# ----------------------------------------------------------------------
# Membership inference: scalar NLL ALONE separates a memorized sequence
# from a fresh one — aggregated scores still leak membership
# ----------------------------------------------------------------------
def test_membership_inference_signal():
    from evastar.eval.harness import loglikelihood

    model, config = _lite_model(seed=411)
    vocab = config.vocab_size
    g = torch.Generator().manual_seed(8701)
    member = torch.randint(0, vocab, (17,), generator=g).tolist()
    nonmember = torch.randint(0, vocab, (17,), generator=g).tolist()

    def nll(seq):
        return -loglikelihood(model, seq[:1], seq[1:]) / (len(seq) - 1)

    nll_mem_pre = nll(member)
    nll_non_pre = nll(nonmember)

    # Memorize the member sequence (200 steps on 16 tokens).
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = torch.tensor([member])
    for _ in range(200):
        logits, _, _ = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1], ids[0, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    nll_mem = nll(member)
    nll_non = nll(nonmember)
    assert nll_mem < 1.0, f"member not memorized: nll {nll_mem:.3f}"
    assert nll_non > 3.0, f"nonmember drifted: nll {nll_non:.3f}"
    assert nll_non > 3.0 * nll_mem, (
        f"member/nonmember gap too small: {nll_mem:.3f} vs {nll_non:.3f}")
    _pass("test_membership_inference_signal",
          f"member nll {nll_mem_pre:.2f}->{nll_mem:.3f} vs nonmember "
          f"{nll_non_pre:.2f}->{nll_non:.2f} (scalar NLL leaks membership)")


# ----------------------------------------------------------------------
# Distillation leak budget: at equal query budget, a student trained on
# full soft logits gets strictly closer to the teacher than one trained
# on argmax hard labels — full-logit endpoints are the extraction surface
# ----------------------------------------------------------------------
def test_distillation_leak_budget():
    teacher, config = _lite_model(seed=421)
    teacher.eval()
    vocab = config.vocab_size
    g = torch.Generator().manual_seed(9021)
    queries = torch.randint(0, vocab, (24, 8), generator=g)
    T = 0.5  # sharpen the near-uniform untrained teacher

    with torch.no_grad():
        t_logits, _, _ = teacher(queries)
    t_soft = torch.softmax(t_logits / T, dim=-1)
    t_hard = t_logits.argmax(-1)

    def dist_to_teacher(student):
        # Mean L1 between the student's and teacher's sharpened
        # distributions — bounded and symmetric in effect, unlike KL(t||s)
        # which explodes when a hard-trained student concentrates mass.
        with torch.no_grad():
            s_logits, _, _ = student(queries)
        s = torch.softmax(s_logits / T, dim=-1)
        return (s - t_soft).abs().sum(-1).mean().item()

    def top1_agreement(student):
        with torch.no_grad():
            s_logits, _, _ = student(queries)
        return (s_logits.argmax(-1) == t_hard).float().mean().item()

    def train(student, soft):
        student.train()
        opt = torch.optim.AdamW(student.parameters(), lr=1e-2)
        for _ in range(150):
            s_logits, _, _ = student(queries)
            if soft:
                loss = torch.nn.functional.kl_div(
                    torch.log_softmax(s_logits / T, dim=-1), t_soft,
                    reduction="batchmean")
            else:
                loss = torch.nn.functional.cross_entropy(
                    s_logits.reshape(-1, vocab), t_hard.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        student.eval()

    # Identical student init in both arms: only the label density differs.
    s_soft, _ = _lite_model(seed=777)
    s_hard, _ = _lite_model(seed=777)
    d0 = dist_to_teacher(s_soft)
    train(s_soft, soft=True)
    train(s_hard, soft=False)
    d_soft = dist_to_teacher(s_soft)
    d_hard = dist_to_teacher(s_hard)
    agree_hard = top1_agreement(s_hard)

    assert d_soft < d0, f"soft student did not approach teacher: {d_soft:.3f}"
    assert agree_hard > 0.9, (
        f"hard student did not learn the argmax mapping: {agree_hard:.3f}")
    assert d_soft < d_hard, (
        f"soft labels must leak more distribution per query: "
        f"soft {d_soft:.3f} vs hard {d_hard:.3f}")
    _pass("test_distillation_leak_budget",
          f"L1 {d0:.3f} -> soft {d_soft:.3f} < hard {d_hard:.3f}, "
          f"hard top-1 agreement {agree_hard:.2f} (equal query budget)")


# ----------------------------------------------------------------------
# Prefix-pool tenant isolation: different fingerprints never serve each
# other's KV (zero cross-tenant hits), while every answer stays bitwise
# equal to from-scratch generation
# ----------------------------------------------------------------------
def test_prefix_pool_tenant_isolation():
    from evastar.src.inference.prefix_pool import PrefixPool

    model, config = _lite_model(seed=431)
    prompt = [10, 20, 30, 40, 50, 60, 70, 80] * 2  # 16 tokens = 2 blocks
    pool_a = PrefixPool(model, block_size=8, max_blocks=8,
                        fingerprint="tenant/a")
    pool_b = PrefixPool(model, block_size=8, max_blocks=8,
                        fingerprint="tenant/b")
    ref = model.generate(torch.tensor([prompt]), max_new_tokens=6,
                         temperature=0)

    # Tenant A stores; tenant B must MISS on the exact same tokens —
    # no cross-tenant KV flow, ever.
    pool_a.store(prompt)
    assert pool_a.stats()["hits"] == 0
    out_b = pool_b.generate(prompt, max_new_tokens=6, temperature=0)
    st_b = pool_b.stats()
    assert st_b["hits"] == 0 and st_b["hit_tokens"] == 0, (
        f"tenant B saw tenant A state: {st_b}")
    assert st_b["misses"] >= 1
    assert torch.equal(out_b, ref), "tenant B output must be from-scratch"

    # Tenant A hits on its own content and is still bitwise-exact.
    out_a = pool_a.generate(prompt, max_new_tokens=6, temperature=0)
    assert pool_a.stats()["hits"] >= 1
    assert torch.equal(out_a, ref)
    _pass("test_prefix_pool_tenant_isolation",
          f"cross-tenant hits 0, misses {st_b['misses']}, outputs "
          "bitwise equal to from-scratch")


# ----------------------------------------------------------------------
# Scorer leak budget: score_record emits exactly four rounded scalars —
# no per-token logprob vector. The schema IS the information budget, and
# the rounding is part of it, not cosmetics
# ----------------------------------------------------------------------
def test_score_stream_leak_budget():
    from evastar.eval.score_stream import score_record

    model, config = _lite_model(seed=441)
    rec = score_record(model, "HeliosLM", "security audit probe",
                       config.vocab_size)
    assert set(rec) == {"sum_logprob", "n_tokens", "nll_per_token", "ppl"}, (
        f"score schema widened: {sorted(rec)}")
    assert all(isinstance(v, (int, float)) for v in rec.values())
    assert not any(hasattr(v, "__len__") for v in rec.values()), (
        "per-token vectors must never appear in the score schema")
    assert rec["sum_logprob"] == round(rec["sum_logprob"], 4)
    assert rec["nll_per_token"] == round(rec["nll_per_token"], 4)
    assert rec["n_tokens"] == len("security audit probe")
    _pass("test_score_stream_leak_budget",
          "schema == 4 rounded scalars, no per-token vector")
