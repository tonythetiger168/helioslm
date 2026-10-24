"""nvfp4_qat_smoke.py — B1-3: NVFP4 QAT end-to-end training smoke at toy
scale, with a QAT→deploy parity oracle and a bitwise gate.

WHY THIS EXISTS
---------------
``evastar/src/quantization/qat.py`` has had a full NVFP4 QAT target
since v5.22-L (E2M1 values x 16-wide blocks, E4M3 block scales under a
per-row fp32 scale, STE gradients into fp32 master weights), and
``tests/case_quant_fused.py::test_nvfp4_qat`` pins the math in isolation.
What was missing is the END-TO-END evidence that NVFP4 QAT actually
trains a model from scratch and that the trained artifact survives
contact with a deploy path — the NVIDIA NVFP4 hardware route (Blackwell)
this codebase aligns with. This script closes that gap at toy scale.

WHAT THIS MEASURES
------------------
Three arms share one corpus, one validation protocol, one seed, and one
data order (the train_toy_checkpoint.py recipe: Adam + cosine, MTP aux
loss), so the only moving part is WHERE the NVFP4 grid enters:

  1. ARM fp32_scratch: EvastarV5(lite) trained from scratch, evaluated
     in fp32 — the quality reference.
  2. ARM ptq_rtn_nvfp4: that same fp32-trained model, every nn.Linear
     weight snapped onto the NVFP4 grid by pure round-to-nearest
     (``_qdq_nvfp4``, zero extra training) — what naive post-training
     quantization deploys.
  3. ARM qat_nvfp4: a fresh model wrapped by ``apply_qat(method=
     "nvfp4")`` BEFORE step 0, so the grid is in the training loop from
     the start (STE gradients reach fp32 master weights). After training,
     the fake-quant weights are BAKED into plain nn.Linear modules — the
     deployable artifact. Because ``_qdq_nvfp4``'s dequantized values are
     exactly what an NVFP4 kernel reconstructs, the baked model IS the
     deploy model; no serving-side module is required.

Three correctness oracles bracket the quality numbers:

  * DEPLOY PARITY: val loss through the STE wrappers (training-time
    forward) vs the baked plain-Linear model must be BITWISE equal —
    the bake drops no information. (This oracle caught a real bug during
    B1-3: MLA's absorbed decode path sliced ``kv_b_proj.weight``
    directly, bypassing the fake-quant grid — fixed via
    ``FakeQuantLinear.ste_weight()``; see v5.50 changelog.)
  * BITWISE GATE (HarnessEvolver discipline): the QAT arm run TWICE with
    the same seed must give an identical baked val loss, identical final
    train loss, and a token-identical greedy probe.
  * CONVERGENCE: the QAT deploy val loss must land within
    ``--qat-tol`` of the fp32 arm's (training-aware quantization should
    converge to the fp32 neighborhood at fixed budget), and QAT training
    loss must descend (first-10 vs last-10 mean).

The PTQ-vs-QAT val-loss delta is RECORDED but not gated: at toy scale
NVFP4 RTN is nearly free on smooth converged weights (16-wide blocks +
E4M3 scales are a fine grid), so there is little for QAT to recover —
the recovery gap matters at production scale, not here.

DELIBERATE SCOPE LIMITS (recorded, not hidden)
----------------------------------------------
- Toy scale (8.5M params, char corpus built from repo docs): a
  training-loop and deploy-parity smoke, not a production quality claim.
  The NVFP4 vs MXFP4 RTN-error oracle lives in case_quant_fused.py.
- A checkpoint-fine-tune variant was explored and rejected: a fresh
  optimizer on the converged toy checkpoint degrades val loss for fp32
  and QAT arms alike (optimizer-restart pathology, not quantization),
  and RTN on the converged checkpoint costs only ~+0.002 val nats —
  there is nothing to recover. From-scratch arms keep the comparison
  clean (same convention as mla_every_sweep.py).
- Training determinism holds for a fixed machine + torch build + thread
  count (same convention as hca_ab_passkey.py / mla_every_sweep.py);
  the bitwise gate verifies it in-process.
- The corpus is rebuilt from the repo's current *.py + *.md, so absolute
  val losses shift as the repo evolves; every arm inside one run shares
  one corpus snapshot, which is what the comparisons need.

BITWISE DISCIPLINE
------------------
All randomness flows through torch.manual_seed / torch.Generator with
fixed seeds; ``results_sha256`` covers the canonical JSON of the
results table ONLY (wall-clock telemetry is excluded, same convention
as mla_every_sweep.py).

Usage:
  python examples/nvfp4_qat_smoke.py                     # full smoke
  python examples/nvfp4_qat_smoke.py --steps 60          # quick pass
  python examples/nvfp4_qat_smoke.py --skip-gate         # single QAT run
  python examples/nvfp4_qat_smoke.py --output benchmarks/nvfp4_qat_2026-10-13.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from evastar.configs.config_v5 import EvastarV5Config  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402
from evastar.src.quantization.qat import (  # noqa: E402
    FakeQuantLinear, _qdq_nvfp4, apply_qat)
from train_toy_checkpoint import build_corpus, encode  # noqa: E402

EXPERIMENT = "B1-3 NVFP4 QAT end-to-end smoke"
SCORING_TAG = "toy-char-lm val loss: NVFP4 QAT-from-scratch vs fp32 vs RTN-PTQ"
PROBE_VERSION = 2
NVFP4_BLOCK = 16           # NVFP4 spec block width


def stable_corpus(vocab_size, attempts=5):
    """build_corpus() with a stability check: two consecutive identical
    reads. Guards the artifact against transient filesystem read skips
    (build_corpus silently drops files on OSError)."""
    prev = None
    for _ in range(attempts):
        ids = torch.tensor(encode(build_corpus(), vocab_size),
                           dtype=torch.long)
        digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        if digest == prev:
            return ids
        prev = digest
    raise RuntimeError("corpus read unstable after "
                       f"{attempts} attempts")


def val_window(ids, seed):
    """Fixed held-out windows: seed+999 generator (train_toy_checkpoint
    convention), 4x512 concatenated, first 256 tokens."""
    vgen = torch.Generator().manual_seed(seed + 999)
    starts = torch.randint(0, len(ids) - 513, (4,), generator=vgen)
    return torch.cat([ids[i:i + 512] for i in starts])[:256]


@torch.no_grad()
def val_loss_of(model, window):
    model.eval()
    vx = window.unsqueeze(0)[:, :-1]
    vy = window.unsqueeze(0)[:, 1:]
    logits, _, _ = model(vx)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), vy.reshape(-1)).item()


@torch.no_grad()
def probe_of(model, cfg, prompt, tokens):
    """Greedy continuation token ids — the bitwise-gate subject."""
    model.eval()
    pids = torch.tensor([encode(prompt, cfg.vocab_size)])
    return model.generate(pids, max_new_tokens=tokens,
                          temperature=0)[0].tolist()


def rtn_deploy(model, block=NVFP4_BLOCK):
    """In-place round-to-nearest NVFP4 PTQ — every nn.Linear weight
    snapped to the grid, zero training. The result is exactly what an
    NVFP4 kernel reconstructs at serving time."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) and not isinstance(m, FakeQuantLinear):
            m.weight.data = _qdq_nvfp4(m.weight.data, block)
            n += 1
    return n


def train_model(cfg, ids, steps, batch, seq, lr, mtp_weight, seed, qat):
    """From-scratch training, mirroring train_toy_checkpoint.py's loop
    (Adam + cosine, MTP aux loss). With ``qat=True`` every Linear is
    NVFP4-fake-quantized (STE) BEFORE step 0. Returns (model, applied,
    first_train_loss10, final_train_loss10)."""
    torch.manual_seed(seed)
    model = EvastarV5(cfg)
    applied = apply_qat(model, method="nvfp4") if qat else None
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                       T_max=max(steps, 1))
    gen = torch.Generator().manual_seed(seed)

    def draw():
        ix = torch.randint(0, len(ids) - seq - 2, (batch,), generator=gen)
        return torch.stack([ids[i:i + seq + 1] for i in ix])

    losses = []
    for _step in range(steps):
        data = draw()
        x, y = data[:, :-1], data[:, 1:]
        logits, hidden, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total = loss
        if mtp_weight > 0 and model.mtp_modules is not None:
            mtp_loss = 0.0
            for mtp in model.mtp_modules:
                d = mtp.module_index + 1
                usable = x.shape[1] - d - 1
                if usable <= 0:
                    continue
                mlogits, _ = mtp.forward_with_hidden(hidden, x)
                tgt = x[:, d + 1: d + 1 + usable].reshape(-1)
                mtp_loss = mtp_loss + F.cross_entropy(
                    mlogits.reshape(-1, mlogits.shape[-1]), tgt)
            total = total + mtp_weight * mtp_loss
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    first10 = sum(losses[:10]) / len(losses[:10]) if losses else float("nan")
    last10 = sum(losses[-10:]) / len(losses[-10:]) if losses else float("nan")
    return model, applied, first10, last10


def bake_deploy(model, applied):
    """Fold each FakeQuantLinear's fake-quant weight into a plain
    nn.Linear — the deployable NVFP4 artifact — and re-bind MTP shared
    modules after replacement (same discipline as apply_qat)."""
    modules = dict(model.named_modules())
    for name, fq in applied:
        baked = nn.Linear(fq.in_features, fq.out_features,
                          bias=fq.bias is not None)
        baked.weight.data = fq.fake_quant_weight()
        if fq.bias is not None:
            baked.bias.data = fq.bias.data.clone()
        parent_name, _, child_name = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child_name, baked)
    mtp_modules = getattr(model, "mtp_modules", None)
    if mtp_modules is not None:
        for mtp in mtp_modules:
            if hasattr(mtp, "lm_head"):
                mtp.lm_head = model.lm_head
            if hasattr(mtp, "embed_tokens"):
                mtp.embed_tokens = model.embed_tokens
    return model


def run_qat_arm(cfg, args, ids, window):
    """One full QAT arm: from-scratch STE training -> bake -> baked val
    + probe. Factored out so the bitwise gate replays it byte-for-byte."""
    model, applied, first10, final10 = train_model(
        cfg, ids, args.steps, args.batch, args.seq, args.lr,
        args.mtp_weight, args.seed, qat=True)
    val_ste = val_loss_of(model, window)
    bake_deploy(model, applied)
    val_baked = val_loss_of(model, window)
    probe = probe_of(model, cfg, args.prompt, args.probe_tokens)
    n_wrapped = len(applied)
    del model, applied
    return {"first_train_loss10": first10, "final_train_loss10": final10,
            "val_loss_ste": val_ste, "val_loss_baked": val_baked,
            "probe_ids": probe, "wrapped_linears": n_wrapped}


def run_smoke(args):
    cfg = EvastarV5Config(size="lite")
    ids = stable_corpus(cfg.vocab_size)
    window = val_window(ids, args.seed)
    telemetry = {}

    # --- arm 1: fp32 from scratch (quality reference) ---
    t0 = time.perf_counter()
    model, _, fp_first10, fp_final10 = train_model(
        cfg, ids, args.steps, args.batch, args.seq, args.lr,
        args.mtp_weight, args.seed, qat=False)
    fp_val = val_loss_of(model, window)
    fp_probe = probe_of(model, cfg, args.prompt, args.probe_tokens)
    telemetry["fp32_scratch"] = {"wall_s": round(time.perf_counter() - t0, 2)}
    arm_fp = {
        "label": "fp32_scratch",
        "val_loss": fp_val,
        "first_train_loss10": fp_first10,
        "final_train_loss10": fp_final10,
        "probe_ids": fp_probe,
        "probe_sha256": hashlib.sha256(
            json.dumps(fp_probe).encode()).hexdigest(),
    }
    print(f"[arm fp32_scratch] val_loss={fp_val:.4f}", flush=True)

    # --- arm 2: RTN PTQ on the fp32-trained model, zero extra training ---
    t0 = time.perf_counter()
    n_rtn = rtn_deploy(model)
    ptq_val = val_loss_of(model, window)
    ptq_probe = probe_of(model, cfg, args.prompt, args.probe_tokens)
    del model
    telemetry["ptq_rtn_nvfp4"] = {"wall_s": round(time.perf_counter() - t0, 2)}
    arm_ptq = {
        "label": "ptq_rtn_nvfp4",
        "quantized_linears": n_rtn,
        "val_loss": ptq_val,
        "delta_vs_fp32": ptq_val - fp_val,
        "probe_ids": ptq_probe,
        "probe_sha256": hashlib.sha256(
            json.dumps(ptq_probe).encode()).hexdigest(),
    }
    print(f"[arm ptq_rtn_nvfp4] val_loss={ptq_val:.4f} "
          f"(delta {ptq_val - fp_val:+.4f})", flush=True)

    # --- arm 3: QAT from scratch -> bake -> deploy ---
    t0 = time.perf_counter()
    qat = run_qat_arm(cfg, args, ids, window)
    telemetry["qat_nvfp4"] = {"wall_s": round(
        time.perf_counter() - t0, 2)}
    parity_exact = qat["val_loss_baked"] == qat["val_loss_ste"]
    arm_qat = {
        "label": "qat_nvfp4",
        "steps": args.steps,
        "wrapped_linears": qat["wrapped_linears"],
        "first_train_loss10": qat["first_train_loss10"],
        "final_train_loss10": qat["final_train_loss10"],
        "val_loss_ste_forward": qat["val_loss_ste"],
        "val_loss_baked": qat["val_loss_baked"],
        "deploy_parity_exact": parity_exact,
        "delta_vs_fp32": qat["val_loss_baked"] - fp_val,
        "delta_vs_ptq": qat["val_loss_baked"] - ptq_val,
        "probe_ids": qat["probe_ids"],
        "probe_sha256": hashlib.sha256(
            json.dumps(qat["probe_ids"]).encode()).hexdigest(),
    }
    print(f"[arm qat_nvfp4] steps={args.steps} "
          f"val_loss={qat['val_loss_baked']:.4f} "
          f"(vs fp32 {qat['val_loss_baked'] - fp_val:+.4f}, "
          f"parity={'EXACT' if parity_exact else 'MISMATCH'})",
          flush=True)

    # --- bitwise gate: replay the QAT arm, must be identical ---
    if args.skip_gate:
        gate = {"pass": None, "reason": "--skip-gate"}
    else:
        t0 = time.perf_counter()
        rerun = run_qat_arm(cfg, args, ids, window)
        telemetry["qat_nvfp4_gate_rerun"] = {
            "wall_s": round(time.perf_counter() - t0, 2)}
        gate = {
            "pass": (rerun["val_loss_baked"] == qat["val_loss_baked"]
                     and rerun["probe_ids"] == qat["probe_ids"]
                     and rerun["final_train_loss10"]
                     == qat["final_train_loss10"]),
            "val_loss_identical":
                rerun["val_loss_baked"] == qat["val_loss_baked"],
            "probe_identical": rerun["probe_ids"] == qat["probe_ids"],
            "rerun_val_loss_baked": rerun["val_loss_baked"],
            "rerun_probe_sha256": hashlib.sha256(
                json.dumps(rerun["probe_ids"]).encode()).hexdigest(),
        }
    print(f"[gate] {'PASS' if gate['pass'] else gate}", flush=True)

    # --- verdict: smoke oracles, not a quality-win claim ---
    converged = (qat["val_loss_baked"] <= fp_val + args.qat_tol
                 and qat["final_train_loss10"] < qat["first_train_loss10"])
    verdict_pass = gate["pass"] is True and parity_exact and converged
    verdict = {
        "pass": verdict_pass,
        "deploy_parity_exact": parity_exact,
        "qat_within_fp32_tol": qat["val_loss_baked"] <= fp_val + args.qat_tol,
        "qat_train_loss_descends":
            qat["final_train_loss10"] < qat["first_train_loss10"],
        "qat_tol": args.qat_tol,
        "ptq_delta_vs_fp32": arm_ptq["delta_vs_fp32"],
        "qat_delta_vs_fp32": arm_qat["delta_vs_fp32"],
        "qat_delta_vs_ptq": arm_qat["delta_vs_ptq"],
        "criterion": ("bitwise gate PASS + deploy parity EXACT + QAT "
                      "deploy val within --qat-tol of fp32 arm + QAT "
                      "train loss descends; PTQ-vs-QAT delta recorded "
                      "but not gated (NVFP4 RTN is nearly free at toy "
                      "scale)"),
    }

    results = {
        "config": {
            "model": "lite from scratch",
            "quant": {"method": "nvfp4", "block_width": NVFP4_BLOCK,
                      "value_grid": "E2M1", "block_scale": "E4M3",
                      "top_scale": "fp32 per-row"},
            "steps": args.steps, "batch": args.batch, "seq": args.seq,
            "lr": args.lr, "mtp_weight": args.mtp_weight,
            "seed": args.seed, "prompt": args.prompt,
            "probe_tokens": args.probe_tokens,
            "corpus_tokens": len(ids),
        },
        "arms": [arm_fp, arm_ptq, arm_qat],
        "bitwise_gate": gate,
        "verdict": verdict,
    }
    return results, telemetry


def main():
    ap = argparse.ArgumentParser(
        description="B1-3: NVFP4 QAT end-to-end training smoke — "
                    "QAT-from-scratch vs fp32 vs RTN-PTQ with deploy "
                    "parity and a bitwise gate")
    ap.add_argument("--steps", type=int, default=240,
                    help="training steps per trained arm")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--mtp-weight", type=float, default=0.3,
                    help="MTP aux loss weight (0 disables)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--prompt", default="HeliosLM")
    ap.add_argument("--probe-tokens", type=int, default=24)
    ap.add_argument("--qat-tol", type=float, default=0.10,
                    help="max QAT-deploy val-loss gap over the fp32 arm")
    ap.add_argument("--skip-gate", action="store_true",
                    help="skip the bitwise-gate replay of the QAT arm")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    results, telemetry = run_smoke(args)
    payload = {
        "experiment": EXPERIMENT,
        "scoring": SCORING_TAG,
        "probe_version": PROBE_VERSION,
        "results": results,
        "telemetry": telemetry,
    }
    canon = json.dumps(results, sort_keys=True).encode()
    payload["results_sha256"] = hashlib.sha256(canon).hexdigest()

    print(f"\n{'arm':<16}{'val_loss':>10}{'vs fp32':>10}")
    fp_val = results["arms"][0]["val_loss"]
    for a in results["arms"]:
        v = a["val_loss"] if "val_loss" in a else a["val_loss_baked"]
        print(f"{a['label']:<16}{v:>10.4f}{v - fp_val:>+10.4f}")
    print(f"verdict: {results['verdict']}")
    print(f"results sha256: {payload['results_sha256']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n",
                       encoding="utf-8")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
