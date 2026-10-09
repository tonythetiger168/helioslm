"""End-to-end: quant-calib probe report -> TrustGate v2 decision (v5.47).

The v5.44 pipeline closed the math between ``eval/quant_calib`` and
``agent/trust_gate_v2`` (``ece_drift_from_report`` +
``trust_calibration_from_report``), but only as unit-tested functions.
This example is the runnable wiring a deployment would actually follow:

1. ``run_probe_demo``: score the SAME lite model bf16 vs its NVFP4
   fake-quant twin on a small MC task set and print the drift report.
   HONEST SCOPE: an untrained lite model yields MECHANICS numbers, not a
   drift claim — the probe module's own disclaimer applies, and this
   script repeats it in the output.
2. ``run_gate_demo``: take ANY well-formed drift report (the probe's,
   or a recorded one) and route one trust decision through TrustGateV2
   calibrated by ``trust_calibration_from_report`` — the widen-only
   band is visible in the printed decision record.

Usage (from the extracted tree root):
    python examples/quant_calib_trust_gate.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.eval.quant_calib import compare_bf16_vs_nvfp4
from helioslm_v5.src.model_v5 import HeliosLMv5


def _tiny_mc_tasks(n, choices=8):
    """Constructed-ground-truth MC tasks (no generation, no reference
    LLM — same discipline as the quant_calib oracle suite)."""
    return [{"prompt_ids": [5, 6, 7],
             "choice_ids": list(range(choices)),
             "answer_idx": i % choices} for i in range(n)]


def run_probe_demo(n_tasks=8, seed=0):
    """bf16 vs NVFP4 fake-quant calibration probe on the lite model.

    Returns the drift report dict. The numbers are mechanics smoke on
    an UNTRAINED model (see quant_calib's honesty constraints) — the
    printed banner says so, and so does the report's "method" field.
    """
    torch.manual_seed(seed)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    report = compare_bf16_vs_nvfp4(model, _tiny_mc_tasks(n_tasks),
                                   model_tag="HeliosLMv5-lite-untrained")
    print("=== quant-calib probe (MECHANICS on an untrained lite model — "
          "not a drift claim) ===")
    for side in ("bf16", "nvfp4"):
        arm = report[side]
        print(f"  {side:6s} acc={arm['accuracy']:.3f} "
              f"ece={arm['ece']:.4f} n={arm['n']}")
    print(f"  recomputed ece drift "
          f"(nvfp4 - bf16): {report['drift']['ece']:+.4f}")
    print(f"  method: {report['method']}")
    return report


def run_gate_demo(report, p_trust=0.84, cost_wrong=1.0, cost_escalate=0.2):
    """Route one decision through TrustGateV2 calibrated from a probe
    report. Returns the decide_explain record so callers (and the
    test) can audit p, p*, the band, and the route.

    The agent modules are path-imported like the oracle suite does
    (the agent/ directory predates package-ization)."""
    agent_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "helioslm_v5", "agent")
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    from decision_head import DecisionHead
    from trust_gate_v2 import (TrustGateV2, trust_calibration_from_report)

    calibration = trust_calibration_from_report(report)
    gate = TrustGateV2(DecisionHead({"trust": ("noul", None)}),
                       cost_wrong=cost_wrong, cost_escalate=cost_escalate,
                       calibration=calibration)
    # Deterministic trust probability for a reproducible demo record.
    gate.router.head.forward = lambda ids: {"trust": (float(p_trust), None)}

    class _Call:
        name = "demo.call"

    rec = gate.decide_explain(_Call(), {})
    print("=== TrustGate v2 decision (calibrated from the probe report; "
          "widen-only band) ===")
    print(f"  calibration record: ece={calibration['ece']:.4f} "
          f"quant_drift={calibration['quant_drift']:+.4f} "
          f"policy={calibration['policy']} n={calibration['n']}")
    print(f"  p_trust={rec['p_trust']:.2f} p*={rec['p_star']:.2f} "
          f"band=[{rec['band'][0]:.2f}, {rec['band'][1]:.2f}] "
          f"route={rec['route']} calibrated={rec['calibrated']}")
    return rec


def main():
    report = run_probe_demo()
    print()
    run_gate_demo(report)
    print()
    print("NOTE: the demo above uses mechanics numbers from an untrained "
          "model. A deployment MUST pass a report from a trained "
          "checkpoint measured on held-out tasks — the pipeline math is "
          "identical, the evidentiary weight is not.")


if __name__ == "__main__":
    main()
