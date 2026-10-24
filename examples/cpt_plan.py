"""cpt_plan.py — route B run plan and cost table (v5.127).

  python3 examples/cpt_plan.py data/cpt/b1_gemma4_e4b.json
  python3 examples/cpt_plan.py data/cpt/b0_gemma4_e2b_pilot.json --json
  python3 examples/cpt_plan.py --bases

Prints each stage's GPU-hours and dollars (evastar/src/training/
cpt_plan.py: 6ND for training, 2ND for forward passes, at the plan's
MFU and GPU price assumption), the base's licence and the mixture's
token split. Spends nothing: the GPU runs need the owner's approval.
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.src.training.cpt_plan import (  # noqa: E402
    BASES, load_plan, plan_cost)
from evastar.src.training.data_mixture import load_mixture  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", nargs="?")
    ap.add_argument("--bases", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.bases:
        for k, b in BASES.items():
            print(f"{k:14s} {b['hf_id']:38s} {b['license']:11s} "
                  f"{b['params'] / 1e9:4.1f}B vocab {b['vocab']:6d} "
                  f"gguf {b['llama_cpp_arch']}")
        return BASES
    if not a.plan:
        ap.error("give a plan file or --bases")
    plan = load_plan(a.plan)
    cost = plan_cost(plan)
    base = BASES[plan["base"]]
    out = {"plan": plan["name"], "base": base["hf_id"],
           "license": base["license"], **cost}
    if plan.get("mixture"):
        mix = load_mixture(_REPO / plan["mixture"])
        tw = sum(s["weight"] for s in mix["sources"]
                 if s.get("phase", "main") == "main"
                 and "tw" in (s.get("zh_variants") or []))
        out["mixture"] = {"name": mix["name"], "main_tw_share": round(tw, 3)}
    if a.json:
        print(json.dumps(out, indent=2))
        return out
    print(f"{plan['name']}: {base['hf_id']} ({base['license']}), "
          f"{plan['gpu']} at MFU {plan['mfu']}")
    for r in cost["rows"]:
        print(f"  {r['name']:12s} {r['kind']:8s} {r['gpu_hours']:8.0f} GPU-h "
              f"  ${r['usd']:9,.0f}")
    print(f"  {'total':21s} {cost['gpu_hours']:8.0f} GPU-h   "
          f"${cost['usd']:9,.0f}  (with contingency "
          f"${cost['usd_with_contingency']:,.0f})")
    return out


if __name__ == "__main__":
    main()
