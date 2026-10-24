"""Route B: continued pretraining of an open base (v5.127).

The owner's choice of 2026-10-23: a strong Traditional Chinese model with
the least training, by continuing the
pretraining of an open Apache-2.0 3-4B base on Taiwanese data instead of
training from scratch. The chosen base is Google's Gemma 4 E4B. The
from-scratch 0.5B stays the research line.
docs/CPT_PLAN_B.md has the reasoning; this module holds the parts that
can be checked on CPU:

- ``BASES``: the approved bases (a plan may only name one of these)
  with their licence, size, vocabulary and llama.cpp architecture.
  Nothing here is downloaded; the revision is pinned on the pod.
- ``load_plan`` / ``validate_plan``: a run plan (``data/cpt/*.json``)
  naming the base, the mixture and the stages (continued pretraining,
  decay, SFT, on-policy distillation / RL, teacher data, eval).
- ``plan_cost``: GPU-hours and dollars per stage. Token stages cost
  6 * N * D FLOPs (training) or 2 * N * D (teacher or eval forward
  passes) at the plan's MFU; fixed stages give GPU-hours directly. The
  GPU prices are planning assumptions (RunPod-class on-demand, 2026-10),
  not quotes.

No torch; ``examples/cpt_plan.py`` prints the table.
"""
import json
from pathlib import Path

# Dense BF16 peak FLOP/s per GPU and an on-demand $/GPU-hour assumption.
GPUS = {
    "h100": {"peak_flops": 989e12, "usd_per_hour": 2.7},
    "h200": {"peak_flops": 989e12, "usd_per_hour": 3.6},
    "b200": {"peak_flops": 2250e12, "usd_per_hour": 6.0},
}

# Candidate bases. "params" is the language model's parameter count used
# for 6ND (embeddings included, vision towers excluded: training is text
# only). "llama_cpp_arch" is the GGUF architecture llama.cpp loads the
# base as, so a continued model exports with the stock converter.
BASES = {
    "gemma4-e4b": {
        "hf_id": "google/gemma-4-E4B", "license": "Apache-2.0",
        "params": 4.5e9, "vocab": 262144, "context": 131072,
        "llama_cpp_arch": "gemma4",
        "note": "4.5B effective (8B with per-layer embedding tables, "
                "which are lookups and cost no matmul FLOPs), 42 layers, "
                "sliding-window 512 + global attention; instruct model "
                "reports MMLU-Pro 69.4, MMMLU 76.6 (vendor numbers)"},
    "gemma4-e2b": {
        "hf_id": "google/gemma-4-E2B", "license": "Apache-2.0",
        "params": 2.3e9, "vocab": 262144, "context": 131072,
        "llama_cpp_arch": "gemma4",
        "note": "same family; the cheap pilot that checks the recipe"},
    "ministral3-3b": {
        "hf_id": "mistralai/Ministral-3-3B-Base-2512",
        "license": "Apache-2.0",
        "params": 3.4e9, "vocab": 131072, "context": 262144,
        "llama_cpp_arch": "mistral3",
        "note": "fallback; base / instruct / reasoning released 2025-12; "
                "hf_id to confirm on the pod"},
}

STAGE_KINDS = ("train", "forward", "fixed")


def validate_plan(plan):
    for key in ("name", "base", "gpu", "mfu", "stages"):
        if key not in plan:
            raise ValueError(f"plan is missing {key!r}")
    if plan["base"] not in BASES:
        raise ValueError(f"unknown base {plan['base']!r}; one of {sorted(BASES)}")
    if plan["gpu"] not in GPUS:
        raise ValueError(f"unknown gpu {plan['gpu']!r}; one of {sorted(GPUS)}")
    if not 0.0 < plan["mfu"] <= 1.0:
        raise ValueError("mfu must be in (0, 1]")
    if not plan["stages"]:
        raise ValueError("plan has no stages")
    names = [s.get("name") for s in plan["stages"]]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("stage names must be present and unique")
    for s in plan["stages"]:
        kind = s.get("kind")
        if kind not in STAGE_KINDS:
            raise ValueError(f"{s['name']}: kind must be one of {STAGE_KINDS}")
        if kind == "fixed":
            need = ("gpu_hours",)
        else:
            need = ("tokens",)
        for k in need:
            v = s.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(f"{s['name']}: {k} must be > 0")
        p = s.get("params")
        if p is not None and (isinstance(p, bool) or not isinstance(p, (int, float))
                              or p <= 0):
            raise ValueError(f"{s['name']}: params must be > 0")
        g = s.get("gpu")
        if g is not None and g not in GPUS:
            raise ValueError(f"{s['name']}: unknown gpu {g!r}")
    c = plan.get("contingency", 0.0)
    if not 0.0 <= c <= 2.0:
        raise ValueError("contingency must be in [0, 2]")
    return plan


def load_plan(path):
    return validate_plan(json.loads(Path(path).read_text(encoding="utf-8")))


def stage_cost(stage, plan):
    """{gpu_hours, usd} of one stage. A stage may override the plan's gpu
    and, for a teacher, the active parameter count (``params``)."""
    gpu = GPUS[stage.get("gpu", plan["gpu"])]
    if stage["kind"] == "fixed":
        hours = float(stage["gpu_hours"])
    else:
        n = stage.get("params", BASES[plan["base"]]["params"])
        flops = (6 if stage["kind"] == "train" else 2) * n * stage["tokens"]
        hours = flops / (gpu["peak_flops"] * plan["mfu"] * 3600)
    return {"gpu_hours": hours, "usd": hours * gpu["usd_per_hour"]}


def plan_cost(plan):
    """Per-stage rows plus totals, with the plan's contingency applied to
    the total only (rows stay the point estimate)."""
    validate_plan(plan)
    rows = [dict(name=s["name"], kind=s["kind"], **stage_cost(s, plan))
            for s in plan["stages"]]
    hours = sum(r["gpu_hours"] for r in rows)
    usd = sum(r["usd"] for r in rows)
    c = plan.get("contingency", 0.0)
    return {"rows": rows, "gpu_hours": hours, "usd": usd,
            "usd_with_contingency": usd * (1.0 + c)}
