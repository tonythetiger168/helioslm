"""Field-level abstention for JSON -> report writing (v5.216).

The OS3 abstention reward (abstain.py) teaches "I don't know" for a
question missing an operand. A report writer needs the field-level
version: when the findings JSON has no value for a measurement, the
report must say so with a fixed phrase (zh-TW 「未測量」, English
"not measured") and never fill in a number.

:class:`ReportAbstainEnv` builds synthetic findings (1–3 nodules in
distinct lobes; no patient data), removes measurement fields at random
(``p_missing`` per field), and scores a report with the grounding grader
(eval/grounded_report.py), sharing its field matching:

  - any value stated for a removed field (fabrication), or any other
    unverified fact / contradiction            -> -penalty
  - otherwise the share of checks passed, in [0, 1]: each removed field
    abstained with the phrase, each required present field stated

Hedging is an answer, as in abstain.py: a report that writes 未測量 and
also a number for that field is scored as a fabrication. Abstention is
attributed to a field by the words just before the phrase (直徑 / 大小 /
diameter / size, 體積 / volume); with neither, it covers every removed
field of the nodule the sentence is about.

Category rules in the synthetic findings are a simplified size / type
mapping for training text only, not Lung-RADS guidance.
"""
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from helioslm_v5.eval.grounded_report import (LOBES, REQUIRED_NODULE_FIELDS,
                                              grade, normalize)

NOT_MEASURED_TEXT = {"zh": "未測量", "en": "not measured"}
REMOVABLE = ("diameter_mm", "volume_mm3")
_FIELD_WORDS = {"diameter_mm": ("直徑", "大小", "diameter", "size"),
                "volume_mm3": ("體積", "volume")}
LOBE_TEXT = {"zh": {"RUL": "右上葉", "RML": "右中葉", "RLL": "右下葉",
                    "LUL": "左上葉", "LLL": "左下葉"},
             "en": {"RUL": "Right upper lobe", "RML": "Right middle lobe",
                    "RLL": "Right lower lobe", "LUL": "Left upper lobe",
                    "LLL": "Left lower lobe"}}
TYPE_TEXT = {"zh": {"solid": "實質", "part-solid": "部分實質", "GGN": "毛玻璃"},
             "en": {"solid": "solid", "part-solid": "part-solid", "GGN": "ground-glass"}}
_CAT_ORDER = ("0", "1", "2", "3", "4A", "4B", "4X")

SYSTEM = {
    "zh": ("只根據 JSON 撰寫繁體中文報告。JSON 沒有的數值寫「未測量」，"
           "不可自行補上任何數字。"),
    "en": ("Write the report from the JSON only. Write \"not measured\" for any "
           "value the JSON does not have; never add a number."),
}


def _category(kind, d):
    if kind == "GGN":
        return "2" if d < 30 else "3"
    if d < 6:
        return "2"
    if d < 8:
        return "3"
    if kind == "part-solid" or d < 15:
        return "4A"
    return "4B"


def synth_findings(rng: random.Random, max_nodules: int = 3) -> dict:
    """Synthetic findings dict (schema of grounded_report.SCHEMA_EXAMPLE)."""
    n = rng.randint(1, max_nodules)
    lobes = rng.sample(LOBES, n)
    nodules = []
    for i, lobe in enumerate(lobes):
        kind = rng.choice(("solid", "part-solid", "GGN"))
        d = round(rng.uniform(2.0, 25.0), 1)
        nodules.append({"id": f"N{i + 1}", "lobe": lobe, "diameter_mm": d,
                        "volume_mm3": float(round(math.pi / 6 * d ** 3)),
                        "type": kind, "lung_rads": _category(kind, d)})
    overall = max((x["lung_rads"] for x in nodules), key=_CAT_ORDER.index)
    return {"exam_id": f"SYN-{rng.randint(0, 10 ** 6):06d}", "lung_rads": overall,
            "nodules": nodules}


def remove_fields(findings: dict, rng: random.Random, p_missing: float):
    """(masked findings, [[nodule id, field], ...]) with each removable
    field set to null with probability ``p_missing``."""
    out = json.loads(json.dumps(findings))
    missing = []
    for nd in out["nodules"]:
        for k in REMOVABLE:
            if nd.get(k) is not None and rng.random() < p_missing:
                nd[k] = None
                missing.append([nd["id"], k])
    return out, missing


def _fmt(v):
    return f"{v:g}"


def render_reference(findings: dict, lang: str = "zh") -> str:
    """Template report from (masked) findings: the SFT target."""
    nm = NOT_MEASURED_TEXT[lang]
    parts = []
    for nd in findings["nodules"]:
        lobe, kind = LOBE_TEXT[lang][nd["lobe"]], TYPE_TEXT[lang][nd["type"]]
        d = nd.get("diameter_mm")
        v = nd.get("volume_mm3")
        if lang == "zh":
            ds = f"直徑 {_fmt(d)} 公釐" if d is not None else f"直徑{nm}"
            vs = f"體積 {_fmt(v)} 立方公釐" if v is not None else f"體積{nm}"
            cat = f"，Lung-RADS {nd['lung_rads']}" if nd.get("lung_rads") else ""
            parts.append(f"{lobe}見一顆{kind}結節，{ds}，{vs}{cat}。")
        else:
            ds = f"diameter {_fmt(d)} mm" if d is not None else f"diameter {nm}"
            vs = f"volume {_fmt(v)} mm3" if v is not None else f"volume {nm}"
            cat = f", Lung-RADS {nd['lung_rads']}" if nd.get("lung_rads") else ""
            parts.append(f"{lobe}: {kind} nodule, {ds}, {vs}{cat}.")
    if findings.get("lung_rads"):
        parts.append(f"整體 Lung-RADS 類別 {findings['lung_rads']}。" if lang == "zh"
                     else f"Overall Lung-RADS category {findings['lung_rads']}.")
    return ("" if lang == "zh" else " ").join(parts)


def build_prompt(findings: dict, lang: str = "zh", system: bool = True) -> str:
    """The user prompt; ``system`` False leaves the instruction out (it
    goes in the SFT record's system field instead)."""
    head = "JSON：" if lang == "zh" else "JSON:"
    body = head + "\n" + json.dumps(findings, ensure_ascii=False, indent=2)
    return SYSTEM[lang] + "\n\n" + body if system else body


def _abstained_fields(text, ab, null_fields):
    """Fields one abstention phrase covers, from the words before it."""
    t = normalize(text)
    before = t[max(0, ab["span"][0] - 8):ab["span"][0]].lower()
    named = [k for k in null_fields if any(w in before for w in _FIELD_WORDS[k])]
    return named or list(null_fields)


def score_report(target: dict, response: str, penalty: float = 1.0) -> Dict:
    """Detailed field-level score; ``reward`` is the GRPO reward."""
    if penalty < 0:
        raise ValueError("penalty must be >= 0")
    findings, missing = target["findings"], [tuple(m) for m in target["missing"]]
    text = response.split("</think>")[-1]
    g = grade(findings, text)
    fabricated = sorted({(u["nodule"], u["field"]) for u in g["unverified_facts"]
                         if u["field"] in REMOVABLE and
                         (u["nodule"] is None or (u["nodule"], u["field"]) in missing)})
    abstained = set()
    for ab in g["abstentions"]:
        for k in _abstained_fields(text, ab, ab["fields"]):
            abstained.add((ab["nodule"], k))
    ok_missing = [m for m in missing if m in abstained and m not in fabricated]
    required = [(nd["id"], k) for nd in findings["nodules"] for k in REQUIRED_NODULE_FIELDS
                if nd.get(k) is not None]
    omitted = {(o["nodule"], o["field"]) for o in g["omissions"]}
    ok_present = [r for r in required if r not in omitted]
    errors = bool(g["unverified_facts"] or g["contradictions"])
    checks = len(missing) + len(required)
    if errors:
        reward = -penalty
    else:
        reward = (len(ok_missing) + len(ok_present)) / checks if checks else 1.0
    return {"reward": reward, "fabricated": [list(f) for f in fabricated],
            "abstained": sorted([list(m) for m in ok_missing]),
            "silent": sorted([list(m) for m in missing if m not in abstained
                              and m not in fabricated]),
            "grade": g}


def report_abstain_reward(target: dict, response: str, penalty: float = 1.0) -> float:
    return score_report(target, response, penalty)["reward"]


@dataclass
class ReportAbstainEnv:
    """GRPO env (MultiEnvBatch-compatible: ``task(seed)`` / ``reward``)."""
    name: str = "report-abstain"
    p_missing: float = 0.3
    penalty: float = 1.0
    lang: Optional[str] = None      # None: zh and en at random
    rng: random.Random = None

    def __post_init__(self):
        if not 0.0 <= self.p_missing <= 1.0:
            raise ValueError("p_missing must be in [0, 1]")
        if self.lang not in (None, "zh", "en"):
            raise ValueError("lang must be None, 'zh' or 'en'")
        if self.rng is None:
            self.rng = random.Random(0)

    def task(self, seed: Optional[int] = None) -> dict:
        rng = random.Random(seed) if seed is not None else self.rng
        lang = self.lang or rng.choice(("zh", "en"))
        masked, missing = remove_fields(synth_findings(rng), rng, self.p_missing)
        return {"prompt": build_prompt(masked, lang),
                "target": {"findings": masked, "missing": missing, "lang": lang}}

    def reward(self, response: str, target: dict) -> float:
        return report_abstain_reward(target, response, self.penalty)


def make_sft_records(n: int, seed: int = 0, p_missing: float = 0.3,
                     lang: Optional[str] = None) -> List[dict]:
    """``n`` synthetic {system, prompt, response} records for sft_hf.py;
    each reference response is checked to score reward 1.0."""
    env = ReportAbstainEnv(p_missing=p_missing, lang=lang)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = env.task(seed=rng.randint(0, 10 ** 9))
        tg = t["target"]
        resp = render_reference(tg["findings"], tg["lang"])
        r = report_abstain_reward(tg, resp)
        if r != 1.0:
            raise AssertionError(f"reference report scored {r}: {resp}")
        out.append({"system": SYSTEM[tg["lang"]],
                    "prompt": build_prompt(tg["findings"], tg["lang"], system=False),
                    "response": resp, "missing": tg["missing"], "lang": tg["lang"]})
    return out
