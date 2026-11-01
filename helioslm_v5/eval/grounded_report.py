"""Grounding grader for JSON -> report data-to-text (v5.214).

A report writer receives a fixed findings JSON (one entry per nodule plus
an overall Lung-RADS category) and writes a zh-TW or English report. This
grader checks, with rules only (no LLM, CPU, deterministic), that every
fact the text states can be found in the JSON:

* ``unverified_facts``: a measurement, lobe or nodule type in the text
  that no JSON entry carries (fabricated numbers, wrong units, an extra
  nodule);
* ``contradictions``: a fact stated for a nodule (or the overall
  category) whose JSON value differs;
* ``omissions``: a required JSON field the text never states.

Facts are read from the NFKC-normalised text (full-width digits, ``㎜``),
so ``６．２公釐``, ``6.2 mm`` and ``6.2毫米`` are the same 6.2 mm. Units:
mm / 公釐 / 毫米, cm / 公分 / 釐米 (x10), mm3 / mm³ / 立方公釐 / 立方毫米.
Lobes: 右上葉 / RUL / right upper lobe and the other four (舌葉 / lingula
count as the left upper lobe). Types: solid / 實質 / 實心, part-solid /
部分實質 / 混合型, GGN / ground-glass / 毛玻璃. Lung-RADS: 0, 1, 2, 3, 4A,
4B, 4X after a ``Lung-RADS`` mention.

A sentence that names exactly one lobe, held by exactly one nodule, is
bound to that nodule: its measurements, type and category are compared
with that nodule's fields, so a value from another nodule is a
contradiction there. Facts in unbound sentences only have to exist
somewhere in the JSON. Numbers without a unit are ignored (counts, dates,
ids); spelled-out Chinese numerals (六公釐) are not read.

A JSON field that is absent or null is not required; a value stated for
it is unverified. ``NOT_MEASURED`` phrases (未測量 / not measured) in a
bound sentence are returned as ``abstentions`` for the field-level
abstention reward (src/training/abstain.py).

No patient data: the example schema and fixtures are synthetic.
"""
import re
import unicodedata

SCHEMA_EXAMPLE = {
    "exam_id": "SYN-0001",
    "lung_rads": "4A",
    "nodules": [
        {"id": "N1", "lobe": "RUL", "diameter_mm": 9.4, "volume_mm3": 310.0,
         "type": "solid", "lung_rads": "4A"},
        {"id": "N2", "lobe": "LLL", "diameter_mm": 4.1, "volume_mm3": None,
         "type": "GGN", "lung_rads": "2"},
    ],
}

REQUIRED_NODULE_FIELDS = ("lobe", "diameter_mm", "type")
OPTIONAL_NODULE_FIELDS = ("volume_mm3", "lung_rads")
LOBES = ("RUL", "RML", "RLL", "LUL", "LLL")
TYPES = ("solid", "part-solid", "GGN")
CATEGORIES = ("0", "1", "2", "3", "4A", "4B", "4X")
TOL = {"diameter_mm": 0.05, "volume_mm3": 0.5}

NOT_MEASURED = re.compile(r"未測量|未量測|未測得|not\s+measured", re.I)

_LOBE_PATTERNS = [
    ("RUL", r"右肺?上葉|\bRUL\b|right\s+upper\s+lobe"),
    ("RML", r"右肺?中葉|\bRML\b|right\s+middle\s+lobe"),
    ("RLL", r"右肺?下葉|\bRLL\b|right\s+lower\s+lobe"),
    ("LUL", r"左肺?上葉|舌葉|\bLUL\b|left\s+upper\s+lobe|lingula"),
    ("LLL", r"左肺?下葉|\bLLL\b|left\s+lower\s+lobe"),
]
# part-solid before solid: "部分實質" contains "實質", "part-solid" "solid"
_TYPE_PATTERNS = [
    ("part-solid", r"部分實質|部分實心|部分固態|混合型|part[\s-]*solid|subsolid"),
    ("GGN", r"毛玻璃|磨玻璃|\bGGNs?\b|\bGGOs?\b|ground[\s-]*glass|non[\s-]*solid|非實質"),
    ("solid", r"(?<!肺)實質|實心|固態|\bsolid\b"),
]
_UNIT = (r"(立方公釐|立方毫米|mm\^?3|cubic\s+millimet(?:er|re)s?"
         r"|公釐|毫米|millimet(?:er|re)s?|mm"
         r"|公分|釐米|centimet(?:er|re)s?|cm)")
_MEASURE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*" + _UNIT + r"(?![a-z])", re.I)
_RADS_VERSION = re.compile(r"(lung[\s-]?rads)\s*(?:v(?:ersion)?\s*)?(?:1\.1|2022|2019|2014)",
                           re.I)
_RADS = re.compile(r"lung[\s-]?rads[^0-9a-z\n]{0,8}?"
                   r"(?:category|cat\.?|類別|分類|第)?[^0-9a-z\n]{0,4}?"
                   r"([0-4])\s*([abx])?(?![0-9a-z])", re.I)
# sentence ends; "." only before space / end so decimals survive
_SENT = re.compile(r"[。；;！？!?\n]+|\.(?=\s|$)")


def normalize(text):
    """NFKC (full-width digits, ㎜ -> mm, ³ -> 3)."""
    return unicodedata.normalize("NFKC", text or "")


def _unit_kind(unit):
    u = unit.lower().replace(" ", "")
    if u.startswith("立方") or u.startswith("cubic") or u.startswith("mm^3") \
            or u == "mm3":
        return "volume_mm3", 1.0
    if u in ("公分", "釐米", "cm") or u.startswith("centimet"):
        return "diameter_mm", 10.0
    return "diameter_mm", 1.0


def _sentences(text):
    out, start = [], 0
    for m in _SENT.finditer(text):
        if m.start() > start:
            out.append((start, m.start()))
        start = m.end()
    if start < len(text):
        out.append((start, len(text)))
    return out


def _blank(text, s, e):
    return text[:s] + " " * (e - s) + text[e:]


def extract_facts(text):
    """Facts stated in ``text``: a list of dicts with ``field``, ``value``,
    ``span`` (in the NFKC text) and ``sentence`` index, sorted by span."""
    t = normalize(text)
    facts = []
    work = _RADS_VERSION.sub(lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), t)
    for m in _RADS.finditer(work):
        cat = m.group(1) + (m.group(2) or "").upper()
        if cat in CATEGORIES:
            facts.append({"field": "lung_rads", "value": cat,
                          "span": [m.start(1), m.end()], "text": t[m.start(1):m.end()]})
        work = _blank(work, m.start(), m.end())
    for m in _MEASURE.finditer(work):
        field, mult = _unit_kind(m.group(2))
        val = round(float(m.group(1)) * mult, 4)
        facts.append({"field": field, "value": val, "span": [m.start(), m.end()],
                      "text": t[m.start():m.end()]})
        work = _blank(work, m.start(), m.end())
    for name, pat in _LOBE_PATTERNS:
        for m in re.finditer(pat, work, re.I):
            facts.append({"field": "lobe", "value": name, "span": [m.start(), m.end()],
                          "text": t[m.start():m.end()]})
            work = _blank(work, m.start(), m.end())
    for name, pat in _TYPE_PATTERNS:
        for m in re.finditer(pat, work, re.I):
            facts.append({"field": "type", "value": name, "span": [m.start(), m.end()],
                          "text": t[m.start():m.end()]})
            work = _blank(work, m.start(), m.end())
    for m in NOT_MEASURED.finditer(t):
        facts.append({"field": "not_measured", "value": True,
                      "span": [m.start(), m.end()], "text": m.group(0)})
    sents = _sentences(t)
    for f in facts:
        f["sentence"] = next((i for i, (s, e) in enumerate(sents)
                              if s <= f["span"][0] < e), -1)
    facts.sort(key=lambda f: (f["span"][0], f["field"]))
    return facts


def _norm_type(v):
    if v is None:
        return None
    s = str(v).strip().lower().replace("_", "-")
    if s in ("ggn", "ggo", "ground-glass", "nonsolid", "non-solid"):
        return "GGN"
    if s in ("part-solid", "partsolid", "subsolid"):
        return "part-solid"
    if s == "solid":
        return "solid"
    raise ValueError(f"unknown nodule type {v!r}")


def _norm_cat(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    if s not in CATEGORIES:
        raise ValueError(f"unknown Lung-RADS category {v!r}")
    return s


def validate_findings(findings):
    """Normalised copy of a findings dict; ValueError on a bad schema."""
    if not isinstance(findings, dict) or not isinstance(findings.get("nodules"), list):
        raise ValueError("findings must be a dict with a 'nodules' list")
    out = {"lung_rads": _norm_cat(findings.get("lung_rads")), "nodules": []}
    seen = set()
    for i, n in enumerate(findings["nodules"]):
        nid = str(n.get("id") or f"N{i + 1}")
        if nid in seen:
            raise ValueError(f"duplicate nodule id {nid}")
        seen.add(nid)
        lobe = n.get("lobe")
        if lobe is not None:
            lobe = str(lobe).upper()
            if lobe not in LOBES:
                raise ValueError(f"unknown lobe {n.get('lobe')!r}")
        out["nodules"].append({
            "id": nid, "lobe": lobe,
            "diameter_mm": None if n.get("diameter_mm") is None else float(n["diameter_mm"]),
            "volume_mm3": None if n.get("volume_mm3") is None else float(n["volume_mm3"]),
            "type": _norm_type(n.get("type")),
            "lung_rads": _norm_cat(n.get("lung_rads")),
        })
    return out


def _same(field, a, b):
    if a is None or b is None:
        return False
    if field in TOL:
        return abs(float(a) - float(b)) <= TOL[field] + 1e-9
    return a == b


def _bindings(facts, nodules):
    """sentence index -> nodule id, for sentences naming exactly one lobe
    that exactly one nodule has."""
    lobes = {}
    for f in facts:
        if f["field"] == "lobe":
            lobes.setdefault(f["sentence"], set()).add(f["value"])
    bound = {}
    for s, ls in lobes.items():
        if len(ls) != 1:
            continue
        (lobe,) = ls
        hits = [n["id"] for n in nodules if n["lobe"] == lobe]
        if len(hits) == 1:
            bound[s] = hits[0]
    return bound


def grade(findings, text):
    """Grade ``text`` against ``findings``. Returns a dict with
    ``unverified_facts``, ``contradictions``, ``omissions``,
    ``abstentions``, ``counts`` and ``grounded`` (no unverified facts and
    no contradictions) and ``passed`` (also no omissions)."""
    fd = validate_findings(findings)
    nodules = fd["nodules"]
    by_id = {n["id"]: n for n in nodules}
    facts = extract_facts(text)
    bound = _bindings(facts, nodules)
    unverified, contradictions, abstentions = [], [], []

    def item(f, **kw):
        d = {"field": f["field"], "value": f["value"], "text": f["text"],
             "span": f["span"]}
        d.update(kw)
        return d

    for f in facts:
        field, nid = f["field"], bound.get(f["sentence"])
        if field == "not_measured":
            if nid is not None:
                missing = [k for k in ("diameter_mm", "volume_mm3")
                           if by_id[nid][k] is None]
                abstentions.append({"nodule": nid, "fields": missing,
                                    "text": f["text"], "span": f["span"]})
            continue
        if field == "lobe":
            if not any(n["lobe"] == f["value"] for n in nodules):
                unverified.append(item(f, nodule=None))
            continue
        if field == "lung_rads":
            if nid is not None and by_id[nid]["lung_rads"] is not None:
                exp = by_id[nid]["lung_rads"]
                if f["value"] != exp:
                    contradictions.append(item(f, nodule=nid, expected=exp))
                continue
            known = {fd["lung_rads"]} | {n["lung_rads"] for n in nodules}
            known.discard(None)
            if f["value"] not in known:
                if fd["lung_rads"] is not None:
                    contradictions.append(item(f, nodule=None, expected=fd["lung_rads"]))
                else:
                    unverified.append(item(f, nodule=None))
            continue
        # diameter_mm, volume_mm3, type
        if nid is not None:
            exp = by_id[nid][field]
            if exp is None:
                unverified.append(item(f, nodule=nid))
            elif not _same(field, f["value"], exp):
                contradictions.append(item(f, nodule=nid, expected=exp))
            continue
        if not any(_same(field, f["value"], n[field]) for n in nodules):
            unverified.append(item(f, nodule=None))

    stated = {}
    for f in facts:
        stated.setdefault(f["field"], []).append(f["value"])
    omissions = []
    if fd["lung_rads"] is not None and fd["lung_rads"] not in stated.get("lung_rads", []):
        omissions.append({"nodule": None, "field": "lung_rads",
                          "expected": fd["lung_rads"]})
    for n in nodules:
        for k in REQUIRED_NODULE_FIELDS:
            if n[k] is None:
                continue
            if not any(_same(k, v, n[k]) for v in stated.get(k, [])):
                omissions.append({"nodule": n["id"], "field": k, "expected": n[k]})

    counts = {"unverified_facts": len(unverified),
              "contradictions": len(contradictions),
              "omissions": len(omissions), "facts": len(facts)}
    grounded = not unverified and not contradictions
    return {"unverified_facts": unverified, "contradictions": contradictions,
            "omissions": omissions, "abstentions": abstentions,
            "counts": counts, "grounded": grounded,
            "passed": grounded and not omissions}


def summarize(results):
    """Aggregate a list of :func:`grade` results: totals, the omission
    rate over required fields and the share of reports with zero
    unverified facts."""
    n = len(results)
    tot = {k: sum(r["counts"][k] for r in results)
           for k in ("unverified_facts", "contradictions", "omissions", "facts")}
    return dict(tot, reports=n,
                grounded_rate=(sum(r["grounded"] for r in results) / n) if n else 0.0,
                passed_rate=(sum(r["passed"] for r in results) / n) if n else 0.0,
                zero_unverified_rate=(sum(r["counts"]["unverified_facts"] == 0
                                          for r in results) / n) if n else 0.0)
