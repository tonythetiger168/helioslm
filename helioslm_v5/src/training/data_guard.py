"""PHI and licence guard for medical training data (v5.215).

Medical SFT / CPT data that still carries patient identifiers is a
serious incident, so medical sources have to be declared and checked
before any training or distillation step loads them.

A data manifest (JSON) lists the sources::

    {"sources": [
      {"name": "report-pairs-v1", "path": "pairs.jsonl", "domain": "medical",
       "deidentified": true, "license": "internal-deidentified",
       "origin": "synthetic generator v3"},
      {"name": "zhwiki", "path": "zhwiki.jsonl", "domain": "general",
       "license": "CC-BY-SA-4.0", "origin": "wikimedia dump"}]}

:func:`check_manifest` refuses a medical source unless it says
``deidentified: true`` and names a licence and an origin; when the
weights may be published (``release=True``, the default) it also refuses
research-only data (``research_only: true``, a non-commercial / research
licence, or a known research-only dataset such as ``medical-bench-tw``).
:func:`guard` additionally scans every medical source file with the PHI
scanner and stops at the first hits with their line numbers.

The PHI scanner is deliberately conservative (false positives stop the
run, a human then looks): Taiwanese national ID / resident certificate
format, medical-record-number fields, Taiwanese phone numbers, e-mail
addresses, a date on the same line as a name field or an honorific
(先生 / 小姐 / 女士), and DICOM patient tags (PatientName, PatientID,
(0010,0010) ...). Hits are reported masked, never with the full value.

Non-medical sources are not scanned, so general CPT / SFT data (which
contains phone numbers and names) runs as before. Medical data must never
be sent to an external API teacher: :func:`check_teacher_endpoint` refuses a
non-local base URL when the manifest has medical sources.
"""
import ipaddress
import json
import re
from pathlib import Path
from urllib.parse import urlparse


class DataGuardError(RuntimeError):
    """A manifest or PHI check failed. Never swallowed."""


RESEARCH_ONLY_DATASETS = ("medical-bench-tw",)
_RESEARCH_LICENSE = re.compile(
    r"research|non[\s-]?commercial|\bnc\b|-nc-|-nc$|academic|evaluation[\s-]?only", re.I)

PHI_PATTERNS = [
    ("tw_national_id", re.compile(r"(?<![A-Za-z0-9])[A-Z][1289]\d{8}(?!\d)")),
    ("medical_record_no", re.compile(
        r"(?:病歷號碼?|病歷編號|病例號|MRN|chart\s*(?:no|number)\.?|"
        r"medical\s+record\s+(?:no|number)\.?)\s*[:：#]?\s*[A-Za-z0-9-]{4,}", re.I)),
    ("phone", re.compile(
        r"(?<!\d)(?:\+?886[-\s]?|0)9\d{2}[-\s]?\d{3}[-\s]?\d{3}(?!\d)"
        r"|(?<!\d)\(?0[2-8]\)?[-\s]?\d{3,4}[-\s]?\d{4}(?!\d)")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("dicom_tag", re.compile(
        r"\b(?:PatientName|PatientID|PatientBirthDate|PatientAddress|"
        r"PatientTelephoneNumbers|OtherPatientIDs|PatientMotherBirthName|"
        r"AccessionNumber|ReferringPhysicianName|InstitutionAddress)\b"
        r"|\(0010,\s*(?:0010|0020|0030|1000|1040|2154)\)", re.I)),
]
_DATE = re.compile(r"(?:19|20)\d\d\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}"
                   r"|民國\s*\d{2,3}\s*年\s*\d{1,2}\s*月"
                   r"|(?<!\d)\d{2,3}[/.]\d{1,2}[/.]\d{1,2}(?!\d)")
_NAME = re.compile(r"(?:姓名|病人|病患|患者|patient\s*name|name)\s*[:：]\s*\S"
                   r"|[一-鿿]{1,3}(?:先生|小姐|女士)", re.I)


def _mask(s):
    s = s.strip()
    return s[:2] + "*" * max(0, len(s) - 2) if len(s) > 2 else "*" * len(s)


def scan_text(text):
    """PHI hits in one string: [{"kind", "masked"}], in pattern order."""
    hits = []
    for kind, pat in PHI_PATTERNS:
        for m in pat.finditer(text):
            hits.append({"kind": kind, "masked": _mask(m.group(0))})
    d, n = _DATE.search(text), _NAME.search(text)
    if d and n:
        hits.append({"kind": "date_with_name", "masked": _mask(d.group(0))})
    return hits


def scan_file(path, max_hits=20):
    """PHI hits in a text / JSONL file: [{"line", "kind", "masked"}], at
    most ``max_hits`` (the run stops anyway)."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            for h in scan_text(line):
                out.append(dict(h, line=i))
                if len(out) >= max_hits:
                    return out
    return out


def load_manifest(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(m, dict) or not isinstance(m.get("sources"), list):
        raise DataGuardError(f"{path}: manifest needs a 'sources' list")
    base = Path(path).resolve().parent
    for s in m["sources"]:
        if s.get("path") and not Path(s["path"]).is_absolute():
            s["path"] = str(base / s["path"])
    return m


def is_medical(source):
    return str(source.get("domain", "")).lower() in ("medical", "clinical", "health")


def research_only(source):
    if source.get("research_only"):
        return True
    if _RESEARCH_LICENSE.search(str(source.get("license", ""))):
        return True
    ident = " ".join(str(source.get(k, "")) for k in ("name", "path", "origin")).lower()
    return any(d in ident for d in RESEARCH_ONLY_DATASETS)


def check_manifest(manifest, release=True):
    """Raise DataGuardError listing every bad source; returns the medical
    source names when all pass."""
    problems, medical = [], []
    for i, s in enumerate(manifest.get("sources", [])):
        name = s.get("name") or f"source[{i}]"
        if release and research_only(s):
            problems.append(f"{name}: research-only data cannot train releasable weights")
        if not is_medical(s):
            continue
        medical.append(name)
        if s.get("deidentified") is not True:
            problems.append(f"{name}: medical source without 'deidentified: true'")
        for k in ("license", "origin"):
            if not str(s.get(k) or "").strip():
                problems.append(f"{name}: medical source without '{k}'")
    if problems:
        raise DataGuardError("data manifest refused:\n  " + "\n  ".join(problems))
    return medical


def guard(manifest, release=True, scan=True):
    """check_manifest, then scan each medical source file; raises on any
    PHI hit with ``name:line kind masked`` lines. Returns a report dict."""
    medical = check_manifest(manifest, release=release)
    report = {"medical_sources": medical, "scanned": [], "hits": []}
    if not scan:
        return report
    for s in manifest["sources"]:
        if not is_medical(s):
            continue
        if not s.get("path") or not Path(s["path"]).is_file():
            raise DataGuardError(f"{s.get('name')}: medical source file not found: {s.get('path')}")
        hits = scan_file(s["path"])
        report["scanned"].append(s.get("name"))
        report["hits"] += [dict(h, source=s.get("name")) for h in hits]
    if report["hits"]:
        lines = [f"{h['source']}:{h['line']} {h['kind']} {h['masked']}" for h in report["hits"]]
        raise DataGuardError("possible PHI in medical data, run stopped:\n  " + "\n  ".join(lines))
    return report


def guard_records(records, source_name="records"):
    """Scan an in-memory list of strings / dicts (e.g. distillation prompts)."""
    hits = []
    for i, r in enumerate(records, 1):
        text = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
        hits += [dict(h, line=i) for h in scan_text(text)]
    if hits:
        lines = [f"{source_name}:{h['line']} {h['kind']} {h['masked']}" for h in hits[:20]]
        raise DataGuardError("possible PHI in medical data, run stopped:\n  " + "\n  ".join(lines))


def is_local_url(url):
    host = urlparse(url).hostname or ""
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def check_teacher_endpoint(manifest, base_url, endpoint="vllm"):
    """Medical data never goes to an external API teacher."""
    medical = [s.get("name") for s in manifest.get("sources", []) if is_medical(s)]
    if medical and (endpoint != "vllm" or not is_local_url(base_url)):
        raise DataGuardError(
            f"medical sources {medical} cannot be sent to an external teacher "
            f"({endpoint} {base_url}); use a self-hosted teacher on localhost / a private address")


def guard_cli(manifest_path, release=True, medical=False):
    """Shared entry for the training scripts. ``medical`` True without a
    manifest is refused; no manifest and not medical is a no-op and
    returns (None, None). Otherwise (manifest, guard report)."""
    if not manifest_path:
        if medical:
            raise DataGuardError("--medical needs --data-manifest (deidentified, licence, origin)")
        return None, None
    m = load_manifest(manifest_path)
    if medical and not any(is_medical(s) for s in m["sources"]):
        raise DataGuardError("--medical given but the manifest declares no medical source")
    return m, guard(m, release=release)


def add_guard_args(parser):
    """--data-manifest / --medical / --no-release on a training CLI."""
    parser.add_argument("--data-manifest",
                        help="JSON data manifest; medical sources are checked and PHI-scanned")
    parser.add_argument("--medical", action="store_true",
                        help="the data is medical: --data-manifest is required")
    parser.add_argument("--no-release", action="store_true",
                        help="weights will not be published: research-only data allowed")
