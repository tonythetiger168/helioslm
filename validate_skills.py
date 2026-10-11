#!/usr/bin/env python3
"""Structural validator for skills/ against docs/SKILL_TEMPLATE.md.

Pure stdlib. Checks, per skills/<name>/SKILL.md:
  - frontmatter present; name == directory; description carries "Use when";
    type in {entry, gateway, atomic}; guard is valid JSON with the
    documented shape when present
  - required sections exist in the template's fixed order
  - pitfalls section has at least 3 bullets
  - [[wiki-links]] resolve (bare -> sibling skill or own references/;
    prefixed skill/file -> skills/<skill>/references/<file>; [[<...>]]
    placeholders never resolve) — in SKILL.md and references/*.md alike
  - at most one `entry` skill

Exit 0 when clean (or when skills/ does not exist yet — the gate is
inert until the first skill lands), 1 with a per-skill report otherwise.
"""

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SKILLS_DIR = _REPO_ROOT / "skills"

_VALID_TYPES = {"entry", "gateway", "atomic"}
_LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_BULLET_RE = re.compile(r"^\s*[-*]\s+", re.M)

_REQUIRED_SECTIONS = {
    "atomic": ["## When to use / when not to use", "## Procedure",
               "## Pitfalls"],
    "entry": ["## When to use / when not to use", "## Procedure",
              "## Pitfalls"],
    "gateway": ["## Full workflow", "## Which atomic skill when",
                "## Pitfalls"],
}


def _parse_frontmatter(text):
    """Return (meta_dict, body) or raise ValueError."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing frontmatter (file must start with ---)")
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        raise ValueError("unterminated frontmatter (no closing ---)")
    meta = {}
    i = 1
    while i < end:
        raw = lines[i]
        i += 1
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if ":" not in raw:
            raise ValueError(f"bad frontmatter line: {raw!r}")
        key, _, value = raw.partition(":")
        key, value = key.strip(), value.strip()
        if value in (">-", ">", "|-", "|"):  # YAML block scalar
            parts = []
            while i < end and (lines[i].startswith("  ")
                               or not lines[i].strip()):
                parts.append(lines[i].strip())
                i += 1
            value = " ".join(p for p in parts if p)
        if len(value) >= 2 and value[0] == value[-1] and \
                value[0] in ("'", '"'):
            value = value[1:-1]  # YAML quoted scalar
        meta[key] = value
    return meta, "\n".join(lines[end + 1:])


def _check_frontmatter(skill, meta, errors):
    name = meta.get("name")
    if not name:
        errors.append("frontmatter missing `name`")
    elif name != skill.name:
        errors.append(f"name {name!r} != directory {skill.name!r}")
    desc = meta.get("description", "")
    if "use when" not in desc.lower():
        errors.append("description must state trigger conditions "
                      "(\"Use when ...\")")
    stype = meta.get("type", "atomic") or "atomic"
    if stype not in _VALID_TYPES:
        errors.append(f"type {stype!r} not in {sorted(_VALID_TYPES)}")
    guard = meta.get("guard")
    if guard:
        try:
            g = json.loads(guard)
            if not isinstance(g.get("require_authorization"), bool):
                errors.append("guard.require_authorization must be a bool")
            if not isinstance(g.get("forbidden", []), list):
                errors.append("guard.forbidden must be a list")
        except json.JSONDecodeError as e:
            errors.append(f"guard is not valid JSON: {e}")
    return stype


def _check_sections(body, stype, errors):
    headings = [ln.strip() for ln in body.splitlines()
                if ln.startswith("#")]
    if not any(re.match(r"^# \S", h) for h in headings):
        errors.append("missing `# Title` heading")
    pos = -1
    for req in _REQUIRED_SECTIONS[stype]:
        found = [i for i, h in enumerate(headings) if h.startswith(req)]
        if not found:
            errors.append(f"missing required section `{req}`")
        else:
            if found[0] <= pos:
                errors.append(f"section `{req}` out of fixed order")
            pos = max(pos, found[0])
    m = re.search(r"^## Pitfalls\s*$([\s\S]*?)(?=^## |\Z)", body, re.M)
    if m and len(_BULLET_RE.findall(m.group(1))) < 3:
        errors.append("pitfalls section needs at least 3 bullets")


def _check_links(skill, body, all_skills, errors, where):
    for target in _LINK_RE.findall(body):
        if "<" in target:
            continue  # placeholder, never resolved
        if "/" in target:
            sname, _, fname = target.partition("/")
            ref = _SKILLS_DIR / sname / "references" / fname
            if not (ref.exists() or ref.with_suffix(".md").exists()):
                errors.append(f"{where}: dead cross-skill link "
                              f"[[{target}]]")
        else:
            own = skill / "references" / target
            if target not in all_skills and not (
                    own.exists() or own.with_suffix(".md").exists()):
                errors.append(f"{where}: dead link [[{target}]] (not a "
                              "skill, not in own references/)")


def validate():
    """Return a list of error strings (empty == clean)."""
    errors = []
    if not _SKILLS_DIR.is_dir():
        return errors
    skills = sorted(p for p in _SKILLS_DIR.iterdir() if p.is_dir())
    all_skills = {p.name for p in skills}
    n_entry = 0
    for skill in skills:
        doc = skill / "SKILL.md"
        where = f"skills/{skill.name}"
        if not doc.is_file():
            errors.append(f"{where}: missing SKILL.md")
            continue
        text = doc.read_text(encoding="utf-8")
        try:
            meta, body = _parse_frontmatter(text)
        except ValueError as e:
            errors.append(f"{where}: {e}")
            continue
        stype = _check_frontmatter(skill, meta, local := [])
        errors.extend(f"{where}: {e}" for e in local)
        if stype == "entry":
            n_entry += 1
        _check_sections(body, stype, local := [])
        errors.extend(f"{where}: {e}" for e in local)
        local = []
        _check_links(skill, body, all_skills, local, f"{where}/SKILL.md")
        refs = skill / "references"
        if refs.is_dir():
            for ref in sorted(refs.glob("*.md")):
                _check_links(skill, ref.read_text(encoding="utf-8"),
                             all_skills, local,
                             f"{where}/references/{ref.name}")
        errors.extend(local)
    if n_entry > 1:
        errors.append(f"expected at most one entry skill, found {n_entry}")
    return errors


def main():
    if not _SKILLS_DIR.is_dir():
        print("validate_skills: no skills/ directory — gate inert, PASS")
        return 0
    errors = validate()
    skills = sorted(p.name for p in _SKILLS_DIR.iterdir() if p.is_dir())
    if errors:
        print(f"validate_skills: FAIL ({len(errors)} error(s) across "
              f"{len(skills)} skill(s))")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"validate_skills: PASS ({len(skills)} skill(s): "
          f"{', '.join(skills)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
