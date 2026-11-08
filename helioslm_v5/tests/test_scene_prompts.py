"""T170 - scene prompt pairs (data/scene_prompts): schema, provenance fields and manifest digest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "scene_prompts"
SLUGS = {"bamboo", "library", "alley", "canyon", "lake", "ruins", "station", "nightmarket"}


def _rows():
    return [json.loads(line) for line in (D / "scene_prompts.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_scene_prompts_manifest():
    man = json.loads((D / "MANIFEST.json").read_text())
    data = (D / "scene_prompts.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == man["sha256"]
    assert len(_rows()) == man["records"]
    print("PASS test_scene_prompts_manifest")


def test_scene_prompts_schema():
    rows = _rows()
    main = [r for r in rows if r["split"] == "main"]
    assert {r["id"] for r in main} == SLUGS, "one main pair per showcase sentence"
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert set(r) == {"id", "input", "output", "split", "generator", "world_id_prefix"}
        assert r["split"] in ("main", "variant")
        assert 5 <= len(r["input"].split()) <= 20 and len(r["output"].split()) >= 100
        assert len(r["world_id_prefix"]) == 8 and int(r["world_id_prefix"], 16) >= 0
        assert r["output"].startswith("The scene is")
    print("PASS test_scene_prompts_schema")


if __name__ == "__main__":
    test_scene_prompts_manifest()
    test_scene_prompts_schema()
