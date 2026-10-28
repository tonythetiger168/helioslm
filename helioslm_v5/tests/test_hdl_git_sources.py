"""T160 — HDL track H5: git sources for the SystemVerilog / UVM mixture,
v5.183.

Pins:

1. ``parse_source`` reads ``git:<https url>@<40-hex commit>`` and refuses
   other forms; a git source without ``include`` is an invalid mixture;
2. ``git_licence`` names Apache-2.0 for the Apache and Solderpad 2.x
   texts and MIT for the MIT text, None otherwise;
3. ``git_checkout`` checks out the pinned commit; ``git_texts`` keeps
   included files minus excluded ones, sorted, drops exact duplicates and
   files over 1 MB, and prefixes each with its path;
4. ``plan`` treats a git source like an hf one (its LICENSE file is the
   card licence);
5. ``data/mixtures/hdl_h5_sv_uvm_v1.json`` loads and pins full commits.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_hdl_git_sources.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from examples import fetch_mixture as fm  # noqa: E402
from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, validate_mixture)

C = "a" * 40
needs_git = pytest.mark.skipif(not shutil.which("git"), reason="needs git")


def test_parse_git_source():
    assert fm.parse_source(f"git:https://github.com/o/r@{C}") == (
        "git", "https://github.com/o/r", C)
    assert fm.parse_source("git:https://github.com/o/r@main") is None
    assert fm.parse_source(f"git:http://example.org/r@{C}") is None
    mix = {"name": "m", "sources": [
        {"name": "g", "source": f"git:https://github.com/o/r@{C}",
         "license": "Apache-2.0", "weight": 1.0}]}
    with pytest.raises(ValueError, match="include"):
        validate_mixture(mix)
    mix["sources"][0]["include"] = ["*.sv"]
    validate_mixture(mix)
    mix["sources"][0]["exclude"] = "vendor"
    with pytest.raises(ValueError, match="exclude"):
        validate_mixture(mix)


def test_git_licence(tmp_path):
    (tmp_path / "LICENSE").write_text("\n   Apache License\n   Version 2.0\n")
    assert fm.git_licence(tmp_path) == "Apache-2.0"
    (tmp_path / "LICENSE").unlink()
    (tmp_path / "LICENSE.md").write_text("# Solderpad Hardware Licence Version 2.0\n")
    assert fm.git_licence(tmp_path) == "Apache-2.0"
    (tmp_path / "LICENSE.md").write_text("MIT License\n\nCopyright (c) x\n")
    assert fm.git_licence(tmp_path) == "MIT"
    (tmp_path / "LICENSE.md").write_text("All rights reserved.\n")
    assert fm.git_licence(tmp_path) is None
    (tmp_path / "LICENSE.md").unlink()
    assert fm.git_licence(tmp_path) is None


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout


@needs_git
def test_checkout_and_texts(tmp_path):
    src = tmp_path / "src"
    (src / "rtl").mkdir(parents=True)
    (src / "dv" / "vendor").mkdir(parents=True)
    (src / "rtl" / "b.sv").write_text("module b; endmodule\n")
    (src / "rtl" / "a.sv").write_text("module a; endmodule\n")
    (src / "rtl" / "copy.sv").write_text("module a; endmodule\n")     # duplicate
    (src / "dv" / "env.svh").write_text("class env extends uvm_env; endclass\n")
    (src / "dv" / "vendor" / "x.sv").write_text("module x; endmodule\n")
    (src / "dv" / "big.sv").write_text("// " + "x" * fm.MAX_FILE_BYTES + "\n")
    (src / "README.md").write_text("readme\n")
    _git(src, "init", "-q")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "one")
    commit = _git(src, "rev-parse", "HEAD").strip()
    (src / "rtl" / "later.sv").write_text("module later; endmodule\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "two")
    _git(src, "config", "uploadpack.allowReachableSHA1InWant", "true")
    root = fm.git_checkout(src.as_uri(), commit, tmp_path / "co")
    texts = list(fm.git_texts(root, ["*.sv", "*.svh"], ["*vendor*"]))
    assert texts == ["// file: dv/env.svh\nclass env extends uvm_env; endclass\n",
                     "// file: rtl/a.sv\nmodule a; endmodule\n",
                     "// file: rtl/b.sv\nmodule b; endmodule\n"]


def test_plan_git_source():
    mix = {"name": "m", "sources": [
        {"name": "g", "source": f"git:https://github.com/o/r@{C}", "include": ["*.sv"],
         "license": "Apache-2.0", "weight": 0.5},
        {"name": "h", "source": "hf:x/h", "license": "MIT", "weight": 0.5}]}
    rows = fm.plan(mix, 1000, 0.2, {"g": "Apache-2.0", "h": "mit"})
    assert rows["g"]["usable"] and rows["g"]["budget"] == 500
    rows = fm.plan(mix, 1000, 0.2, {"g": "MIT", "h": "mit"})
    assert not rows["g"]["usable"] and rows["h"]["budget"] == 1000


def test_h5_mixture():
    mix = load_mixture(ROOT / "data" / "mixtures" / "hdl_h5_sv_uvm_v1.json")
    git = [s for s in mix["sources"] if s["source"].startswith("git:")]
    assert {s["name"] for s in git} == {"opentitan-sv", "ibex-sv", "cv32e40p-sv",
                                        "core-v-verif-sv"}
    for s in git:
        kind, url, commit = fm.parse_source(s["source"])
        assert kind == "git" and url.startswith("https://github.com/")
        assert len(commit) == 40 and "*vendor*" in s["exclude"]
    # the Stack: rows filtered by repository licence (v5.186)
    stack = next(s for s in mix["sources"] if s["name"] == "stack-systemverilog")
    assert stack["license"] == "per-repository"
    # an empty projects dict still means "gate every project" (v5.193)
    assert stack["projects"] == {} and fm.project_source(stack)
    assert all(fm.project_source(s) for s in mix["sources"]
               if s["name"] not in ("fineweb-edu", "finemath", "core-v-verif-sv"))
