"""CI gate: skills/ docs must satisfy docs/SKILL_TEMPLATE.md.

Runs validate_skills.py as a subprocess so the gate is identical locally
and in CI. Collected by the pytest step (named test_*.py); the test_v5.py
runner does not import it. Passes vacuously while skills/ does not exist.
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_skill_docs_validate():
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "validate_skills.py")],
        capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"validate_skills.py failed:\n{proc.stdout}\n{proc.stderr}")
