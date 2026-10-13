"""CI PROBE: skip the suite to bisect a py3.11-only CI failure.

Temporary debug commit on debug/ci-py311-probe ONLY. Never merge this.
"""
import sys

print('PROBE: test_v5.py stubbed out (exit 0)')
sys.exit(0)
