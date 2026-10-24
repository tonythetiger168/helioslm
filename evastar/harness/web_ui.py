"""Deprecated shim: the v1.36 web UI moved to harness/legacy/web_ui_v1.py
(#54). Use evastar/harness/web_ui_v2.py.

``from harness.web_ui import ...`` and ``python evastar/harness/web_ui.py``
keep working: importing this module returns the archived module itself.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.warn("harness.web_ui (v1.36) is archived; use harness/web_ui_v2.py",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    import runpy
    runpy.run_module("harness.legacy.web_ui_v1", run_name="__main__")
else:
    from harness.legacy import web_ui_v1 as _v1
    sys.modules[__name__] = _v1
