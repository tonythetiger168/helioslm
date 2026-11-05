"""HeliosLM release version: the single source of truth (v5.236).

Before v5.236 the version lived in three places that drifted apart:
``pyproject.toml`` (5.234.0), ``HeliosLMv5Config.model_name`` (v5.225)
and the CHANGELOG head (v5.235). Now ``pyproject.toml`` reads this file
(``[tool.setuptools.dynamic]``), ``model_name`` is built from it, and
``case_harness_repair.test_version_matches_changelog`` fails CI unless
it equals the newest ``## vX.Y`` heading in ``helioslm_v5/CHANGELOG.md``.

A PR that adds a CHANGELOG entry bumps this one line too.
"""

__version__ = "5.239"
