"""HeliosLM Phase 5 — Self-Reflection & Improvement

Meta-cognitive capabilities:
  - Self-evaluation of responses
  - Error detection and correction
  - Strategy learning from failures
  - Confidence calibration
"""
from .self_reflection import SelfReflector, ReflectionResult
from .strategy_learner import StrategyLearner

__all__ = ["SelfReflector", "ReflectionResult", "StrategyLearner"]
