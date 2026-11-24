"""Decode-strategy layer (roadmap 2026Q4-v2, improvement 5).

Separates *inference strategy* (this module) from *model / training
structure* (the ``mtp`` block in ``configs.config_v5`` and the MTP
modules on ``HeliosLMv5``). The two knobs that select a decode path —
``use_mtp`` (which machinery runs: the plain loop or speculative MTP
decoding) and ``thinking_budget`` (a per-row strategy object from
``inference.thinking_budget``) — used to be validated inline inside
``HeliosLMv5.generate``. The rule now lives here, in the strategy
layer:

- ``DecodeStrategy`` is the single interpretation of the two knobs;
  constructing one validates the combination;
- ``HeliosLMv5.generate`` remains a thin compatibility facade: it
  builds a ``DecodeStrategy`` from its keyword arguments (so the rule
  fires at the same boundary, with the same ``ValueError`` type) and
  dispatches on it. The default strategy (``DecodeStrategy()``) is the
  plain path, bit-identical to the pre-refactor default.

Why the exclusion exists: the MTP path delegates decoding to
``MTPDecoder``, which drives the model's forward directly and never
consults per-row logit processors — a thinking budget would be
*silently ignored* there. Silently dropping a strategy object is worse
than refusing the combination, so the strategy layer refuses it loudly,
with a message that names both knobs and the fix.
"""
from dataclasses import dataclass
from typing import Optional

from helioslm_v5.src.inference.thinking_budget import ThinkingBudget

__all__ = ["DecodeStrategy", "STRATEGY_CONFLICT_MSG", "validate_decode_strategy"]


STRATEGY_CONFLICT_MSG = (
    "conflicting decode strategy: thinking_budget requires the plain "
    "decode path, but use_mtp=True selects MTP speculative decoding "
    "(MTPDecoder drives forward() directly and never applies per-row "
    "logit processors, so the budget would be silently ignored). "
    "Fix: pass thinking_budget only with use_mtp=False, or drop the "
    "budget to use MTP."
)


def validate_decode_strategy(use_mtp, thinking_budget):
    """Single owner of the ``use_mtp`` x ``thinking_budget`` exclusion.

    Raises ``ValueError`` iff the combination selects MTP speculative
    decoding *and* attaches a thinking budget (which the MTP path would
    silently drop).
    """
    if use_mtp and thinking_budget is not None:
        raise ValueError(STRATEGY_CONFLICT_MSG)


@dataclass(frozen=True)
class DecodeStrategy:
    """Validated combination of decode-path selection and strategy objects.

    ``DecodeStrategy()`` (``use_mtp=False``, ``thinking_budget=None``)
    is the plain decode path and is bit-identical to the pre-refactor
    ``HeliosLMv5.generate()`` default.
    """

    use_mtp: bool = False
    thinking_budget: Optional[ThinkingBudget] = None

    def __post_init__(self):
        validate_decode_strategy(self.use_mtp, self.thinking_budget)
