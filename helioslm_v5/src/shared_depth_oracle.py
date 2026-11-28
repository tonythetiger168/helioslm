"""Experimental fixed-depth shared block oracle (issue #298).

Not wired into HeliosLMv5 serving or checkpoint formats.
"""
from torch import nn


class SharedDepth(nn.Module):
    """Repeat one block with separate caller-owned cache slots.

    Block contract: block(hidden, cache=..., use_cache=...) -> (hidden, cache).
    No cache is retained on this module between requests.
    """

    def __init__(self, block: nn.Module, repeats: int = 2):
        super().__init__()
        if type(repeats) is not int or repeats not in (1, 2):
            raise ValueError("experimental repeats must be 1 or 2")
        self.block = block
        self.repeats = repeats

    def forward(self, hidden, caches=None, *, use_cache=False):
        if caches is not None:
            if not use_cache:
                raise ValueError("input caches require use_cache=True")
            if not isinstance(caches, (tuple, list)) or len(caches) != self.repeats:
                raise ValueError("one cache slot per logical iteration required")
        else:
            caches = (None,) * self.repeats
        present = []
        for slot in range(self.repeats):
            hidden, cache = self.block(hidden, cache=caches[slot], use_cache=use_cache)
            if use_cache:
                if cache is None:
                    raise ValueError("block did not return a cache")
                present.append(cache)
        return hidden, tuple(present) if use_cache else None
