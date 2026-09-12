"""Data mixing with temperature sampling for balanced pre-training."""
import random
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass


@dataclass
class DataBlend:
    """A single data source blend configuration."""
    source_name: str
    target_tokens: int
    weight: float
    temperature: float = 1.0  # >1 = more uniform, <1 = more peaky


class DataMixer:
    """
    Mix multiple data sources with configurable sampling weights.

    Uses temperature-based sampling to control data diversity:
      - temperature=1.0: proportional to target tokens
      - temperature>1.0: more uniform (helps underrepresented sources)
      - temperature<1.0: more peaky (focus on high-quality sources)
    """

    def __init__(self, blends: List[DataBlend], total_tokens: int = 15_000_000_000_000):
        self.blends = blends
        self.total_tokens = total_tokens
        self._compute_sampling_weights()

    def _compute_sampling_weights(self):
        """Compute per-source sampling probabilities with temperature."""
        raw_weights = [b.weight * (b.target_tokens ** (1.0 / b.temperature)) for b in self.blends]
        total = sum(raw_weights)
        self.sampling_probs = [w / total for w in raw_weights]

        # Compute actual token allocation
        self.token_allocation = {
            b.source_name: int(self.total_tokens * prob)
            for b, prob in zip(self.blends, self.sampling_probs)
        }

    def sample_source(self) -> str:
        """Sample a data source according to mixing weights."""
        names = [b.source_name for b in self.blends]
        return random.choices(names, weights=self.sampling_probs, k=1)[0]

    def get_blend_config(self) -> Dict[str, Dict]:
        """Return the full blend configuration for logging."""
        return {
            b.source_name: {
                "target_tokens": b.target_tokens,
                "weight": b.weight,
                "temperature": b.temperature,
                "sampling_prob": prob,
                "allocated_tokens": self.token_allocation[b.source_name],
            }
            for b, prob in zip(self.blends, self.sampling_probs)
        }

    def create_streaming_iterator(
        self,
        sources: Dict[str, Iterator],
        buffer_size: int = 10000,
    ) -> Iterator[Dict]:
        """
        Create a streaming iterator that samples from multiple sources.

        Args:
            sources: Dict mapping source_name -> iterator
            buffer_size: Number of documents to buffer per source
        """
        buffers = {name: [] for name in sources}

        def refill_buffer(name: str):
            source_iter = sources[name]
            try:
                for _ in range(buffer_size):
                    buffers[name].append(next(source_iter))
            except StopIteration:
                pass

        # Initial fill
        for name in sources:
            refill_buffer(name)

        while any(len(buffers[name]) > 0 for name in sources):
            source_name = self.sample_source()

            if len(buffers[source_name]) == 0:
                # Try to refill
                refill_buffer(source_name)
                if len(buffers[source_name]) == 0:
                    continue

            doc = buffers[source_name].pop(0)
            doc["_source"] = source_name
            yield doc

            # Refill if buffer getting low
            if len(buffers[source_name]) < buffer_size // 4:
                refill_buffer(source_name)
