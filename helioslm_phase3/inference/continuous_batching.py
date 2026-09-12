"""Continuous batching for maximum GPU utilization.

Unlike static batching (wait for batch to fill), continuous batching
adds/removes requests dynamically as sequences complete.

Key insight: When one sequence in a batch generates EOS, immediately
replace it with a new request rather than waiting for the whole batch.
"""
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import torch


@dataclass
class BatchRequest:
    """A single inference request."""
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    arrival_time: float = 0.0
    priority: int = 0  # Higher = more urgent

    # State
    generated_tokens: List[int] = None
    is_finished: bool = False

    def __post_init__(self):
        if self.generated_tokens is None:
            self.generated_tokens = []
        if self.arrival_time == 0.0:
            self.arrival_time = time.time()


class ContinuousBatcher:
    """
    Continuous batching scheduler.

    Maintains a queue of pending requests and dynamically
    manages the active batch to maximize GPU utilization.
    """

    def __init__(
        self,
        engine,
        max_batch_size: int = 32,
        max_seq_len: int = 32768,
        max_wait_time: float = 0.1,  # Max time to wait for batch to fill
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_wait_time = max_wait_time

        self.pending_queue: List[BatchRequest] = []
        self.active_requests: Dict[str, BatchRequest] = {}
        self.completed_requests: Dict[str, BatchRequest] = {}

        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0

    def add_request(self, request: BatchRequest):
        """Add a request to the pending queue."""
        self.pending_queue.append(request)
        # Sort by priority (higher first), then FIFO
        self.pending_queue.sort(key=lambda r: (-r.priority, r.arrival_time))

    def schedule(self) -> List[BatchRequest]:
        """
        Select requests to form the next batch.

        Strategy:
          1. Fill batch up to max_batch_size
          2. Prefer requests that fit within max_seq_len
          3. Don't wait longer than max_wait_time for batch to fill
        """
        batch = []
        current_len = 0

        # Add already active requests that aren't finished
        for req in list(self.active_requests.values()):
            if not req.is_finished:
                seq_len = len(req.prompt_tokens) + len(req.generated_tokens) + 1
                if seq_len <= self.max_seq_len:
                    batch.append(req)
                    current_len = max(current_len, seq_len)

        # Add new requests from queue
        remaining = []
        for req in self.pending_queue:
            if len(batch) >= self.max_batch_size:
                remaining.append(req)
                continue

            seq_len = len(req.prompt_tokens) + 1
            if current_len > 0 and seq_len > current_len * 1.5:
                # Skip very long requests to avoid padding waste
                remaining.append(req)
                continue

            batch.append(req)
            self.active_requests[req.request_id] = req
            current_len = max(current_len, seq_len)

        self.pending_queue = remaining
        return batch

    def step(self) -> Dict[str, List[int]]:
        """Execute one decoding step."""
        batch = self.schedule()
        if not batch:
            return {}

        # Run engine step
        outputs = self.engine.step_batch(batch)

        # Update request states
        for req in batch:
            if req.request_id in outputs:
                new_tokens = outputs[req.request_id]
                req.generated_tokens.extend(new_tokens)
                self.total_tokens_generated += len(new_tokens)

                # Check finish conditions
                if len(req.generated_tokens) >= req.max_tokens:
                    req.is_finished = True
                if new_tokens and new_tokens[-1] == 2:  # EOS
                    req.is_finished = True

                if req.is_finished:
                    self.completed_requests[req.request_id] = req
                    if req.request_id in self.active_requests:
                        del self.active_requests[req.request_id]

        return outputs

    def run(self) -> Dict[str, List[int]]:
        """Run until all requests complete."""
        all_results = {}

        while self.pending_queue or self.active_requests:
            outputs = self.step()
            all_results.update(outputs)

        return all_results

    def get_stats(self) -> Dict:
        """Get batching statistics."""
        total_reqs = len(self.completed_requests)
        avg_latency = 0
        if total_reqs > 0:
            latencies = [
                time.time() - r.arrival_time
                for r in self.completed_requests.values()
            ]
            avg_latency = sum(latencies) / len(latencies)

        return {
            "total_requests": total_reqs,
            "total_tokens_generated": self.total_tokens_generated,
            "total_prompt_tokens": self.total_prompt_tokens,
            "avg_latency_sec": avg_latency,
            "throughput_tok_per_sec": self.total_tokens_generated / max(avg_latency, 0.001),
        }
