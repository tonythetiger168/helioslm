"""Authentication and rate limiting for API service."""
import time
import hashlib
from typing import Dict, Optional, Set
from collections import defaultdict, deque


class APIKeyValidator:
    """Validate API keys."""

    def __init__(self, valid_keys: Optional[Set[str]] = None):
        self.valid_keys = valid_keys or set()
        self.key_usage: Dict[str, int] = defaultdict(int)

    def validate(self, api_key: str) -> bool:
        """Check if API key is valid."""
        if not self.valid_keys:
            return True  # No validation if no keys configured

        is_valid = api_key in self.valid_keys
        if is_valid:
            self.key_usage[api_key] += 1
        return is_valid

    def generate_key(self, prefix: str = "hlm") -> str:
        """Generate a new API key."""
        random_part = hashlib.sha256(str(time.time()).encode()).hexdigest()[:24]
        return f"{prefix}_{random_part}"

    def revoke_key(self, api_key: str):
        """Revoke an API key."""
        self.valid_keys.discard(api_key)
        if api_key in self.key_usage:
            del self.key_usage[api_key]


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
        burst_size: int = 10,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.burst_size = burst_size

        # Per-client state
        self.request_buckets: Dict[str, deque] = defaultdict(deque)
        self.token_buckets: Dict[str, float] = defaultdict(lambda: tokens_per_minute)
        self.last_update: Dict[str, float] = defaultdict(float)

    def check_rate_limit(self, client_id: str, requested_tokens: int = 0) -> tuple:
        """
        Check if request is within rate limit.

        Returns: (allowed: bool, retry_after: Optional[int])
        """
        now = time.time()

        # Update token bucket
        if client_id in self.last_update:
            elapsed = now - self.last_update[client_id]
            self.token_buckets[client_id] = min(
                self.tokens_per_minute,
                self.token_buckets[client_id] + elapsed * (self.tokens_per_minute / 60)
            )
        self.last_update[client_id] = now

        # Check request rate
        request_window = self.request_buckets[client_id]
        # Remove old requests (> 60 seconds)
        while request_window and request_window[0] < now - 60:
            request_window.popleft()

        if len(request_window) >= self.requests_per_minute:
            retry_after = int(60 - (now - request_window[0]))
            return False, retry_after

        # Check token rate
        if requested_tokens > self.token_buckets[client_id]:
            retry_after = int((requested_tokens - self.token_buckets[client_id]) / (self.tokens_per_minute / 60))
            return False, max(retry_after, 1)

        # Allow request
        request_window.append(now)
        self.token_buckets[client_id] -= requested_tokens

        return True, None

    def get_stats(self, client_id: str) -> Dict:
        """Get rate limit stats for a client."""
        now = time.time()
        request_window = self.request_buckets[client_id]
        recent_requests = sum(1 for t in request_window if t > now - 60)

        return {
            "requests_remaining": max(0, self.requests_per_minute - recent_requests),
            "tokens_remaining": int(self.token_buckets[client_id]),
            "reset_time": int(request_window[0] + 60) if request_window else int(now + 60),
        }
