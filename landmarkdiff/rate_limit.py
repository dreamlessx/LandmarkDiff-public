"""Token bucket rate limiter for API endpoints.

Provides configurable per-client rate limiting with burst support.
Suitable for use as FastAPI middleware or standalone.

Usage:
    from landmarkdiff.rate_limit import RateLimiter

    limiter = RateLimiter(rate=10.0, burst=20)
    if limiter.allow("client_ip"):
        handle_request()
    else:
        return 429, "Too Many Requests"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for a single client.

    Tokens refill at a constant rate up to the burst capacity.
    Each request consumes one token.

    Args:
        rate: Tokens added per second.
        burst: Maximum token capacity.
    """

    rate: float
    burst: int
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = float(self.burst)

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def consume(self, n: int = 1) -> bool:
        """Try to consume n tokens.

        Args:
            n: Number of tokens to consume.

        Returns:
            True if tokens were available and consumed.
        """
        self.refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    @property
    def available(self) -> int:
        """Number of tokens currently available (after refill)."""
        self.refill()
        return int(self.tokens)


class RateLimiter:
    """Per-client token bucket rate limiter.

    Each client (identified by a string key such as IP address or
    API key) gets an independent token bucket with the configured
    rate and burst capacity.

    Args:
        rate: Requests per second allowed per client.
        burst: Maximum burst size (bucket capacity).
        cleanup_interval: Seconds between stale bucket cleanup.
    """

    def __init__(
        self,
        rate: float = 10.0,
        burst: int = 20,
        cleanup_interval: float = 300.0,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")

        self.rate = rate
        self.burst = burst
        self.cleanup_interval = cleanup_interval
        self._buckets: dict[str, TokenBucket] = {}
        self._last_cleanup = time.monotonic()
        self._total_allowed = 0
        self._total_denied = 0

    def allow(self, client_id: str, cost: int = 1) -> bool:
        """Check if a request from the given client is allowed.

        Args:
            client_id: Client identifier (IP, API key, etc.).
            cost: Number of tokens this request costs.

        Returns:
            True if the request is allowed, False if rate limited.
        """
        self._maybe_cleanup()

        bucket = self._buckets.get(client_id)
        if bucket is None:
            bucket = TokenBucket(rate=self.rate, burst=self.burst)
            self._buckets[client_id] = bucket

        if bucket.consume(cost):
            self._total_allowed += 1
            return True

        self._total_denied += 1
        logger.debug("Rate limited: %s", client_id)
        return False

    def remaining(self, client_id: str) -> int:
        """Get remaining tokens for a client.

        Args:
            client_id: Client identifier.

        Returns:
            Number of available tokens, or burst if client is new.
        """
        bucket = self._buckets.get(client_id)
        if bucket is None:
            return self.burst
        return bucket.available

    def reset(self, client_id: str) -> None:
        """Reset a client's token bucket to full capacity.

        Args:
            client_id: Client identifier to reset.
        """
        if client_id in self._buckets:
            del self._buckets[client_id]

    def _maybe_cleanup(self) -> None:
        """Remove stale buckets that have been fully refilled."""
        now = time.monotonic()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        stale = [cid for cid, bucket in self._buckets.items() if bucket.available >= self.burst]
        for cid in stale:
            del self._buckets[cid]

        if stale:
            logger.debug("Cleaned up %d stale rate limit buckets", len(stale))
        self._last_cleanup = now

    @property
    def active_clients(self) -> int:
        """Number of clients with active rate limit buckets."""
        return len(self._buckets)

    @property
    def stats(self) -> dict[str, Any]:
        """Rate limiter statistics."""
        total = self._total_allowed + self._total_denied
        return {
            "rate": self.rate,
            "burst": self.burst,
            "active_clients": self.active_clients,
            "total_allowed": self._total_allowed,
            "total_denied": self._total_denied,
            "deny_rate": round(self._total_denied / total, 4) if total > 0 else 0.0,
        }
