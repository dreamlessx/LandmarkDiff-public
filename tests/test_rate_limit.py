"""Tests for token bucket rate limiter."""

from __future__ import annotations

import time

import pytest

from landmarkdiff.rate_limit import RateLimiter, TokenBucket


class TestTokenBucket:
    def test_initial_tokens_equal_burst(self):
        bucket = TokenBucket(rate=10.0, burst=20)
        assert bucket.tokens == 20.0

    def test_consume_reduces_tokens(self):
        bucket = TokenBucket(rate=10.0, burst=5)
        assert bucket.consume() is True
        assert bucket.tokens < 5.0

    def test_consume_fails_when_empty(self):
        bucket = TokenBucket(rate=1.0, burst=2)
        assert bucket.consume(2) is True
        assert bucket.consume() is False

    def test_refill_adds_tokens(self):
        bucket = TokenBucket(rate=1000.0, burst=100)
        bucket.tokens = 0
        bucket.last_refill = time.monotonic() - 0.1  # 100ms ago
        bucket.refill()
        assert bucket.tokens >= 90  # ~100 tokens at 1000/s in 0.1s

    def test_refill_caps_at_burst(self):
        bucket = TokenBucket(rate=1000.0, burst=10)
        bucket.tokens = 5
        bucket.last_refill = time.monotonic() - 1.0  # 1s ago
        bucket.refill()
        assert bucket.tokens == 10  # capped at burst

    def test_available_after_refill(self):
        bucket = TokenBucket(rate=100.0, burst=50)
        bucket.tokens = 0
        bucket.last_refill = time.monotonic() - 0.5
        assert bucket.available >= 45  # ~50 tokens at 100/s in 0.5s


class TestRateLimiterInit:
    def test_default_config(self):
        limiter = RateLimiter()
        assert limiter.rate == 10.0
        assert limiter.burst == 20

    def test_custom_config(self):
        limiter = RateLimiter(rate=5.0, burst=10)
        assert limiter.rate == 5.0
        assert limiter.burst == 10

    def test_invalid_rate(self):
        with pytest.raises(ValueError, match="rate must be positive"):
            RateLimiter(rate=0)

    def test_invalid_burst(self):
        with pytest.raises(ValueError, match="burst must be >= 1"):
            RateLimiter(burst=0)


class TestRateLimiterAllow:
    def test_allows_within_burst(self):
        limiter = RateLimiter(rate=1.0, burst=5)
        for _ in range(5):
            assert limiter.allow("client1") is True

    def test_denies_after_burst_exhausted(self):
        limiter = RateLimiter(rate=1.0, burst=3)
        for _ in range(3):
            limiter.allow("client1")
        assert limiter.allow("client1") is False

    def test_independent_clients(self):
        limiter = RateLimiter(rate=1.0, burst=2)
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True
        assert limiter.allow("a") is False  # "a" exhausted
        assert limiter.allow("b") is True  # "b" independent

    def test_custom_cost(self):
        limiter = RateLimiter(rate=1.0, burst=10)
        assert limiter.allow("client", cost=8) is True
        assert limiter.allow("client", cost=5) is False  # only ~2 left

    def test_tokens_refill_over_time(self):
        limiter = RateLimiter(rate=1000.0, burst=5)
        # Exhaust tokens
        for _ in range(5):
            limiter.allow("client")
        assert limiter.allow("client") is False

        # Manually age the bucket
        limiter._buckets["client"].last_refill = time.monotonic() - 0.1
        assert limiter.allow("client") is True  # refilled


class TestRateLimiterRemaining:
    def test_new_client_has_full_burst(self):
        limiter = RateLimiter(rate=10.0, burst=20)
        assert limiter.remaining("new_client") == 20

    def test_remaining_decreases_after_allow(self):
        limiter = RateLimiter(rate=1.0, burst=10)
        limiter.allow("client")
        assert limiter.remaining("client") < 10


class TestRateLimiterReset:
    def test_reset_restores_burst(self):
        limiter = RateLimiter(rate=1.0, burst=3)
        for _ in range(3):
            limiter.allow("client")
        assert limiter.allow("client") is False
        limiter.reset("client")
        assert limiter.allow("client") is True

    def test_reset_nonexistent_is_noop(self):
        limiter = RateLimiter()
        limiter.reset("nonexistent")  # should not raise


class TestRateLimiterStats:
    def test_initial_stats(self):
        limiter = RateLimiter(rate=5.0, burst=10)
        stats = limiter.stats
        assert stats["rate"] == 5.0
        assert stats["burst"] == 10
        assert stats["total_allowed"] == 0
        assert stats["total_denied"] == 0
        assert stats["deny_rate"] == 0.0

    def test_stats_tracking(self):
        limiter = RateLimiter(rate=1.0, burst=2)
        limiter.allow("client")
        limiter.allow("client")
        limiter.allow("client")  # denied
        stats = limiter.stats
        assert stats["total_allowed"] == 2
        assert stats["total_denied"] == 1
        assert stats["deny_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_active_clients(self):
        limiter = RateLimiter(rate=10.0, burst=20)
        limiter.allow("a")
        limiter.allow("b")
        limiter.allow("c")
        assert limiter.active_clients == 3


class TestRateLimiterCleanup:
    def test_cleanup_removes_full_buckets(self):
        limiter = RateLimiter(rate=10.0, burst=5, cleanup_interval=0)
        limiter.allow("client")
        # Manually fill the bucket
        limiter._buckets["client"].tokens = 5
        # Trigger cleanup
        limiter._last_cleanup = 0
        limiter.allow("other")  # triggers _maybe_cleanup
        assert "client" not in limiter._buckets

    def test_cleanup_keeps_active_buckets(self):
        limiter = RateLimiter(rate=1.0, burst=5, cleanup_interval=0)
        for _ in range(5):
            limiter.allow("active")
        limiter._last_cleanup = 0
        limiter.allow("other")
        assert "active" in limiter._buckets  # still depleted
