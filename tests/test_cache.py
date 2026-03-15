"""Tests for prediction result caching."""

from __future__ import annotations

import time

import numpy as np
import pytest

from landmarkdiff.cache import CacheEntry, PredictionCache


@pytest.fixture
def small_cache():
    return PredictionCache(max_size=3, ttl=0)


@pytest.fixture
def sample_image():
    return np.zeros((64, 64, 3), dtype=np.uint8)


class TestCacheEntry:
    def test_age_increases(self):
        entry = CacheEntry(key="test", value="val", created_at=time.time() - 10)
        assert entry.age_seconds >= 10

    def test_default_hit_count(self):
        entry = CacheEntry(key="test", value="val")
        assert entry.hit_count == 0


class TestMakeKey:
    def test_deterministic(self, sample_image):
        key1 = PredictionCache.make_key(sample_image, "rhinoplasty", 65.0)
        key2 = PredictionCache.make_key(sample_image, "rhinoplasty", 65.0)
        assert key1 == key2

    def test_different_procedure_different_key(self, sample_image):
        k1 = PredictionCache.make_key(sample_image, "rhinoplasty", 65.0)
        k2 = PredictionCache.make_key(sample_image, "blepharoplasty", 65.0)
        assert k1 != k2

    def test_different_intensity_different_key(self, sample_image):
        k1 = PredictionCache.make_key(sample_image, "rhinoplasty", 50.0)
        k2 = PredictionCache.make_key(sample_image, "rhinoplasty", 80.0)
        assert k1 != k2

    def test_different_image_different_key(self):
        img1 = np.zeros((32, 32, 3), dtype=np.uint8)
        img2 = np.ones((32, 32, 3), dtype=np.uint8) * 128
        k1 = PredictionCache.make_key(img1, "rhinoplasty", 65.0)
        k2 = PredictionCache.make_key(img2, "rhinoplasty", 65.0)
        assert k1 != k2

    def test_different_seed_different_key(self, sample_image):
        k1 = PredictionCache.make_key(sample_image, seed=1)
        k2 = PredictionCache.make_key(sample_image, seed=2)
        assert k1 != k2

    def test_key_length(self, sample_image):
        key = PredictionCache.make_key(sample_image)
        assert len(key) == 16  # truncated SHA-256

    def test_extra_params(self, sample_image):
        k1 = PredictionCache.make_key(sample_image, mode="tps")
        k2 = PredictionCache.make_key(sample_image, mode="controlnet")
        assert k1 != k2


class TestPredictionCache:
    def test_put_and_get(self, small_cache):
        small_cache.put("k1", "value1")
        assert small_cache.get("k1") == "value1"

    def test_miss_returns_none(self, small_cache):
        assert small_cache.get("nonexistent") is None

    def test_lru_eviction(self, small_cache):
        small_cache.put("a", 1)
        small_cache.put("b", 2)
        small_cache.put("c", 3)
        # cache is full (max_size=3), adding one more evicts "a"
        small_cache.put("d", 4)
        assert small_cache.get("a") is None
        assert small_cache.get("b") == 2
        assert small_cache.get("d") == 4

    def test_access_prevents_eviction(self, small_cache):
        small_cache.put("a", 1)
        small_cache.put("b", 2)
        small_cache.put("c", 3)
        # Access "a" to move it to end
        small_cache.get("a")
        # Now "b" is the LRU entry
        small_cache.put("d", 4)
        assert small_cache.get("a") == 1  # still present
        assert small_cache.get("b") is None  # evicted

    def test_update_existing_key(self, small_cache):
        small_cache.put("k1", "old")
        small_cache.put("k1", "new")
        assert small_cache.get("k1") == "new"
        assert small_cache.size == 1

    def test_size_tracking(self, small_cache):
        assert small_cache.size == 0
        small_cache.put("a", 1)
        assert small_cache.size == 1
        small_cache.put("b", 2)
        assert small_cache.size == 2

    def test_invalidate(self, small_cache):
        small_cache.put("k1", "val")
        assert small_cache.invalidate("k1") is True
        assert small_cache.get("k1") is None
        assert small_cache.size == 0

    def test_invalidate_nonexistent(self, small_cache):
        assert small_cache.invalidate("nope") is False

    def test_clear(self, small_cache):
        small_cache.put("a", 1)
        small_cache.put("b", 2)
        removed = small_cache.clear()
        assert removed == 2
        assert small_cache.size == 0


class TestCacheTTL:
    def test_expired_entry_returns_none(self):
        cache = PredictionCache(max_size=10, ttl=1)
        cache.put("k1", "val")
        # Manually age the entry
        cache._cache["k1"].created_at = time.time() - 2
        assert cache.get("k1") is None

    def test_non_expired_entry_returns_value(self):
        cache = PredictionCache(max_size=10, ttl=60)
        cache.put("k1", "val")
        assert cache.get("k1") == "val"

    def test_evict_expired(self):
        cache = PredictionCache(max_size=10, ttl=1)
        cache.put("old", "val")
        cache._cache["old"].created_at = time.time() - 2
        cache.put("fresh", "val2")
        evicted = cache.evict_expired()
        assert evicted == 1
        assert cache.get("old") is None
        assert cache.get("fresh") == "val2"

    def test_evict_expired_no_ttl(self):
        cache = PredictionCache(max_size=10, ttl=0)
        cache.put("k1", "val")
        assert cache.evict_expired() == 0


class TestCacheStats:
    def test_hit_rate_empty(self):
        cache = PredictionCache()
        assert cache.hit_rate == 0.0

    def test_hit_rate_tracking(self):
        cache = PredictionCache(max_size=10, ttl=0)
        cache.put("k1", "val")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        assert cache.hit_rate == 0.5

    def test_stats_dict(self):
        cache = PredictionCache(max_size=5, ttl=60)
        cache.put("k1", "val")
        cache.get("k1")
        stats = cache.stats
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["ttl"] == 60

    def test_clear_resets_stats(self):
        cache = PredictionCache(max_size=10, ttl=0)
        cache.put("k1", "val")
        cache.get("k1")
        cache.get("miss")
        cache.clear()
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0


class TestCacheSave:
    def test_save_creates_file(self, tmp_path):
        cache = PredictionCache(max_size=10, ttl=0)
        cache.put("k1", "val1")
        cache.put("k2", "val2")
        path = tmp_path / "cache_meta.json"
        cache.save(path)
        assert path.exists()

    def test_save_contains_keys(self, tmp_path):
        import json

        cache = PredictionCache(max_size=10, ttl=0)
        cache.put("abc", "val")
        path = tmp_path / "cache_meta.json"
        cache.save(path)
        data = json.loads(path.read_text())
        assert "abc" in data["keys"]
        assert data["stats"]["size"] == 1
