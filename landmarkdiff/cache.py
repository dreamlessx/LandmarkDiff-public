"""Prediction result caching for the inference pipeline.

Provides an LRU cache keyed on image content hash + prediction parameters,
with optional disk persistence for cross-session reuse.

Usage:
    from landmarkdiff.cache import PredictionCache

    cache = PredictionCache(max_size=100)
    key = cache.make_key(image, procedure="rhinoplasty", intensity=65.0)

    if (result := cache.get(key)) is not None:
        return result  # cache hit

    result = run_prediction(image)
    cache.put(key, result)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 50
_DEFAULT_TTL = 3600  # 1 hour


@dataclass
class CacheEntry:
    """A single cached prediction result."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class PredictionCache:
    """LRU cache for prediction results.

    Avoids redundant predictions when the same image + parameters
    are requested multiple times. Uses content-based hashing so
    identical images produce the same cache key regardless of
    file path.

    Args:
        max_size: Maximum number of cached results.
        ttl: Time-to-live in seconds (0 = no expiry).
    """

    def __init__(
        self,
        max_size: int = _DEFAULT_MAX_SIZE,
        ttl: float = _DEFAULT_TTL,
    ) -> None:
        self.max_size = max(1, max_size)
        self.ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(
        image: np.ndarray,
        procedure: str = "",
        intensity: float = 0.0,
        seed: int = 0,
        **extra: Any,
    ) -> str:
        """Generate a cache key from image content and parameters.

        Uses SHA-256 of the image bytes combined with prediction
        parameters to produce a deterministic key.

        Args:
            image: Input image array.
            procedure: Procedure name.
            intensity: Intensity value.
            seed: Random seed.
            **extra: Additional parameters to include in the key.

        Returns:
            Hex digest cache key.
        """
        h = hashlib.sha256()
        h.update(image.tobytes())
        params = {
            "procedure": procedure,
            "intensity": intensity,
            "seed": seed,
            **extra,
        }
        h.update(json.dumps(params, sort_keys=True).encode())
        return h.hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        """Retrieve a cached result.

        Moves the entry to the end of the LRU order on hit.
        Returns None on miss or if the entry has expired.

        Args:
            key: Cache key from make_key().

        Returns:
            Cached value or None.
        """
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if self.ttl > 0 and entry.age_seconds > self.ttl:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.last_accessed = time.time()
        entry.hit_count += 1
        self._hits += 1
        return entry.value

    def put(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a result in the cache.

        Evicts the least recently used entry if the cache is full.

        Args:
            key: Cache key from make_key().
            value: Value to cache.
            metadata: Optional metadata to store with the entry.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key].value = value
            return

        if len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("Cache eviction: %s", evicted_key)

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            metadata=metadata or {},
        )

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            True if the entry existed and was removed.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
            Number of entries removed.
        """
        count = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        return count

    def evict_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries evicted.
        """
        if self.ttl <= 0:
            return 0

        expired = [k for k, v in self._cache.items() if v.age_seconds > self.ttl]
        for k in expired:
            del self._cache[k]
        return len(expired)

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 - 1.0)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics summary."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl": self.ttl,
        }

    def save(self, path: str | Path) -> None:
        """Save cache metadata to disk (keys and stats, not values).

        Args:
            path: File path for the JSON metadata.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stats": self.stats,
            "keys": list(self._cache.keys()),
        }
        out.write_text(json.dumps(data, indent=2))
        logger.info("Saved cache metadata to %s", out)
