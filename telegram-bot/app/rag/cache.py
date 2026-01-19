"""LRU cache for query embeddings."""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CachedEmbedding:
    """Cached embedding data."""

    dense: list[float]
    sparse: dict[int, float]


class EmbeddingCache:
    """
    LRU cache for query embeddings.

    Caches embedding results to avoid re-computing vectors for identical queries.
    Uses OrderedDict for O(1) access with LRU eviction.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of entries to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CachedEmbedding] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        """
        Create cache key from query text.

        Args:
            query: Query text

        Returns:
            MD5 hash of normalized query
        """
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> CachedEmbedding | None:
        """
        Retrieve cached embedding for query.

        Args:
            query: Query text

        Returns:
            CachedEmbedding if found, None otherwise
        """
        key = self._make_key(query)

        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        return None

    def put(
        self, query: str, dense: list[float], sparse: dict[int, float]
    ) -> None:
        """
        Store embedding in cache.

        Args:
            query: Query text
            dense: Dense embedding vector
            sparse: Sparse embedding vector
        """
        key = self._make_key(query)

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = CachedEmbedding(dense=dense, sparse=sparse)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CachedEmbedding(dense=dense, sparse=sparse)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """
        Get cache hit rate.

        Returns:
            Hit rate as a percentage (0.0 to 1.0)
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, size, and hit_rate
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }
