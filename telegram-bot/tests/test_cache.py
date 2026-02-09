"""Tests for embedding cache."""

import pytest

from app.rag.cache import EmbeddingCache


class TestEmbeddingCache:
    """Test suite for EmbeddingCache."""

    def test_cache_initialization(self):
        """Verify cache initializes with correct max size."""
        cache = EmbeddingCache(max_size=100)

        assert cache.max_size == 100
        assert cache.size == 0

    def test_put_and_get(self):
        """Verify put and get operations work correctly."""
        cache = EmbeddingCache(max_size=10)

        dense = [0.1] * 1024
        sparse = {100: 0.5, 200: 0.3}

        cache.put("test query", dense, sparse)
        result = cache.get("test query")

        assert result is not None
        assert result.dense == dense
        assert result.sparse == sparse

    def test_get_nonexistent_returns_none(self):
        """Verify get returns None for missing keys."""
        cache = EmbeddingCache()

        result = cache.get("nonexistent query")

        assert result is None

    def test_cache_is_case_insensitive(self):
        """Verify cache normalizes query case."""
        cache = EmbeddingCache()

        cache.put("Test Query", [0.1], {})
        result = cache.get("test query")

        assert result is not None

    def test_cache_strips_whitespace(self):
        """Verify cache normalizes whitespace."""
        cache = EmbeddingCache()

        cache.put("  test query  ", [0.1], {})
        result = cache.get("test query")

        assert result is not None

    def test_lru_eviction(self):
        """Verify LRU eviction when max size is reached."""
        cache = EmbeddingCache(max_size=3)

        cache.put("query1", [0.1], {})
        cache.put("query2", [0.2], {})
        cache.put("query3", [0.3], {})

        # Access query1 to make it recently used
        cache.get("query1")

        # Add new entry, should evict query2 (least recently used)
        cache.put("query4", [0.4], {})

        assert cache.get("query1") is not None
        assert cache.get("query2") is None
        assert cache.get("query3") is not None
        assert cache.get("query4") is not None

    def test_update_existing_entry(self):
        """Verify updating existing entry works."""
        cache = EmbeddingCache()

        cache.put("query", [0.1], {1: 0.5})
        cache.put("query", [0.2], {2: 0.6})

        result = cache.get("query")

        assert result is not None
        assert result.dense == [0.2]
        assert result.sparse == {2: 0.6}

    def test_clear_removes_all_entries(self):
        """Verify clear removes all cached entries."""
        cache = EmbeddingCache()

        cache.put("query1", [0.1], {})
        cache.put("query2", [0.2], {})
        cache.clear()

        assert cache.size == 0
        assert cache.get("query1") is None


class TestEmbeddingCacheStats:
    """Test suite for cache statistics."""

    def test_hit_rate_calculation(self):
        """Verify hit rate is calculated correctly."""
        cache = EmbeddingCache()

        cache.put("query", [0.1], {})

        # 2 hits
        cache.get("query")
        cache.get("query")
        # 1 miss
        cache.get("other")

        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_stats_returns_complete_info(self):
        """Verify stats returns all expected fields."""
        cache = EmbeddingCache(max_size=100)

        cache.put("query", [0.1], {})
        cache.get("query")
        cache.get("miss")

        stats = cache.stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hit_rate"] == 0.5

    def test_initial_hit_rate_is_zero(self):
        """Verify hit rate is 0 when cache is empty."""
        cache = EmbeddingCache()

        assert cache.hit_rate == 0.0
