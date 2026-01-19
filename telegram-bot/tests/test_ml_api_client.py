"""Tests for ML API client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.services.ml_api_client import MLAPIClient


@pytest.fixture
def ml_client():
    """Create ML API client for testing."""
    return MLAPIClient(base_url="http://localhost:8001", timeout=30.0)


class TestMLAPIClientEmbed:
    """Test suite for embedding operations."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self, ml_client):
        """Verify embedding a single text works."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "sparse_vecs": [{100: 0.5, 200: 0.3}],
        }

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await ml_client.embed_single("test query", is_query=True)

            assert "dense" in result
            assert "sparse" in result
            assert len(result["dense"]) == 1024

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, ml_client):
        """Verify embedding multiple texts works."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "dense_vecs": [[0.1] * 1024, [0.2] * 1024],
            "sparse_vecs": [{100: 0.5}, {200: 0.3}],
        }

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await ml_client.embed(["text 1", "text 2"])

            assert len(result["dense_vecs"]) == 2
            assert len(result["sparse_vecs"]) == 2


class TestMLAPIClientRerank:
    """Test suite for reranking operations."""

    @pytest.mark.asyncio
    async def test_rerank_returns_sorted_results(self, ml_client):
        """Verify reranking returns (index, score) tuples."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "score": 0.95},
                {"index": 0, "score": 0.85},
                {"index": 1, "score": 0.75},
            ]
        }

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await ml_client.rerank(
                query="what is the policy",
                documents=["doc 1", "doc 2", "doc 3"],
                top_k=3,
            )

            assert len(result) == 3
            assert result[0] == (2, 0.95)
            assert result[1] == (0, 0.85)


class TestMLAPIClientHealth:
    """Test suite for health check operations."""

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_healthy(self, ml_client):
        """Verify health check returns True when service is up."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_response

            result = await ml_client.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self, ml_client):
        """Verify health check returns False on connection error."""
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            result = await ml_client.health_check()

            assert result is False


class TestMLAPIClientLifecycle:
    """Test suite for client lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_closes_client(self, ml_client):
        """Verify close properly closes the HTTP client."""
        # Force client creation
        await ml_client._get_client()
        assert ml_client._client is not None

        # Close
        await ml_client.close()
        assert ml_client._client is None
