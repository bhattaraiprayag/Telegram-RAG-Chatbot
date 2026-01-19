"""Tests for RAG orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.orchestrator import RAGOrchestrator, RetrievedContext
from app.rag.cache import EmbeddingCache


@pytest.fixture
def mock_db():
    """Create mock Qdrant database."""
    db = MagicMock()
    db.hybrid_search.return_value = [
        {
            "id": "chunk-1",
            "score": 0.9,
            "content": "Test content 1",
            "parent_id": "parent-1",
            "file_name": "test.md",
            "header_path": "Test > Section",
        },
        {
            "id": "chunk-2",
            "score": 0.8,
            "content": "Test content 2",
            "parent_id": "parent-2",
            "file_name": "test.md",
            "header_path": "Test > Another",
        },
    ]
    db.get_parents.return_value = [
        {
            "id": "parent-1",
            "content": "Full parent content 1",
            "file_name": "test.md",
            "header_path": "Test > Section",
        },
        {
            "id": "parent-2",
            "content": "Full parent content 2",
            "file_name": "test.md",
            "header_path": "Test > Another",
        },
    ]
    return db


@pytest.fixture
def mock_ml_client():
    """Create mock ML API client."""
    client = MagicMock()
    client.embed_single = AsyncMock(
        return_value={
            "dense": [0.1] * 1024,
            "sparse": {100: 0.5, 200: 0.3},
        }
    )
    client.rerank = AsyncMock(
        return_value=[
            (0, 0.95),
            (1, 0.85),
        ]
    )
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = MagicMock()

    async def mock_generate(*args, **kwargs):
        yield "This is "
        yield "a test "
        yield "response."

    provider.generate_streaming = mock_generate
    return provider


@pytest.fixture
def orchestrator(mock_db, mock_ml_client, mock_llm_provider):
    """Create RAG orchestrator with mocked dependencies."""
    return RAGOrchestrator(
        db=mock_db,
        ml_client=mock_ml_client,
        llm_provider=mock_llm_provider,
        embedding_cache=EmbeddingCache(max_size=100),
    )


class TestRAGOrchestratorQuery:
    """Test suite for RAG query operations."""

    @pytest.mark.asyncio
    async def test_query_returns_response(self, orchestrator):
        """Verify query returns streamed response."""
        tokens = []
        async for token in orchestrator.query("What is the policy?"):
            tokens.append(token)

        response = "".join(tokens)
        assert "test response" in response
        assert "Sources" in response

    @pytest.mark.asyncio
    async def test_query_includes_sources(self, orchestrator):
        """Verify query includes source citations."""
        tokens = []
        async for token in orchestrator.query("Test query"):
            tokens.append(token)

        response = "".join(tokens)
        assert "test.md" in response

    @pytest.mark.asyncio
    async def test_query_handles_no_results(self, orchestrator, mock_db):
        """Verify query handles empty search results."""
        mock_db.hybrid_search.return_value = []

        tokens = []
        async for token in orchestrator.query("Unknown query"):
            tokens.append(token)

        response = "".join(tokens)
        assert "couldn't find" in response.lower()


class TestRAGOrchestratorCaching:
    """Test suite for embedding cache integration."""

    @pytest.mark.asyncio
    async def test_uses_cached_embedding(self, orchestrator, mock_ml_client):
        """Verify cached embeddings are reused."""
        # First query
        async for _ in orchestrator.query("test query"):
            pass

        # Second query with same text
        async for _ in orchestrator.query("test query"):
            pass

        # embed_single should only be called once
        assert mock_ml_client.embed_single.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_stats_available(self, orchestrator):
        """Verify cache stats are accessible."""
        stats = orchestrator.get_cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats


class TestRAGOrchestratorWithHistory:
    """Test suite for chat history integration."""

    @pytest.mark.asyncio
    async def test_query_accepts_chat_history(self, orchestrator):
        """Verify query accepts chat history parameter."""
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        tokens = []
        async for token in orchestrator.query("Follow up", chat_history=history):
            tokens.append(token)

        assert len(tokens) > 0


class TestRAGOrchestratorLifecycle:
    """Test suite for orchestrator lifecycle."""

    @pytest.mark.asyncio
    async def test_close_closes_clients(self, orchestrator, mock_ml_client):
        """Verify close properly closes ML client."""
        await orchestrator.close()

        mock_ml_client.close.assert_called_once()


class TestRetrievedContext:
    """Test suite for RetrievedContext dataclass."""

    def test_context_creation(self):
        """Verify RetrievedContext can be created."""
        ctx = RetrievedContext(
            parent_id="p1",
            parent_content="content",
            file_name="test.md",
            header_path="Test",
            relevance_score=0.95,
        )

        assert ctx.parent_id == "p1"
        assert ctx.relevance_score == 0.95
