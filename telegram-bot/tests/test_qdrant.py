"""Tests for Qdrant database client."""

from unittest.mock import MagicMock, patch

import pytest

from app.database.qdrant_client import (
    QdrantDB,
    CHUNKS_COLLECTION,
    PARENTS_COLLECTION,
    DENSE_VECTOR_SIZE,
)


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    with patch("app.database.qdrant_client.QdrantClient") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_instance.get_collections.return_value = mock_collections

        yield mock_instance


class TestQdrantDBInit:
    """Test suite for QdrantDB initialization."""

    def test_creates_collections_on_init(self, mock_qdrant_client):
        """Verify collections are created if they don't exist."""
        db = QdrantDB(url="http://localhost:6333")

        assert mock_qdrant_client.create_collection.call_count == 2

    def test_skips_creation_if_collections_exist(self, mock_qdrant_client):
        """Verify collections are not recreated if they exist."""
        mock_collection1 = MagicMock()
        mock_collection1.name = CHUNKS_COLLECTION
        mock_collection2 = MagicMock()
        mock_collection2.name = PARENTS_COLLECTION

        mock_qdrant_client.get_collections.return_value.collections = [
            mock_collection1,
            mock_collection2,
        ]

        db = QdrantDB(url="http://localhost:6333")

        assert mock_qdrant_client.create_collection.call_count == 0


class TestQdrantDBFileOperations:
    """Test suite for file-related operations."""

    def test_file_exists_returns_true_when_found(self, mock_qdrant_client):
        """Verify file_exists returns True when file is found."""
        mock_qdrant_client.scroll.return_value = ([MagicMock()], None)

        db = QdrantDB(url="http://localhost:6333")
        result = db.file_exists("test_hash_123")

        assert result is True
        mock_qdrant_client.scroll.assert_called_once()

    def test_file_exists_returns_false_when_not_found(self, mock_qdrant_client):
        """Verify file_exists returns False when file is not found."""
        mock_qdrant_client.scroll.return_value = ([], None)

        db = QdrantDB(url="http://localhost:6333")
        result = db.file_exists("nonexistent_hash")

        assert result is False

    def test_get_all_files_returns_file_list(self, mock_qdrant_client):
        """Verify get_all_files returns list of files."""
        mock_point = MagicMock()
        mock_point.payload = {"file_hash": "hash1", "file_name": "test.md"}
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        db = QdrantDB(url="http://localhost:6333")
        files = db.get_all_files()

        assert len(files) == 1
        assert files[0]["file_hash"] == "hash1"
        assert files[0]["file_name"] == "test.md"

    def test_delete_file_deletes_from_both_collections(self, mock_qdrant_client):
        """Verify delete_file removes from both chunks and parents."""
        db = QdrantDB(url="http://localhost:6333")
        db.delete_file("test_hash")

        assert mock_qdrant_client.delete.call_count == 2


class TestQdrantDBStoreOperations:
    """Test suite for store operations."""

    def test_store_chunks_calls_upsert(self, mock_qdrant_client):
        """Verify store_chunks calls upsert with correct data."""
        chunks = [
            {
                "id": "chunk-id-1",
                "content": "test content",
                "parent_id": "parent-id-1",
                "file_hash": "hash123",
                "file_name": "test.md",
                "chunk_index": 0,
                "header_path": "Test > Section",
            }
        ]
        dense_vectors = [[0.1] * DENSE_VECTOR_SIZE]
        sparse_vectors = [{100: 0.5, 200: 0.3}]

        db = QdrantDB(url="http://localhost:6333")
        db.store_chunks(chunks, dense_vectors, sparse_vectors)

        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == CHUNKS_COLLECTION

    def test_store_parents_calls_upsert(self, mock_qdrant_client):
        """Verify store_parents calls upsert with correct data."""
        parents = [
            {
                "id": "parent-id-1",
                "content": "parent content",
                "file_hash": "hash123",
                "file_name": "test.md",
                "header_path": "Test",
                "child_ids": ["child-1", "child-2"],
            }
        ]

        db = QdrantDB(url="http://localhost:6333")
        db.store_parents(parents)

        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == PARENTS_COLLECTION


class TestQdrantDBSearchOperations:
    """Test suite for search operations."""

    def test_hybrid_search_returns_results(self, mock_qdrant_client):
        """Verify hybrid_search returns formatted results."""
        mock_point = MagicMock()
        mock_point.id = "chunk-1"
        mock_point.score = 0.95
        mock_point.payload = {
            "content": "test content",
            "parent_id": "parent-1",
            "file_name": "test.md",
            "header_path": "Test",
        }

        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_qdrant_client.query_points.return_value = mock_result

        db = QdrantDB(url="http://localhost:6333")
        results = db.hybrid_search(
            query_dense=[0.1] * DENSE_VECTOR_SIZE,
            query_sparse={100: 0.5},
        )

        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "test content"

    def test_hybrid_search_with_file_filter(self, mock_qdrant_client):
        """Verify hybrid_search applies file filter."""
        mock_result = MagicMock()
        mock_result.points = []
        mock_qdrant_client.query_points.return_value = mock_result

        db = QdrantDB(url="http://localhost:6333")
        db.hybrid_search(
            query_dense=[0.1] * DENSE_VECTOR_SIZE,
            query_sparse={100: 0.5},
            file_hashes=["hash1", "hash2"],
        )

        mock_qdrant_client.query_points.assert_called_once()

    def test_get_parents_returns_parent_data(self, mock_qdrant_client):
        """Verify get_parents returns parent chunks."""
        mock_point = MagicMock()
        mock_point.id = "parent-1"
        mock_point.payload = {
            "content": "parent content",
            "file_name": "test.md",
            "header_path": "Test",
        }
        mock_qdrant_client.retrieve.return_value = [mock_point]

        db = QdrantDB(url="http://localhost:6333")
        parents = db.get_parents(["parent-1"])

        assert len(parents) == 1
        assert parents[0]["id"] == "parent-1"
        assert parents[0]["content"] == "parent content"
