"""Tests for the chunking engine."""

import pytest

from app.chunking import ChildChunk, ChunkingEngine, ParentChunk


@pytest.fixture
def chunking_engine():
    """Create a chunking engine instance for testing."""
    return ChunkingEngine(
        parent_max_tokens=200,
        parent_min_tokens=50,
        child_tokens=100,
        child_overlap=20,
    )


class TestChunkingEngine:
    """Test suite for ChunkingEngine."""

    def test_chunk_empty_document_returns_empty(self, chunking_engine):
        """Verify empty content returns empty results."""
        parents, children = chunking_engine.chunk_document("", "hash123", "test.md")

        assert parents == []
        assert children == []

    def test_chunk_whitespace_only_returns_empty(self, chunking_engine):
        """Verify whitespace-only content returns empty results."""
        parents, children = chunking_engine.chunk_document(
            "   \n\n  ", "hash123", "test.md"
        )

        assert parents == []
        assert children == []

    def test_chunk_document_without_headers(self, chunking_engine):
        """Verify content without headers is treated as single section."""
        content = "This is some plain text without any markdown headers."

        parents, children = chunking_engine.chunk_document(
            content, "hash123", "test.md"
        )

        assert len(parents) >= 1
        assert parents[0].header_path == "Document"
        assert parents[0].file_hash == "hash123"
        assert parents[0].file_name == "test.md"

    def test_chunk_document_with_headers(
        self, chunking_engine, sample_markdown_content
    ):
        """Verify markdown with headers is split correctly."""
        parents, children = chunking_engine.chunk_document(
            sample_markdown_content, "hash123", "company_faq.md"
        )

        assert len(parents) > 0
        assert len(children) > 0

        for parent in parents:
            assert parent.file_hash == "hash123"
            assert parent.file_name == "company_faq.md"
            assert parent.header_path != ""
            assert parent.id != ""

    def test_children_reference_parents(self, chunking_engine, sample_markdown_content):
        """Verify children reference their parent chunk."""
        parents, children = chunking_engine.chunk_document(
            sample_markdown_content, "hash123", "test.md"
        )

        parent_ids = {p.id for p in parents}

        for child in children:
            assert child.parent_id in parent_ids
            assert child.file_hash == "hash123"
            assert child.file_name == "test.md"
            assert child.chunk_index >= 0

    def test_parent_chunk_ids_are_deterministic(self, chunking_engine):
        """Verify same content produces same chunk IDs."""
        content = "# Test\n\nSome content here for testing purposes."

        parents1, _ = chunking_engine.chunk_document(content, "hash1", "test.md")
        parents2, _ = chunking_engine.chunk_document(content, "hash1", "test.md")

        assert len(parents1) == len(parents2)
        for p1, p2 in zip(parents1, parents2):
            assert p1.id == p2.id

    def test_different_file_hash_produces_different_ids(self, chunking_engine):
        """Verify different file hash produces different chunk IDs."""
        content = "# Test\n\nSome content."

        parents1, _ = chunking_engine.chunk_document(content, "hash_a", "test.md")
        parents2, _ = chunking_engine.chunk_document(content, "hash_b", "test.md")

        assert parents1[0].id != parents2[0].id

    def test_child_chunks_have_valid_structure(
        self, chunking_engine, sample_markdown_content
    ):
        """Verify child chunks have all required fields."""
        _, children = chunking_engine.chunk_document(
            sample_markdown_content, "hash123", "test.md"
        )

        for child in children:
            assert isinstance(child, ChildChunk)
            assert child.id != ""
            assert child.content != ""
            assert child.parent_id != ""
            assert child.header_path != ""

    def test_parent_chunks_have_valid_structure(
        self, chunking_engine, sample_markdown_content
    ):
        """Verify parent chunks have all required fields."""
        parents, _ = chunking_engine.chunk_document(
            sample_markdown_content, "hash123", "test.md"
        )

        for parent in parents:
            assert isinstance(parent, ParentChunk)
            assert parent.id != ""
            assert parent.content != ""
            assert parent.header_path != ""
            assert isinstance(parent.child_ids, list)

    def test_header_hierarchy_preserved(self, chunking_engine):
        """Verify header hierarchy is captured in header_path."""
        content = """# Main Title

## Section One

Content for section one.

### Subsection A

Content for subsection A.

## Section Two

Content for section two.
"""
        parents, _ = chunking_engine.chunk_document(content, "hash", "test.md")

        header_paths = [p.header_path for p in parents]

        assert any("Main Title" in hp for hp in header_paths)


class TestChunkingEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_unicode_content(self, chunking_engine):
        """Verify unicode characters are handled correctly."""
        content = "# Título\n\nContenido con caracteres especiales: ñ, ü, 中文, 日本語"

        parents, children = chunking_engine.chunk_document(
            content, "hash", "unicode.md"
        )

        assert len(parents) >= 1
        assert "ñ" in parents[0].content or "中文" in parents[0].content

    def test_handles_special_markdown_characters(self, chunking_engine):
        """Verify special markdown characters don't break chunking."""
        content = """# Test **bold** and *italic*

Here's a [link](https://example.com) and `code`.

```python
def hello():
    print("world")
```
"""
        parents, children = chunking_engine.chunk_document(
            content, "hash", "special.md"
        )

        assert len(parents) >= 1
        assert "bold" in parents[0].content or "code" in parents[0].content
