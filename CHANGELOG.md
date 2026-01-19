# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-19

### Added

- **Telegram Bot Integration**
  - `/start` - Welcome message with onboarding
  - `/help` - Comprehensive usage instructions
  - `/ask <question>` - RAG-powered question answering
  - `/files` - List all indexed documents
  - `/stats` - System statistics (cache, history)
  - `/clear` - Clear conversation history
  - Document upload support (PDF, TXT, MD, DOCX)

- **RAG Pipeline**
  - Parent-child hierarchical chunking with markdown awareness
  - Hybrid search with dense + sparse vectors (RRF fusion)
  - BGE-M3 embeddings (1024-dim) via ML-API
  - BGE-Reranker for semantic reranking
  - OpenAI GPT-4o for response generation
  - Source citations in all answers

- **Performance Optimizations**
  - LRU embedding cache for query deduplication
  - Per-user conversation history (last 3 turns)
  - Document deduplication via SHA256 hashing

- **Infrastructure**
  - Docker Compose orchestration (Qdrant + ML-API + Bot)
  - uv-based Python dependency management
  - Multi-stage Dockerfile for optimized images
  - Comprehensive test suite (83 tests)

- **Documentation**
  - README.md with project overview
  - QUICKSTART.md with setup instructions
  - ARCHITECTURE.md with technical deep-dive

### Technical Stack

- Python 3.10+
- python-telegram-bot 22.x
- Qdrant (vector database)
- BGE-M3 + BGE-Reranker (local ML models)
- OpenAI GPT-4o (LLM)
- FastAPI (ML-API service)
- uv (package management)

---

## Template for Future Releases

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security patches
```
