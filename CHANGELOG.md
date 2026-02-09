# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-02-09

### Added
- Comprehensive documentation overhaul following Diataxis principles
- Mermaid diagrams for architecture visualization
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality enforcement
- `DEPLOYMENT.md` with production deployment guides
- `ROADMAP.md` with project milestones
- `CONTRIBUTING.md` with contribution guidelines

### Changed
- Improved docstrings across codebase
- Standardized code formatting with Ruff

## [0.1.1] - 2026-01-20

### Added
- **`/summarize <filename>` command** - Generate concise summaries of indexed documents
  - Searches in both `sample_docs/` and `uploads/` directories
  - Streams summary response in real-time
  - Truncates large documents to first 10,000 characters

## [0.1.0] - 2026-01-19

### Added

- **Telegram Bot Integration**
  - `/start` - Welcome message with onboarding
  - `/help` - Comprehensive usage instructions
  - `/ask <question>` - RAG-powered question answering
  - `/files` - List all indexed documents
  - `/stats` - System statistics (cache, history)
  - `/clear` - Clear conversation history
  - Document upload support (PDF, TXT, MD, DOCX, EPUB)

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
  - Health checks for all services

- **ML-API Service**
  - GPU-accelerated BGE-M3 embeddings (ONNX INT8)
  - GPU-accelerated BGE-Reranker (FP16)
  - Async request batching for throughput
  - FastAPI with health endpoints

### Technical Stack

- Python 3.10+
- python-telegram-bot 22.x
- Qdrant (vector database)
- BGE-M3 + BGE-Reranker (local ML models)
- OpenAI GPT-4o (LLM)
- FastAPI (ML-API service)
- uv (package management)

[0.1.2]: https://github.com/user/telegram-rag-chatbot/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/user/telegram-rag-chatbot/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/user/telegram-rag-chatbot/releases/tag/v0.1.0
