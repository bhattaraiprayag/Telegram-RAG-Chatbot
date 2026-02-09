# Roadmap

This document outlines the project milestones, completed phases, and future development plans.

## Current Status: v0.1.x (MVP)

The Telegram RAG Chatbot is currently in its initial release phase with core functionality complete and tested.

## Completed Milestones

### ✅ Phase 1: Core RAG Pipeline (v0.1.0)

| Feature | Status |
|---------|--------|
| Parent-child hierarchical chunking | Complete |
| BGE-M3 embeddings (dense + sparse) | Complete |
| Hybrid search with RRF fusion | Complete |
| BGE-Reranker semantic reranking | Complete |
| OpenAI GPT-4o integration | Complete |
| LRU embedding cache | Complete |

### ✅ Phase 2: Telegram Integration (v0.1.0)

| Feature | Status |
|---------|--------|
| `/start`, `/help` commands | Complete |
| `/ask <question>` RAG queries | Complete |
| `/files` list indexed documents | Complete |
| `/stats` cache and history statistics | Complete |
| `/clear` conversation history | Complete |
| Document upload handling | Complete |
| Multi-format support (PDF, TXT, MD, DOCX, EPUB) | Complete |

### ✅ Phase 3: Document Summarization (v0.1.1)

| Feature | Status |
|---------|--------|
| `/summarize <filename>` command | Complete |
| Streaming summary generation | Complete |

### ✅ Phase 4: Infrastructure (v0.1.0)

| Feature | Status |
|---------|--------|
| Docker Compose orchestration | Complete |
| Multi-stage Dockerfile builds | Complete |
| GPU-accelerated ML-API | Complete |
| Health checks | Complete |
| Comprehensive test suite (80+ tests) | Complete |

### ✅ Phase 5: Repository Hygiene (v0.1.2)

| Feature | Status |
|---------|--------|
| Documentation overhaul | Complete |
| Mermaid diagram integration | Complete |
| CI/CD pipeline (GitHub Actions) | Complete |
| Pre-commit hooks | Complete |

## Planned Features

### 📋 Phase 6: Enhanced User Experience

| Feature | Priority | Description |
|---------|----------|-------------|
| `/delete <file>` command | High | Allow users to remove indexed documents |
| Inline query mode | Medium | Answer questions directly in any chat |
| Multi-language support | Medium | Detect and respond in user's language |
| Progress indicators | Low | Better feedback during long operations |

### 📋 Phase 7: Advanced RAG Features

| Feature | Priority | Description |
|---------|----------|-------------|
| Query rewriting | High | Improve retrieval with reformulated queries |
| Hypothetical document embeddings (HyDE) | Medium | Generate hypothetical answers for better retrieval |
| Multi-document synthesis | Medium | Aggregate information across documents |
| Fact verification | Low | Cross-reference answers with sources |

### 📋 Phase 8: Production Hardening

| Feature | Priority | Description |
|---------|----------|-------------|
| Webhook mode | High | Replace polling for production reliability |
| Redis session storage | High | Persistent conversation history |
| Prometheus metrics | Medium | Observability and monitoring |
| Rate limiting | Medium | Protect against abuse |
| User authentication | Low | Restrict bot access |

### 📋 Phase 9: Scalability

| Feature | Priority | Description |
|---------|----------|-------------|
| Qdrant cluster mode | Medium | High availability for vector database |
| ML-API horizontal scaling | Medium | Load balancer for embedding service |
| CDN for model weights | Low | Faster model downloads globally |

### 📋 Phase 10: Alternative Backends

| Feature | Priority | Description |
|---------|----------|-------------|
| Ollama integration | High | Local LLM option (no OpenAI required) |
| Anthropic Claude | Medium | Alternative to OpenAI |
| Groq API | Low | Ultra-fast inference option |

## Version Timeline

| Version | Target | Focus |
|---------|--------|-------|
| v0.1.2 | Q1 2026 | Repository hygiene, CI/CD |
| v0.2.0 | Q2 2026 | Enhanced UX, `/delete` command |
| v0.3.0 | Q3 2026 | Production hardening |
| v1.0.0 | Q4 2026 | Stable release with full feature set |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose new features or contribute to existing roadmap items.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.
