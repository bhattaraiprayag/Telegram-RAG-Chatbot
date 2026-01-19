# Architecture

This document describes the technical architecture, design decisions, and data flow of the Telegram RAG Bot.

## System Overview

The application implements a modular, containerized Retrieval-Augmented Generation (RAG) system designed for Telegram integration. It follows a microservices pattern with three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Telegram Users                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ (Telegram Bot API)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              telegram-bot (python-telegram-bot)                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Handlers: /start, /help, /ask, /files, /clear, /stats      │ │
│  │ + Document upload handler                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                     │
│  ┌────────────────────────┴─────────────────────────────────┐   │
│  │                    RAG Orchestrator                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │ Embedding   │  │   History   │  │    Chunking     │   │   │
│  │  │   Cache     │  │   Manager   │  │     Engine      │   │   │
│  │  │   (LRU)     │  │ (In-Memory) │  │ (Parent-Child)  │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────┬───────────────────────────────────────────────────┘
              │
    ┌─────────┴─────────┬───────────────────────┐
    │                   │                       │
    ▼                   ▼                       ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐
│   ML-API    │   │   Qdrant    │   │      OpenAI API         │
│  (FastAPI)  │   │ (Vector DB) │   │      (GPT-4o)           │
│             │   │             │   │                         │
│ • BGE-M3    │   │ • Chunks    │   │ • Response Generation   │
│ • Reranker  │   │ • Parents   │   │ • Streaming             │
└─────────────┘   └─────────────┘   └─────────────────────────┘
    GPU                Docker             External API
```

## Components

### 1. Telegram Bot (`telegram-bot/`)

The main application handling user interactions via Telegram.

**Key Modules:**

| Module | Purpose |
|--------|---------|
| `handlers/` | Command handlers (/start, /help, /ask, etc.) |
| `rag/orchestrator.py` | Coordinates the RAG pipeline |
| `rag/cache.py` | LRU cache for query embeddings |
| `chunking/engine.py` | Parent-child hierarchical chunking |
| `database/qdrant_client.py` | Qdrant vector database operations |
| `services/ml_api_client.py` | HTTP client for ML-API |
| `models/model_factory.py` | OpenAI provider abstraction |
| `utils/history.py` | Per-user conversation history |

### 2. ML-API (`ml-api/`)

Dedicated FastAPI microservice for compute-intensive ML inference.

**Endpoints:**
- `POST /embed` - Generate BGE-M3 embeddings (dense + sparse)
- `POST /rerank` - Score documents with BGE-Reranker
- `GET /health` - Health check

**Why Separate?**
- Independent scaling (GPU-heavy service vs lightweight bot)
- Resource isolation (GPU memory management)
- Simplified dependencies (bot doesn't need PyTorch/CUDA)

### 3. Qdrant Vector Database

Stores document embeddings for semantic retrieval.

**Collections:**
- `chunks` - Child chunks with dense + sparse vectors
- `parents` - Parent chunks (context source, no vectors)

## Data Flow

### Document Ingestion

```
User uploads document
        │
        ▼
┌───────────────────┐
│ Download to disk  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Calculate SHA256  │──── Already indexed? ──→ Skip
│     file hash     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Convert to        │  Using MarkItDown library
│ Markdown text     │  (supports PDF, DOCX, etc.)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Parent-Child      │  Split by headers (semantic)
│ Chunking          │  Children: 384 tokens, 64 overlap
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Generate          │  BGE-M3: 1024-dim dense
│ Embeddings        │        + sparse vectors
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Store in Qdrant   │  chunks collection: vectors
│                   │  parents collection: content
└───────────────────┘
```

### RAG Query

```
User: /ask "What is the refund policy?"
        │
        ▼
┌───────────────────┐
│ Check embedding   │──── Cache hit? ──→ Use cached vectors
│     cache (LRU)   │
└─────────┬─────────┘
          │ (cache miss)
          ▼
┌───────────────────┐
│ Embed query       │  ML-API /embed
│ (dense + sparse)  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Hybrid search     │  Qdrant prefetch (dense + sparse)
│ with RRF fusion   │  Reciprocal Rank Fusion
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Rerank results    │  ML-API /rerank
│ (BGE-Reranker)    │  Top-K most relevant
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Fetch parent      │  Full context from parents
│ chunks            │  Deduplicated by parent_id
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Build prompt +    │  System prompt + context + history
│ stream response   │  OpenAI GPT-4o streaming
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Update history    │  Store user question + response
│ manager           │  (last 3 turns per user)
└───────────────────┘
```

## Design Decisions

### Parent-Child Chunking

Instead of flat chunking, we use a two-level hierarchy:

- **Parent Chunks** (~1200 tokens): Full sections with complete context
- **Child Chunks** (~384 tokens): Overlapping retrieval targets

**Why?**
- Children are small enough for precise retrieval
- Parents provide complete context to the LLM
- Reduces hallucination from fragmented context

### Hybrid Search with RRF

We combine dense and sparse vectors:

1. **Dense vectors (BGE-M3)**: Semantic similarity
2. **Sparse vectors (BM25-style)**: Keyword matching

**Reciprocal Rank Fusion (RRF)** combines both rankings:
```
RRF_score = Σ 1 / (k + rank_i)
```

This captures both semantic meaning AND exact term matches.

### Embedding Cache

LRU cache for query embeddings:

- **Key**: MD5 hash of normalized query
- **Value**: Dense + sparse vectors
- **Size**: Configurable (default 1000)

**Why?**
- Repeated queries skip ML-API call
- Significant latency reduction
- Task requirement: "don't re-embed already seen queries"

### In-Memory Conversation History

Per-user message history (last 3 turns):

- Stored in Python dict
- Cleared on bot restart
- Passed to LLM for context

**Production Enhancement:**
Could be persisted to Redis or Qdrant metadata for durability.

## Security Considerations

| Concern | Mitigation |
|---------|------------|
| API Keys | Stored in `.env`, never committed |
| File Uploads | Size limit (10MB), extension whitelist |
| User Data | In-memory only (clears on restart) |
| Network | Internal Docker network for services |

## Scalability

### Current Limitations

- Single bot instance (Telegram polling)
- In-memory cache and history
- Single Qdrant instance

### Production Enhancements

| Component | Enhancement |
|-----------|-------------|
| Bot | Webhook mode + load balancer |
| Cache | Redis cluster |
| History | PostgreSQL or Redis |
| Qdrant | Cluster mode with replication |
| ML-API | Horizontal scaling behind LB |

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_TOKEN` | - | Bot token from @BotFather |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `DEFAULT_MODEL` | `gpt-4o` | LLM model identifier |
| `QDRANT_URL` | `localhost:6333` | Qdrant server URL |
| `ML_API_URL` | `localhost:8001` | ML-API server URL |
| `EMBEDDING_CACHE_SIZE` | `1000` | Max cached embeddings |
| `MAX_HISTORY_TURNS` | `3` | Conversation turns to keep |
