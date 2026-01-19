# Telegram RAG Bot

A lightweight Telegram bot with Retrieval-Augmented Generation (RAG) capabilities for document Q&A.

## Overview

This project implements a Telegram bot that can:
- Answer questions based on uploaded documents
- Use local ML infrastructure for cost-effective embeddings and reranking
- Leverage OpenAI GPT-4o for intelligent response generation
- Maintain conversation context across multiple turns

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-format Document Support** | PDF, TXT, MD, DOCX files |
| **Parent-Child Chunking** | Intelligent hierarchical document splitting |
| **Hybrid Search** | Dense + sparse vectors with RRF fusion |
| **Semantic Reranking** | BGE-Reranker for precision |
| **Query Caching** | LRU cache for repeated queries |
| **Conversation Memory** | Last 3 turns per user |
| **Source Citations** | Every answer includes references |

## Project Structure

```
.
├── telegram-bot/           # Main Telegram bot application
│   ├── app/
│   │   ├── handlers/       # Telegram command handlers
│   │   ├── rag/            # RAG pipeline (orchestrator, cache)
│   │   ├── chunking/       # Parent-child document chunking
│   │   ├── database/       # Qdrant vector database client
│   │   ├── models/         # LLM provider (OpenAI)
│   │   ├── services/       # ML API client
│   │   └── utils/          # Conversation history manager
│   ├── sample_docs/        # Pre-loaded sample documents
│   ├── tests/              # Comprehensive test suite
│   └── Dockerfile
├── ml-api/                 # Embedding + Reranking service (GPU)
│   ├── ml_api.py           # FastAPI service
│   └── Dockerfile
├── docker-compose.yml      # Full stack orchestration
└── docs/
    ├── QUICKSTART.md       # Getting started guide
    └── ARCHITECTURE.md     # Technical deep-dive
```

## Quick Start

### Prerequisites

- Docker and Docker Compose (for containerized deployment)
- Python 3.10+ and [uv](https://github.com/astral-sh/uv) (for local development)
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- OpenAI API Key

### Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your credentials:
   ```env
   TELEGRAM_TOKEN=your-telegram-bot-token
   OPENAI_API_KEY=your-openai-api-key
   ```

### Run with Docker Compose

```bash
docker-compose up --build
```

This starts:
- **Qdrant** (vector database) on port 6333
- **ML-API** (embeddings/reranking) on port 8001
- **Telegram Bot** (your bot instance)

### Run Locally (Development)

```bash
cd telegram-bot
uv sync --dev
.venv/Scripts/Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac

# Start dependencies
docker-compose up qdrant ml-api -d

# Run tests
pytest tests/ -v

# Run bot
python -m app.main
```

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | Usage instructions |
| `/ask <question>` | Ask a question about documents |
| `/files` | List indexed documents |
| `/stats` | System statistics |
| `/clear` | Clear conversation history |

## Technology Stack

- **Bot Framework**: [python-telegram-bot](https://python-telegram-bot.org/) 22.x
- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **Embeddings**: [BGE-M3](https://huggingface.co/BAAI/bge-m3) (1024-dim dense + sparse)
- **Reranking**: [BGE-Reranker-Base](https://huggingface.co/BAAI/bge-reranker-base)
- **LLM**: OpenAI GPT-4o
- **Package Manager**: [uv](https://github.com/astral-sh/uv)

## Documentation

- [**QUICKSTART.md**](QUICKSTART.md) - Detailed setup instructions
- [**ARCHITECTURE.md**](ARCHITECTURE.md) - System design and data flow

## Development

### Running Tests

```bash
cd telegram-bot
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Code Quality

```bash
# Format code
black app/

# Lint
ruff check app/

# Type check
mypy app/
```
