# Quick Start Guide

This guide covers setting up and running the Telegram RAG Chatbot locally and via Docker.

## Prerequisites

### Required

| Tool | Version | Purpose |
|------|---------|---------|
| Docker | 24+ | Container runtime |
| Docker Compose | 2.x | Multi-container orchestration |
| Telegram Bot Token | - | From [@BotFather](https://t.me/BotFather) |
| OpenAI API Key | - | For GPT-4o access |

### Optional (Local Development)

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime |
| uv | Latest | Fast Python package manager |
| NVIDIA GPU | CUDA 12.1+ | Accelerated ML inference |

## Configuration

### 1. Clone the Repository

```bash
git clone <repository-url>
cd telegram-rag-chatbot
```

### 2. Create Environment File

```bash
cp .env.example .env
```

### 3. Configure Credentials

Edit `.env` with your credentials:

```env
# Required
TELEGRAM_TOKEN=your-telegram-bot-token
OPENAI_API_KEY=sk-your-openai-api-key

# Optional (defaults shown)
DEFAULT_MODEL=gpt-4o
EMBEDDING_CACHE_SIZE=1000
MAX_HISTORY_TURNS=3
```

## Deployment Options

### Option A: Docker Compose (Recommended)

The simplest way to run the complete system:

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d

# View logs
docker-compose logs -f telegram-bot
```

**Services Started:**

| Service | Port | Description |
|---------|------|-------------|
| qdrant | 6333 | Vector database |
| ml-api | 8001 | Embeddings + Reranking (GPU) |
| telegram-bot | - | Telegram bot instance |

**Non-GPU Systems:**

Edit `docker-compose.yml` and remove the GPU reservation section from `ml-api`:

```yaml
# Remove this block from ml-api service:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

The service falls back to CPU (slower but functional).

### Option B: Local Development

For development with hot-reload and debugging:

```bash
# Start infrastructure services only
docker-compose up qdrant ml-api -d

# Navigate to bot directory
cd telegram-bot

# Create virtual environment and install dependencies
uv sync --dev

# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

# Run tests
pytest tests/ -v

# Start the bot
python -m app.main
```

## Verification

### 1. Check Service Health

```bash
# Qdrant health
curl http://localhost:6333/healthz

# ML-API health
curl http://localhost:8001/health
```

### 2. Test the Bot

1. Open Telegram and search for your bot
2. Send `/start` to begin
3. Send `/help` to see available commands
4. Upload a document (PDF, TXT, MD, DOCX)
5. Ask a question with `/ask <your question>`
6. Summarize a document with `/summarize <filename>`

## Troubleshooting

### Bot Not Responding

**Symptom:** Bot shows "online" but doesn't respond to commands.

**Solutions:**
1. Verify `TELEGRAM_TOKEN` is correctly set in `.env`
2. Check logs: `docker-compose logs telegram-bot`
3. Restart the bot: `docker-compose restart telegram-bot`

### ML-API Connection Failed

**Symptom:** Embedding or reranking errors.

**Solutions:**
1. Wait for ML-API initialization (2-3 minutes on first start for model downloads)
2. Check health: `curl http://localhost:8001/health`
3. Verify GPU detection: `docker-compose logs ml-api | grep -i cuda`

### GPU Out of Memory

**Symptom:** ML-API crashes with CUDA OOM errors.

**Solutions:**
1. Reduce batch sizes in `docker-compose.yml`:
   ```yaml
   environment:
     - EMBED_MAX_BATCH_SIZE=16
     - RERANK_MAX_BATCH_SIZE=8
   ```
2. Use a GPU with more VRAM
3. Fall back to CPU (remove GPU reservation)

### Document Processing Fails

**Symptom:** "Error processing document" message.

**Solutions:**
1. Check file size (max 10MB)
2. Verify supported extension (.pdf, .txt, .md, .docx, .epub)
3. Ensure document contains extractable text
4. Review logs: `docker-compose logs telegram-bot`

### OpenAI API Errors

**Symptom:** "OpenAI API error" in responses.

**Solutions:**
1. Verify `OPENAI_API_KEY` is valid
2. Check OpenAI account has credits
3. Confirm access to the specified model (gpt-4o)

## Useful Commands

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v

# Rebuild specific service
docker-compose build telegram-bot

# Real-time logs
docker-compose logs -f

# Execute command in container
docker-compose exec telegram-bot python -c "from app.config import settings; print(settings)"
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design
- Explore sample documents in `telegram-bot/sample_docs/`
- Upload your own documents to create a custom knowledge base
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment options
