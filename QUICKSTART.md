# Quick Start Guide

This guide covers how to set up and run the Telegram RAG Bot locally and via Docker.

## Prerequisites

### Required

| Tool | Version | Purpose |
|------|---------|---------|
| Docker | 24+ | Container runtime |
| Docker Compose | 2.x | Multi-container orchestration |
| Telegram Bot Token | - | From [@BotFather](https://t.me/BotFather) |
| OpenAI API Key | - | For GPT-4o access |

### Optional (for local development)

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime |
| uv | Latest | Fast Python package manager |
| NVIDIA GPU | CUDA 11.8+ | Accelerated ML inference |

## Environment Setup

### 1. Clone and Configure

```bash
# Clone the repository
git clone <repository-url>
cd telegram-rag-bot

# Copy environment template
cp .env.example .env
```

### 2. Edit `.env`

```env
# Required: Your Telegram bot token
TELEGRAM_TOKEN=8130984091340912390423:AA230974nb2k345bkj2gb4j5kb2k34

# Required: Your OpenAI API key
OPENAI_API_KEY=sk-...

# Optional: Adjust model
DEFAULT_MODEL=gpt-4o
```

## Deployment Options

### Option A: Docker Compose (Recommended)

This is the simplest way to run the complete system.

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d

# View logs
docker-compose logs -f telegram-bot
```

**Services started:**
| Service | Port | Description |
|---------|------|-------------|
| qdrant | 6333 | Vector database |
| ml-api | 8001 | Embedding + Reranking (GPU) |
| telegram-bot | - | Your Telegram bot |

**Note for non-GPU systems:**
Edit `docker-compose.yml` and remove the `deploy.resources.reservations.devices` section from the `ml-api` service. The service will fall back to CPU (slower but functional).

### Option B: Local Development

For development with hot-reload and debugging:

```bash
# Start infrastructure only
docker-compose up qdrant ml-api -d

# Set up Python environment
cd telegram-bot
uv sync --dev

# Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux/Mac)
# source .venv/bin/activate

# Run tests
pytest tests/ -v

# Run the bot
python -m app.main
```

## Verifying the Setup

### 1. Check services are running

```bash
# Qdrant health
curl http://localhost:6333/healthz

# ML-API health
curl http://localhost:8001/health
```

### 2. Test the bot

1. Open Telegram and search for your bot (e.g., `@DocuRAGV1Bot`)
2. Send `/start` to begin
3. Send `/help` to see available commands
4. Upload a document (PDF, TXT, MD, DOCX)
5. Ask a question with `/ask What is...`

## Troubleshooting

### Bot not responding

**Symptom:** Bot shows "online" but doesn't respond to commands.

**Solutions:**
1. Check that `TELEGRAM_TOKEN` is correctly set in `.env`
2. Verify docker-compose logs: `docker-compose logs telegram-bot`
3. Ensure the bot was started fresh: `docker-compose restart telegram-bot`

### Connection to ML-API failed

**Symptom:** Error messages about embedding or reranking failures.

**Solutions:**
1. Wait for ML-API to fully initialize (can take 2-3 minutes on first start)
2. Check ML-API health: `curl http://localhost:8001/health`
3. Verify GPU is detected: `docker-compose logs ml-api | grep -i cuda`

### Out of memory (GPU)

**Symptom:** ML-API crashes with CUDA out of memory errors.

**Solutions:**
1. Reduce batch sizes in `docker-compose.yml`:
   ```yaml
   environment:
     - EMBED_MAX_BATCH_SIZE=16
     - RERANK_MAX_BATCH_SIZE=8
   ```
2. Use a GPU with more VRAM
3. Fall back to CPU (remove GPU reservation)

### Document upload fails

**Symptom:** Bot says "Error processing document".

**Solutions:**
1. Check file size (max 10MB)
2. Verify file extension is supported (.pdf, .txt, .md, .docx)
3. Check if the document contains extractable text
4. Review logs: `docker-compose logs telegram-bot`

### OpenAI API errors

**Symptom:** "OpenAI API error" in responses.

**Solutions:**
1. Verify `OPENAI_API_KEY` is valid
2. Check your OpenAI account has credits
3. Ensure you have access to the specified model (gpt-4o)

## Useful Commands

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v

# Rebuild a specific service
docker-compose build telegram-bot

# View real-time logs
docker-compose logs -f

# Execute command in container
docker-compose exec telegram-bot python -c "from app.config import settings; print(settings)"
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design
- Explore the sample documents in `telegram-bot/sample_docs/`
- Add your own documents to create a custom knowledge base
- Integrate additional data sources via the Qdrant API
