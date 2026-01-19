# syntax=docker/dockerfile:1
# Telegram RAG Bot Dockerfile
# Multi-stage build for optimized image size

# ============================================
# Stage 1: Build dependencies
# ============================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Create venv and install dependencies
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies (production only)
# We use pip compile to generate requirements.txt from pyproject.toml
# then pip install to install them into the venv
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt

# ============================================
# Stage 2: Runtime image
# ============================================
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy application code
COPY app/ /app/app/
COPY sample_docs/ /app/sample_docs/

# Create uploads directory
RUN mkdir -p /app/uploads

# Download NLTK data
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p /usr/local/share/nltk_data && \
    python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab

# Run as non-root user
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /usr/local/share/nltk_data
USER appuser

# Health check (just verify Python can import the app)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from app.config import settings; print('healthy')" || exit 1

# Entry point
CMD ["python", "-m", "app.main"]
