"""Pytest configuration and fixtures."""

import os
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Set up test environment variables."""
    env_vars = {
        "TELEGRAM_TOKEN": "test-token-12345",
        "OPENAI_API_KEY": "test-openai-key",
        "QDRANT_URL": "http://localhost:6333",
        "ML_API_URL": "http://localhost:8001",
        "EMBEDDING_CACHE_SIZE": "500",
        "MAX_HISTORY_TURNS": "5",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for chunking tests."""
    return """# Company FAQ

## General Questions

### What is our company?
We are a technology company focused on AI solutions.
Our mission is to make AI accessible to everyone.

### Where are we located?
Our headquarters is in San Francisco, California.
We also have offices in New York and London.

## Product Information

### What products do we offer?
We offer three main products:
1. AI Assistant - A conversational AI
2. Data Analytics - Business intelligence platform
3. ML Pipeline - End-to-end machine learning

### How do I get started?
Visit our website and sign up for a free trial.
You can upgrade to a paid plan at any time.

## Support

### How can I contact support?
Email us at support@company.com or call 1-800-COMPANY.
Our support team is available 24/7.
"""


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What products do you offer?"


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    return [0.1] * 1024


@pytest.fixture
def sample_sparse_embedding():
    """Sample sparse embedding for testing."""
    return {100: 0.5, 200: 0.3, 300: 0.2}
