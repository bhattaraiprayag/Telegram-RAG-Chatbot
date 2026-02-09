"""Application configuration using pydantic-settings."""

import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Suppress HuggingFace symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Telegram Configuration
    telegram_token: str = ""

    # Model Provider Configuration
    default_provider: Literal["openai"] = "openai"
    default_model: str = "gpt-4o"

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Service URLs
    qdrant_url: str = "http://localhost:6333"
    ml_api_url: str = "http://localhost:8001"

    # Caching Configuration
    embedding_cache_size: int = 1000

    # Conversation History
    max_history_turns: int = 3

    # HuggingFace
    hf_home: str = "./models_cache"

    # HTTP Client Configuration
    http_timeout: float = 120.0


settings = Settings()
