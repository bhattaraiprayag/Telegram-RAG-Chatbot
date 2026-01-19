"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from app.config import Settings


class TestSettings:
    """Test suite for Settings configuration."""

    def test_settings_loads_from_environment(self):
        """Verify settings loads values from environment variables."""
        settings = Settings()

        assert settings.telegram_token == "test-token-12345"
        assert settings.openai_api_key == "test-openai-key"
        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.ml_api_url == "http://localhost:8001"

    def test_settings_respects_cache_size(self):
        """Verify embedding cache size is loaded correctly."""
        settings = Settings()

        assert settings.embedding_cache_size == 500

    def test_settings_respects_history_turns(self):
        """Verify max history turns is loaded correctly."""
        settings = Settings()

        assert settings.max_history_turns == 5

    def test_settings_has_default_values(self):
        """Verify default values are set for optional settings."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            assert settings.default_provider == "openai"
            assert settings.default_model == "gpt-4o"
            assert settings.openai_base_url == "https://api.openai.com/v1"
            assert settings.http_timeout == 120.0

    def test_settings_provider_must_be_openai(self):
        """Verify provider is restricted to openai."""
        settings = Settings()

        assert settings.default_provider == "openai"

    def test_settings_extra_fields_ignored(self):
        """Verify extra environment variables don't cause errors."""
        with patch.dict(os.environ, {"EXTRA_FIELD": "should_be_ignored"}):
            settings = Settings()
            assert not hasattr(settings, "extra_field")
