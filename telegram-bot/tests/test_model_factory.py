"""Tests for model factory."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.model_factory import (
    ModelFactory,
    ModelProviderError,
    OpenAIProvider,
)


class TestModelFactory:
    """Test suite for ModelFactory."""

    def test_factory_creates_openai_provider(self):
        """Verify factory creates OpenAI provider with valid config."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from app.config import Settings

            with patch("app.models.model_factory.settings", Settings()):
                factory = ModelFactory()
                provider = factory.get_provider()

                assert isinstance(provider, OpenAIProvider)

    def test_factory_raises_error_without_api_key(self):
        """Verify factory raises error when API key is missing."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            from app.config import Settings

            with patch(
                "app.models.model_factory.settings", Settings(openai_api_key="")
            ):
                factory = ModelFactory()

                with pytest.raises(ModelProviderError) as exc_info:
                    factory.get_provider()

                assert "API key not configured" in str(exc_info.value)


class TestOpenAIProvider:
    """Test suite for OpenAIProvider."""

    def test_provider_initialization(self):
        """Verify provider initializes with correct parameters."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
        )

        assert provider.model == "gpt-4o"
        assert provider.client is not None

    @pytest.mark.asyncio
    async def test_generate_streaming_yields_tokens(self):
        """Verify streaming generation yields tokens."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o",
        )

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"

        async def mock_stream():
            yield mock_chunk

        mock_response = mock_stream()

        with patch.object(
            provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            tokens = []
            async for token in provider.generate_streaming(
                messages=[{"role": "user", "content": "Hi"}]
            ):
                tokens.append(token)

            assert "Hello" in tokens

    @pytest.mark.asyncio
    async def test_generate_returns_complete_response(self):
        """Verify non-streaming generation returns complete response."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"

        with patch.object(
            provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.generate(
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_generate_handles_empty_response(self):
        """Verify generation handles None content gracefully."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch.object(
            provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.generate(
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert result == ""
