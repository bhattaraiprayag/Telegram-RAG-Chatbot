"""LLM model provider for OpenAI."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, AsyncIterator

from openai import AsyncOpenAI

from ..config import settings


class ModelProviderError(Exception):
    """Exception raised for model provider errors."""

    pass


class ModelProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_streaming(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Yields:
            Token strings from the LLM response
        """
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider implementation."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model identifier (e.g., "gpt-4o")
            base_url: API base URL
        """
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate_streaming(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenAI."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:  # type: ignore
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            error_msg = f"OpenAI API error: {type(e).__name__}"
            if hasattr(e, "message"):
                error_msg += f" - {e.message}"
            elif hasattr(e, "body"):
                error_msg += f" - {e.body}"
            else:
                error_msg += f" - {e}"
            raise RuntimeError(error_msg) from e

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate non-streaming response from OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Complete response string
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            # Response is ChatCompletion when stream=False
            assert hasattr(response, "choices"), "Unexpected response type"
            return response.choices[0].message.content or ""
        except Exception as e:
            error_msg = f"OpenAI API error: {type(e).__name__} - {e}"
            raise RuntimeError(error_msg) from e


class ModelFactory:
    """Factory for creating LLM provider instances."""

    def __init__(self) -> None:
        """Initialize model factory with settings."""
        self.model_name = settings.default_model

    def get_provider(self) -> OpenAIProvider:
        """
        Get the configured OpenAI provider instance.

        Returns:
            OpenAIProvider instance

        Raises:
            ModelProviderError: If API key is not configured
        """
        if not settings.openai_api_key:
            raise ModelProviderError(
                "OpenAI API key not configured. Set OPENAI_API_KEY in .env"
            )

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=self.model_name,
            base_url=settings.openai_base_url,
        )
