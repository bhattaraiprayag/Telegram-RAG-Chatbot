"""ML API client for embeddings and reranking."""

from typing import Any

import httpx

from ..config import settings


class MLAPIClient:
    """Client for the ML API service (embeddings + reranking)."""

    def __init__(
        self, base_url: str | None = None, timeout: float | None = None
    ) -> None:
        """
        Initialize ML API client.

        Args:
            base_url: ML API base URL. Defaults to settings.ml_api_url.
            timeout: Request timeout. Defaults to settings.http_timeout.
        """
        self.base_url = base_url or settings.ml_api_url
        self.timeout = timeout or settings.http_timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(
        self, texts: list[str], is_query: bool = False
    ) -> dict[str, list[Any]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            is_query: Whether these are query texts (affects embedding)

        Returns:
            Dict with 'dense_vecs' and 'sparse_vecs' lists
        """
        client = await self._get_client()

        response = await client.post(
            f"{self.base_url}/embed",
            json={"text": texts, "is_query": is_query},
        )

        response.raise_for_status()
        result: dict[str, list[Any]] = response.json()
        return result

    async def embed_single(self, text: str, is_query: bool = False) -> dict[str, Any]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            is_query: Whether this is a query text

        Returns:
            Dict with 'dense' and 'sparse' vectors
        """
        result = await self.embed([text], is_query=is_query)
        return {
            "dense": result["dense_vecs"][0],
            "sparse": result["sparse_vecs"][0],
        }

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by score descending
        """
        client = await self._get_client()

        response = await client.post(
            f"{self.base_url}/rerank",
            json={
                "query": query,
                "documents": documents,
                "top_k": top_k,
            },
        )

        response.raise_for_status()
        data = response.json()

        return [(r["index"], r["score"]) for r in data["results"]]

    async def health_check(self) -> bool:
        """
        Check if the ML API is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
