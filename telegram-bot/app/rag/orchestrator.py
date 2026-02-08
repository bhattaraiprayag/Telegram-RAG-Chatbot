"""RAG orchestrator coordinating retrieval, reranking, and generation."""

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional
from pathlib import Path

from ..config import settings
from ..database.qdrant_client import QdrantDB
from ..models.model_factory import ModelFactory, OpenAIProvider
from ..services.ml_api_client import MLAPIClient
from .cache import EmbeddingCache


HYBRID_SEARCH_LIMIT = 30
RERANK_TOP_K = 5


@dataclass
class RetrievedContext:
    """Retrieved context with metadata."""

    parent_id: str
    parent_content: str
    file_name: str
    header_path: str
    relevance_score: float


@dataclass
class RAGResponse:
    """Complete RAG response with answer and sources."""

    answer: str
    sources: list[RetrievedContext]


class RAGOrchestrator:
    """Main RAG pipeline orchestrator with caching support."""

    def __init__(
        self,
        db: Optional[QdrantDB] = None,
        ml_client: Optional[MLAPIClient] = None,
        llm_provider: Optional[OpenAIProvider] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
    ) -> None:
        """
        Initialize RAG orchestrator.

        Args:
            db: Qdrant database client
            ml_client: ML API client for embeddings/reranking
            llm_provider: LLM provider for generation
            embedding_cache: Cache for query embeddings
        """
        self.db = db or QdrantDB()
        self.ml_client = ml_client or MLAPIClient()
        self.embedding_cache = embedding_cache or EmbeddingCache(
            max_size=settings.embedding_cache_size
        )

        if llm_provider is None:
            factory = ModelFactory()
            self.llm_provider = factory.get_provider()
        else:
            self.llm_provider = llm_provider

    async def query(
        self,
        user_query: str,
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Execute RAG query pipeline.

        Args:
            user_query: User's question
            chat_history: Optional conversation history

        Yields:
            Response tokens from LLM
        """
        # Step 1: Get query embedding (with caching)
        query_vectors = await self._get_query_embedding(user_query)

        # Step 2: Hybrid search
        search_results = self.db.hybrid_search(
            query_dense=query_vectors["dense"],
            query_sparse=query_vectors["sparse"],
            limit=HYBRID_SEARCH_LIMIT,
        )

        if not search_results:
            yield "I couldn't find any relevant information in the documents."
            return

        # Step 3: Rerank
        documents = [r["content"] for r in search_results]
        reranked = await self.ml_client.rerank(
            query=user_query,
            documents=documents,
            top_k=RERANK_TOP_K * 2,
        )

        # Step 4: Get parent contexts (deduplicated)
        parent_ids = []
        seen_parents: set[str] = set()
        for idx, score in reranked[:RERANK_TOP_K]:
            parent_id = search_results[idx]["parent_id"]
            if parent_id not in seen_parents:
                parent_ids.append(parent_id)
                seen_parents.add(parent_id)

        parents = self.db.get_parents(parent_ids)

        # Build context objects
        contexts = []
        for parent in parents:
            best_score = 0.0
            for idx, score in reranked:
                if search_results[idx]["parent_id"] == parent["id"]:
                    best_score = max(best_score, score)
                    break

            contexts.append(
                RetrievedContext(
                    parent_id=parent["id"],
                    parent_content=parent["content"],
                    file_name=parent["file_name"],
                    header_path=parent["header_path"],
                    relevance_score=best_score,
                )
            )

        # Step 5: Generate response
        async for token in self._generate(user_query, contexts, chat_history):
            yield token

        # Append source references
        yield "\n\n---\n**Sources:**\n"
        for i, ctx in enumerate(contexts, 1):
            yield f"- [{i}] {ctx.file_name} > {ctx.header_path}\n"

    async def query_with_sources(
        self,
        user_query: str,
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> RAGResponse:
        """
        Execute RAG query and return complete response with sources.

        Args:
            user_query: User's question
            chat_history: Optional conversation history

        Returns:
            RAGResponse with answer and sources
        """
        answer_parts = []
        sources: list[RetrievedContext] = []

        async for token in self.query(user_query, chat_history):
            if token.startswith("\n\n---\n**Sources"):
                # Parse sources after this marker
                continue
            if token.startswith("- ["):
                # Skip source lines in answer
                continue
            answer_parts.append(token)

        return RAGResponse(
            answer="".join(answer_parts).rstrip(),
            sources=sources,
        )
    
    async def summarize(self, file_name: str) -> AsyncGenerator[str, None]:
        search_paths = [
            Path("sample_docs") / file_name,
            Path("uploads") / file_name
        ]

        content = None
        for path in search_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                break
        
        if not content:
            yield f"Could not find file: {file_name}"
            return
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that SUMMARIZES documents concisely."},
            {"role": "user", "content": f"Please summarize this document: {file_name}.\n\nCONTENT:\n{content[:10000]}"}
        ]

        # Stream from LLM provider
        async for token in self.llm_provider.generate_streaming(
            messages=messages, temperature=0.3, max_tokens=1024
        ):
            yield token
    async def _get_query_embedding(self, query: str) -> dict[str, Any]:
        """
        Get embedding for query, using cache if available.

        Args:
            query: Query text

        Returns:
            Dict with 'dense' and 'sparse' vectors
        """
        # Check cache first
        cached = self.embedding_cache.get(query)
        if cached is not None:
            return {"dense": cached.dense, "sparse": cached.sparse}

        # Get fresh embedding
        result = await self.ml_client.embed_single(query, is_query=True)

        # Cache it
        self.embedding_cache.put(query, result["dense"], result["sparse"])

        return result

    async def _generate(
        self,
        query: str,
        contexts: list[RetrievedContext],
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response using LLM provider.

        Args:
            query: User query
            contexts: Retrieved contexts
            chat_history: Optional chat history

        Yields:
            Response tokens from LLM
        """
        # Build context string
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(
                f"[Source {i}: {ctx.file_name} > {ctx.header_path}]\n"
                f"{ctx.parent_content}\n"
            )
        context_str = "\n---\n".join(context_parts)

        # System prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.

RULES:
1. Answer ONLY using information from the provided context.
2. If the context doesn't contain the answer, say "I don't have information about that in the provided documents."
3. Cite your sources using [Source N] notation.
4. Be concise and direct.
5. Use markdown formatting for readability."""

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            messages.extend(chat_history[-6:])  # Last 3 turns

        user_message = f"""CONTEXT:
{context_str}

QUESTION: {query}

Provide a helpful answer based on the context above."""

        messages.append({"role": "user", "content": user_message})

        # Stream from LLM provider
        async for token in self.llm_provider.generate_streaming(
            messages=messages, temperature=0.3, max_tokens=1024
        ):
            yield token

    async def close(self) -> None:
        """Close all clients."""
        await self.ml_client.close()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return self.embedding_cache.stats()
