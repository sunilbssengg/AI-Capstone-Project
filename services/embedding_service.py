"""
Embedding service — converts text into vector embeddings using Google
Gemini's embedding model, via LangChain's GoogleGenerativeAIEmbeddings
wrapper. Centralizing this in one place means the vector store and any
future re-ranking logic all use a single, consistent embedding function.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from core.config import settings
from core.exceptions import EmbeddingError
from utils.logger import logger


@lru_cache(maxsize=1)
def get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    """Return a cached embedding function instance (one per process)."""
    if not settings.GOOGLE_API_KEY:
        raise EmbeddingError(
            "GOOGLE_API_KEY is not set. Add it to your .env file "
            "(see .env.example)."
        )
    try:
        return GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError("Failed to initialize embedding model.", str(exc)) from exc


def embed_query(query: str) -> list[float]:
    """Embed a single user query for similarity search."""
    try:
        return get_embedding_function().embed_query(query)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Embedding query failed: {exc}")
        raise EmbeddingError("Failed to embed the query.", str(exc)) from exc
