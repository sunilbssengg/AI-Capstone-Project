"""
Vector-based knowledge store using Chroma (persisted to `data/chroma_db`,
committed as part of the GitHub repo structure per the project spec).

Responsibilities:
  - Initialize / connect to a persistent Chroma collection.
  - Add document chunks (embedding happens automatically via the
    configured embedding function).
  - Run cosine-similarity search for a query and return top-N chunks.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from langchain.docstore.document import Document
from langchain_chroma import Chroma

from core.config import settings
from core.exceptions import VectorStoreError
from services.embedding_service import get_embedding_function
from utils.logger import logger


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Return a cached Chroma vector store instance backed by disk persistence."""
    try:
        return Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            persist_directory=settings.resolve(settings.CHROMA_PERSIST_DIR),
            # Cosine similarity is Chroma's default distance metric ("l2" is
            # the alternative); we set it explicitly for clarity.
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError("Failed to initialize the Chroma vector store.", str(exc)) from exc


def add_documents(chunks: List[Document]) -> int:
    """Embed and persist a list of document chunks. Returns count added."""
    if not chunks:
        return 0
    try:
        store = get_vector_store()
        store.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector store "
                    f"'{settings.CHROMA_COLLECTION_NAME}'.")
        return len(chunks)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError("Failed to write chunks to the vector store.", str(exc)) from exc


def similarity_search(query: str, k: int | None = None) -> List[Tuple[Document, float]]:
    """
    Perform cosine similarity search for `query` and return the top-k
    (document, score) pairs. Lower score = more similar for Chroma's
    default relevance function; we normalize below so higher = better.
    """
    k = k or settings.TOP_K
    try:
        store = get_vector_store()
        results = store.similarity_search_with_relevance_scores(query, k=k)
        # results: List[Tuple[Document, float]] where float is already a
        # normalized relevance score (higher = more similar) in [0, 1]-ish range
        filtered = [(doc, score) for doc, score in results
                    if score >= settings.SIMILARITY_SCORE_THRESHOLD]
        logger.info(f"Retrieved {len(filtered)}/{len(results)} chunks above "
                    f"threshold {settings.SIMILARITY_SCORE_THRESHOLD} for query.")
        return filtered or results  # fall back to unfiltered if all are below threshold
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError("Similarity search failed.", str(exc)) from exc


def collection_document_count() -> int:
    """Return how many chunks currently exist in the collection."""
    try:
        store = get_vector_store()
        return store._collection.count()  # noqa: SLF001 (Chroma has no public count() API)
    except Exception:  # noqa: BLE001
        return 0
