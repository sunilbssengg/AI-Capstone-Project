"""
Retrieval Agent — responsible for the "retrieve" step of the RAG flow:
embed the query, run cosine similarity search against the Chroma vector
store, and return the top-N relevant chunks with their source metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.exceptions import RetrievalError
from services.vector_store_service import similarity_search
from utils.logger import logger


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class RetrievalAgent:
    """A focused agent whose only job is high-quality context retrieval."""

    name = "retrieval_agent"
    description = (
        "Retrieves the most relevant chunks of enterprise document content "
        "for a given natural-language question using semantic (cosine) "
        "similarity search over the vector store."
    )

    def run(self, question: str, k: int | None = None) -> List[RetrievedChunk]:
        try:
            results = similarity_search(question, k=k)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError("Retrieval step failed.", str(exc)) from exc

        chunks = [
            RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=float(score),
            )
            for doc, score in results
        ]
        logger.info(f"RetrievalAgent returned {len(chunks)} chunks for question: '{question[:60]}...'")
        return chunks

    def as_tool_fn(self):
        """Expose this agent as a plain callable, usable as a LangChain Tool."""
        def _tool(question: str) -> str:
            chunks = self.run(question)
            if not chunks:
                return "No relevant content was found in the uploaded documents."
            return "\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)
        return _tool
