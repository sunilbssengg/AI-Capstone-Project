"""
Chunking strategy for semantic search.

Concept: raw document text is first conceptually "tokenized" (broken into
words/sub-word units by the embedding model later on); here we pre-split
the text into overlapping, model-friendly chunks so each chunk stays under
the embedding model's context window and retains enough surrounding
context (via overlap) to avoid cutting sentences/ideas in half.

We use LangChain's RecursiveCharacterTextSplitter (a smarter variant of
CharacterTextSplitter that tries paragraph -> sentence -> word boundaries
in order) for production use, matching the CharacterTextSplitter example
given in the requirements:

    from langchain.text_splitter import CharacterTextSplitter
    splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_text(text)
"""
from __future__ import annotations

from typing import List

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from core.config import settings
from utils.logger import logger


def get_splitter(use_recursive: bool = True):
    """Return a configured LangChain text splitter."""
    if use_recursive:
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return CharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separator="\n",
    )


def chunk_text(text: str, metadata: dict | None = None) -> List[Document]:
    """Split raw text into LangChain Document chunks with metadata attached."""
    if not text or not text.strip():
        logger.warning("chunk_text called with empty text; returning no chunks.")
        return []

    splitter = get_splitter(use_recursive=True)
    raw_chunks = splitter.split_text(text)

    documents = [
        Document(page_content=chunk, metadata=dict(metadata or {}, chunk_index=i))
        for i, chunk in enumerate(raw_chunks)
    ]
    logger.info(f"Split text into {len(documents)} chunks "
                f"(chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}).")
    return documents


if __name__ == "__main__":
    # Quick demo matching the example in the project spec
    demo_text = (
        "LangChain simplifies AI workflows. It enables advanced "
        "retrieval-augmented generation systems for NLP tasks."
    )
    splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator=" ")
    print(splitter.split_text(demo_text))
