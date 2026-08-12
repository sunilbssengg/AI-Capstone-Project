"""
Centralized application configuration.
Reads values from the root `.env` file (see `.env.example`) and exposes
them as a single, typed `settings` object used across the whole app.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env from project root regardless of current working directory
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _get_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Gemini ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

    # --- Vector store ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "enterprise_docs")

    # --- Ingestion ---
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
    ALLOWED_EXTENSIONS: List[str] = field(
        default_factory=lambda: _get_list("ALLOWED_EXTENSIONS", ".pdf,.txt,.csv,.xlsx,.xls,.docx")
    )

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # --- Retrieval ---
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    SIMILARITY_SCORE_THRESHOLD: float = float(os.getenv("SIMILARITY_SCORE_THRESHOLD", "0.3"))

    # --- Guardrails ---
    MAX_QUESTION_LENGTH: int = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
    ENABLE_PROFANITY_FILTER: bool = _get_bool("ENABLE_PROFANITY_FILTER", True)
    ENABLE_PROMPT_INJECTION_FILTER: bool = _get_bool("ENABLE_PROMPT_INJECTION_FILTER", True)
    GROUNDEDNESS_MIN_OVERLAP: float = float(os.getenv("GROUNDEDNESS_MIN_OVERLAP", "0.15"))

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "data/app.log")

    # --- App ---
    APP_ENV: str = os.getenv("APP_ENV", "development")

    def resolve(self, relative_path: str) -> str:
        """Resolve a path relative to the project root."""
        return str(ROOT_DIR / relative_path)


settings = Settings()

# Ensure runtime directories exist
os.makedirs(settings.resolve(settings.UPLOAD_DIR), exist_ok=True)
os.makedirs(settings.resolve(settings.CHROMA_PERSIST_DIR), exist_ok=True)
