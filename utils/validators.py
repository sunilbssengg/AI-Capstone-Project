"""Input validation helpers for uploaded files and user queries."""
from __future__ import annotations

import os
from pathlib import Path

from core.config import settings
from core.exceptions import InvalidFileError


def validate_uploaded_file(filename: str, file_bytes: bytes) -> None:
    """Validate a file before it enters the ingestion pipeline.

    Checks: non-empty, allowed extension, size limit.
    """
    if not filename:
        raise InvalidFileError("No filename provided.")

    if file_bytes is None or len(file_bytes) == 0:
        raise InvalidFileError(f"'{filename}' is empty.")

    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidFileError(
            f"Unsupported file type '{ext}'. Allowed types: "
            f"{', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise InvalidFileError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{settings.MAX_FILE_SIZE_MB} MB limit."
        )


def safe_filename(filename: str) -> str:
    """Strip path components to prevent path traversal on save."""
    return os.path.basename(filename)
