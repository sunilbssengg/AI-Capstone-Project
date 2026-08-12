"""
Document ingestion service.

Responsible for:
  1. Saving uploaded files to disk (data/uploads).
  2. Extracting raw text from PDF, TXT, CSV, XLSX/XLS, DOCX.
  3. Handing extracted text off to the chunking utility.

Each format has an isolated extractor function so adding a new format later
only requires adding one function + one dispatch entry.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
from langchain.docstore.document import Document
from pypdf import PdfReader

from core.config import settings
from core.exceptions import DocumentProcessingError
from utils.logger import logger
from utils.text_splitter import chunk_text
from utils.validators import safe_filename, validate_uploaded_file


def save_upload(filename: str, file_bytes: bytes) -> str:
    """Validate then persist an uploaded file to the uploads directory."""
    validate_uploaded_file(filename, file_bytes)
    clean_name = safe_filename(filename)
    dest_path = os.path.join(settings.resolve(settings.UPLOAD_DIR), clean_name)
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved upload -> {dest_path}")
    return dest_path


def _extract_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"Failed to parse PDF '{Path(path).name}'.", str(exc)) from exc


def _extract_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"Failed to read TXT '{Path(path).name}'.", str(exc)) from exc


def _extract_csv(path: str) -> str:
    try:
        df = pd.read_csv(path)
        return df.to_string(index=False)
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"Failed to parse CSV '{Path(path).name}'.", str(exc)) from exc


def _extract_excel(path: str) -> str:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
        parts = []
        for sheet_name, df in sheets.items():
            parts.append(f"--- Sheet: {sheet_name} ---\n{df.to_string(index=False)}")
        return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"Failed to parse Excel '{Path(path).name}'.", str(exc)) from exc


def _extract_docx(path: str) -> str:
    try:
        import docx  # python-docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"Failed to parse DOCX '{Path(path).name}'.", str(exc)) from exc


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".txt": _extract_txt,
    ".csv": _extract_csv,
    ".xlsx": _extract_excel,
    ".xls": _extract_excel,
    ".docx": _extract_docx,
}


def extract_text(path: str) -> str:
    """Dispatch to the correct extractor based on file extension."""
    ext = Path(path).suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise DocumentProcessingError(f"No extractor available for extension '{ext}'.")

    text = extractor(path)
    if not text or not text.strip():
        raise DocumentProcessingError(
            f"No extractable text found in '{Path(path).name}'. "
            "The file may be a scanned image or empty."
        )
    return text


def process_document(filename: str, file_bytes: bytes) -> List[Document]:
    """Full ingestion pipeline for one uploaded file: save -> extract -> chunk."""
    path = save_upload(filename, file_bytes)
    text = extract_text(path)
    metadata = {"source": Path(path).name}
    chunks = chunk_text(text, metadata=metadata)
    logger.info(f"Processed '{filename}' into {len(chunks)} chunks.")
    return chunks
