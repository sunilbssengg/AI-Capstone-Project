"""Custom exception hierarchy used across the application.

Having a dedicated exception hierarchy lets the Streamlit / API layer
catch precise failure modes and show the user a clean, actionable message
instead of a raw stack trace.
"""


class AppBaseException(Exception):
    """Base class for all application-specific errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(message)


class InvalidFileError(AppBaseException):
    """Raised when an uploaded file fails validation (type, size, empty)."""


class DocumentProcessingError(AppBaseException):
    """Raised when a document cannot be parsed / text cannot be extracted."""


class EmbeddingError(AppBaseException):
    """Raised when embedding generation fails (e.g. API error, quota)."""


class VectorStoreError(AppBaseException):
    """Raised when the vector database read/write operation fails."""


class RetrievalError(AppBaseException):
    """Raised when semantic retrieval fails or returns nothing usable."""


class LLMGenerationError(AppBaseException):
    """Raised when the LLM call fails (timeout, API error, safety block)."""


class GuardrailViolation(AppBaseException):
    """Raised when input/output guardrail checks block a request."""


class InvalidInputError(AppBaseException):
    """Raised when user-provided input fails validation."""
