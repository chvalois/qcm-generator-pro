"""
QCM Generator Pro - Custom Exceptions

This module defines all custom exceptions used throughout the application
with proper error codes, messages, and context information.
"""

from typing import Any, Union


class QCMGeneratorException(Exception):
    """Base exception class for QCM Generator Pro."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}')"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


# ============================================================================
# Configuration and Validation Exceptions
# ============================================================================

class ConfigurationError(QCMGeneratorException):
    """Raised when there's a configuration error."""
    pass


class ValidationError(QCMGeneratorException):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        **kwargs
    ):
        self.field = field
        self.value = value
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        super().__init__(message, details=details, **kwargs)


class SchemaValidationError(ValidationError):
    """Raised when Pydantic schema validation fails."""
    pass


# ============================================================================
# File and Document Processing Exceptions
# ============================================================================

class FileError(QCMGeneratorException):
    """Base exception for file-related errors."""
    pass


class FileNotFoundError(FileError):
    """Raised when a file is not found."""
    pass


class FileTooLargeError(FileError):
    """Raised when a file exceeds size limits."""

    def __init__(self, message: str, file_size: int, max_size: int, **kwargs):
        self.file_size = file_size
        self.max_size = max_size
        details = kwargs.get('details', {})
        details.update({
            'file_size': file_size,
            'max_size': max_size,
            'size_mb': file_size / (1024 * 1024),
            'max_mb': max_size / (1024 * 1024),
        })
        super().__init__(message, details=details, **kwargs)


class UnsupportedFileTypeError(FileError):
    """Raised when file type is not supported."""

    def __init__(self, message: str, file_type: str, supported_types: list[str], **kwargs):
        self.file_type = file_type
        self.supported_types = supported_types
        details = kwargs.get('details', {})
        details.update({
            'file_type': file_type,
            'supported_types': supported_types,
        })
        super().__init__(message, details=details, **kwargs)


class DocumentProcessingError(QCMGeneratorException):
    """Raised when document processing fails."""

    def __init__(self, message: str, document_id: int | None = None, stage: str | None = None, **kwargs):
        self.document_id = document_id
        self.stage = stage
        details = kwargs.get('details', {})
        if document_id:
            details['document_id'] = document_id
        if stage:
            details['processing_stage'] = stage
        super().__init__(message, details=details, **kwargs)


class PDFProcessingError(DocumentProcessingError):
    """Raised when PDF processing fails."""
    pass


class ThemeExtractionError(DocumentProcessingError):
    """Raised when theme extraction fails."""
    pass


class ChunkingError(DocumentProcessingError):
    """Raised when document chunking fails."""
    pass


# ============================================================================
# Database Exceptions
# ============================================================================

class DatabaseError(QCMGeneratorException):
    """Base exception for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class RecordNotFoundError(DatabaseError):
    """Raised when a database record is not found."""

    def __init__(self, message: str, model: str | None = None, record_id: int | str | None = None, **kwargs):
        self.model = model
        self.record_id = record_id
        details = kwargs.get('details', {})
        if model:
            details['model'] = model
        if record_id:
            details['record_id'] = str(record_id)
        super().__init__(message, details=details, **kwargs)


class DuplicateRecordError(DatabaseError):
    """Raised when attempting to create a duplicate record."""

    def __init__(self, message: str, model: str | None = None, conflicting_fields: dict[str, Any] | None = None, **kwargs):
        self.model = model
        self.conflicting_fields = conflicting_fields or {}
        details = kwargs.get('details', {})
        if model:
            details['model'] = model
        if conflicting_fields:
            details['conflicting_fields'] = conflicting_fields
        super().__init__(message, details=details, **kwargs)


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity constraints are violated."""
    pass


# ============================================================================
# LLM and Generation Exceptions
# ============================================================================

class LLMError(QCMGeneratorException):
    """Base exception for LLM-related errors."""
    pass


class ModelNotAvailableError(LLMError):
    """Raised when requested LLM model is not available."""

    def __init__(self, message: str, model_name: str, available_models: list[str] | None = None, **kwargs):
        self.model_name = model_name
        self.available_models = available_models or []
        details = kwargs.get('details', {})
        details.update({
            'requested_model': model_name,
            'available_models': self.available_models,
        })
        super().__init__(message, details=details, **kwargs)


class ModelResponseError(LLMError):
    """Raised when LLM model returns an error or invalid response."""

    def __init__(self, message: str, model_name: str | None = None, response: str | None = None, **kwargs):
        self.model_name = model_name
        self.response = response
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        if response:
            details['model_response'] = response[:500] + "..." if len(response) > 500 else response
        super().__init__(message, details=details, **kwargs)


class GenerationTimeoutError(LLMError):
    """Raised when question generation times out."""

    def __init__(self, message: str, timeout_seconds: int, **kwargs):
        self.timeout_seconds = timeout_seconds
        details = kwargs.get('details', {})
        details['timeout_seconds'] = timeout_seconds
        super().__init__(message, details=details, **kwargs)


class GenerationError(QCMGeneratorException):
    """Raised when question generation fails."""

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        question_number: int | None = None,
        **kwargs
    ):
        self.session_id = session_id
        self.question_number = question_number
        details = kwargs.get('details', {})
        if session_id:
            details['session_id'] = session_id
        if question_number:
            details['question_number'] = question_number
        super().__init__(message, details=details, **kwargs)


class InsufficientContentError(GenerationError):
    """Raised when document doesn't have enough content for generation."""
    pass


class InvalidGenerationConfigError(GenerationError):
    """Raised when generation configuration is invalid."""
    pass


# ============================================================================
# Vector Store Exceptions
# ============================================================================

class VectorStoreError(QCMGeneratorException):
    """Base exception for vector store operations."""
    pass


class EmbeddingError(VectorStoreError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, text_sample: str | None = None, model_name: str | None = None, **kwargs):
        self.text_sample = text_sample
        self.model_name = model_name
        details = kwargs.get('details', {})
        if text_sample:
            details['text_sample'] = text_sample[:100] + "..." if len(text_sample) > 100 else text_sample
        if model_name:
            details['embedding_model'] = model_name
        super().__init__(message, details=details, **kwargs)


class VectorSearchError(VectorStoreError):
    """Raised when vector search fails."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when vector collection is not found."""
    pass


# ============================================================================
# Question Validation Exceptions
# ============================================================================

class QuestionValidationError(QCMGeneratorException):
    """Base exception for question validation errors."""

    def __init__(
        self,
        message: str,
        question_id: int | None = None,
        validation_issues: list[str] | None = None,
        **kwargs
    ):
        self.question_id = question_id
        self.validation_issues = validation_issues or []
        details = kwargs.get('details', {})
        if question_id:
            details['question_id'] = question_id
        if validation_issues:
            details['validation_issues'] = validation_issues
        super().__init__(message, details=details, **kwargs)


class InvalidQuestionStructureError(QuestionValidationError):
    """Raised when question structure is invalid."""
    pass


class InvalidOptionsError(QuestionValidationError):
    """Raised when question options are invalid."""
    pass


class QualityThresholdError(QuestionValidationError):
    """Raised when question doesn't meet quality thresholds."""

    def __init__(
        self,
        message: str,
        quality_score: float,
        threshold: float,
        **kwargs
    ):
        self.quality_score = quality_score
        self.threshold = threshold
        details = kwargs.get('details', {})
        details.update({
            'quality_score': quality_score,
            'required_threshold': threshold,
        })
        super().__init__(message, details=details, **kwargs)


# ============================================================================
# Export Exceptions
# ============================================================================

class ExportError(QCMGeneratorException):
    """Base exception for export operations."""
    pass


class UnsupportedExportFormatError(ExportError):
    """Raised when export format is not supported."""

    def __init__(self, message: str, format_name: str, supported_formats: list[str], **kwargs):
        self.format_name = format_name
        self.supported_formats = supported_formats
        details = kwargs.get('details', {})
        details.update({
            'requested_format': format_name,
            'supported_formats': supported_formats,
        })
        super().__init__(message, details=details, **kwargs)


class ExportGenerationError(ExportError):
    """Raised when export file generation fails."""
    pass


class NoQuestionsToExportError(ExportError):
    """Raised when no questions are available for export."""
    pass


# ============================================================================
# API and Network Exceptions
# ============================================================================

class APIError(QCMGeneratorException):
    """Base exception for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        endpoint: str | None = None,
        **kwargs
    ):
        self.status_code = status_code
        self.endpoint = endpoint
        details = kwargs.get('details', {})
        if status_code:
            details['status_code'] = status_code
        if endpoint:
            details['endpoint'] = endpoint
        super().__init__(message, details=details, **kwargs)


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit: int,
        window_seconds: int,
        retry_after: int | None = None,
        **kwargs
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        details = kwargs.get('details', {})
        details.update({
            'rate_limit': limit,
            'window_seconds': window_seconds,
        })
        if retry_after:
            details['retry_after_seconds'] = retry_after
        super().__init__(message, details=details, **kwargs)


class NetworkError(QCMGeneratorException):
    """Raised when network operations fail."""
    pass


class TimeoutError(QCMGeneratorException):
    """Raised when operations timeout."""

    def __init__(self, message: str, operation: str, timeout_seconds: int, **kwargs):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        details = kwargs.get('details', {})
        details.update({
            'operation': operation,
            'timeout_seconds': timeout_seconds,
        })
        super().__init__(message, details=details, **kwargs)


# ============================================================================
# Resource and Performance Exceptions
# ============================================================================

class ResourceError(QCMGeneratorException):
    """Base exception for resource-related errors."""
    pass


class InsufficientResourcesError(ResourceError):
    """Raised when system resources are insufficient."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        required: int | float,
        available: int | float,
        **kwargs
    ):
        self.resource_type = resource_type
        self.required = required
        self.available = available
        details = kwargs.get('details', {})
        details.update({
            'resource_type': resource_type,
            'required': required,
            'available': available,
        })
        super().__init__(message, details=details, **kwargs)


class MemoryError(ResourceError):
    """Raised when memory limits are exceeded."""
    pass


class DiskSpaceError(ResourceError):
    """Raised when disk space is insufficient."""
    pass


class ConcurrencyLimitError(ResourceError):
    """Raised when concurrency limits are exceeded."""

    def __init__(self, message: str, current: int, limit: int, **kwargs):
        self.current = current
        self.limit = limit
        details = kwargs.get('details', {})
        details.update({
            'current_count': current,
            'limit': limit,
        })
        super().__init__(message, details=details, **kwargs)


# ============================================================================
# Session and State Exceptions
# ============================================================================

class SessionError(QCMGeneratorException):
    """Base exception for session-related errors."""
    pass


class SessionNotFoundError(SessionError):
    """Raised when generation session is not found."""

    def __init__(self, message: str, session_id: str, **kwargs):
        self.session_id = session_id
        details = kwargs.get('details', {})
        details['session_id'] = session_id
        super().__init__(message, details=details, **kwargs)


class SessionStateError(SessionError):
    """Raised when session is in invalid state for operation."""

    def __init__(
        self,
        message: str,
        session_id: str,
        current_state: str,
        required_state: str | None = None,
        **kwargs
    ):
        self.session_id = session_id
        self.current_state = current_state
        self.required_state = required_state
        details = kwargs.get('details', {})
        details.update({
            'session_id': session_id,
            'current_state': current_state,
        })
        if required_state:
            details['required_state'] = required_state
        super().__init__(message, details=details, **kwargs)


# ============================================================================
# Cache Exceptions
# ============================================================================

class CacheError(QCMGeneratorException):
    """Base exception for caching operations."""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""
    pass


class CacheKeyError(CacheError):
    """Raised when cache key is invalid or not found."""
    pass


# ============================================================================
# Exception Utilities
# ============================================================================

def format_validation_error(errors: list[dict[str, Any]]) -> str:
    """Format Pydantic validation errors into a readable string."""
    error_messages = []
    for error in errors:
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        message = error.get("msg", "Invalid value")
        error_messages.append(f"{field}: {message}")
    return "; ".join(error_messages)


def create_api_error_response(
    exception: QCMGeneratorException,
    status_code: int = 500
) -> dict[str, Any]:
    """Create a standardized API error response from an exception."""
    return {
        "error": {
            "type": exception.__class__.__name__,
            "message": exception.message,
            "code": exception.error_code,
            "details": exception.details,
        },
        "status_code": status_code,
    }


# Exception hierarchy for easy catching
VALIDATION_EXCEPTIONS = (
    ValidationError,
    SchemaValidationError,
    QuestionValidationError,
    InvalidQuestionStructureError,
    InvalidOptionsError,
)

FILE_EXCEPTIONS = (
    FileError,
    FileNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    DocumentProcessingError,
    PDFProcessingError,
)

DATABASE_EXCEPTIONS = (
    DatabaseError,
    DatabaseConnectionError,
    RecordNotFoundError,
    DuplicateRecordError,
    DatabaseIntegrityError,
)

GENERATION_EXCEPTIONS = (
    GenerationError,
    LLMError,
    ModelNotAvailableError,
    ModelResponseError,
    GenerationTimeoutError,
    InsufficientContentError,
)

RESOURCE_EXCEPTIONS = (
    ResourceError,
    InsufficientResourcesError,
    MemoryError,
    DiskSpaceError,
    ConcurrencyLimitError,
)
