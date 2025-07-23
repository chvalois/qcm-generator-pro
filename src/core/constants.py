"""
QCM Generator Pro - Application Constants

This module defines all application-wide constants and configuration values
that are used throughout the system.
"""

from pathlib import Path

# ============================================================================
# Application Information
# ============================================================================

APP_NAME = "QCM Generator Pro"
APP_DESCRIPTION = "Local multilingual QCM generation system from PDF documents"
APP_VERSION = "0.1.0"
API_VERSION = "v1"

# ============================================================================
# File and Directory Constants
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
TESTS_DIR = BASE_DIR / "tests"

# Data subdirectories
PDFS_DIR = DATA_DIR / "pdfs"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
DATABASE_DIR = DATA_DIR / "database"
EXPORTS_DIR = DATA_DIR / "exports"
CACHE_DIR = DATA_DIR / "cache"

# Configuration files
ENV_FILE = BASE_DIR / ".env"
EXAMPLE_ENV_FILE = BASE_DIR / ".env.example"

# ============================================================================
# Database Constants
# ============================================================================

# Default database URLs
DEFAULT_DATABASE_URL = f"sqlite:///{DATABASE_DIR}/qcm_generator.db"
TEST_DATABASE_URL = f"sqlite:///{DATABASE_DIR}/test_qcm_generator.db"

# Connection pool settings
DB_POOL_SIZE = 5
DB_POOL_RECYCLE = 300
DB_POOL_TIMEOUT = 30

# ============================================================================
# File Processing Constants
# ============================================================================

# File size limits (in bytes)
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MIN_FILE_SIZE_BYTES = 1024  # 1 KB

# Supported file types
SUPPORTED_PDF_EXTENSIONS = [".pdf"]
SUPPORTED_MIME_TYPES = ["application/pdf"]

# Text processing limits
MAX_TEXT_LENGTH = 1_000_000  # 1 million characters
MIN_TEXT_LENGTH = 100  # Minimum text for processing

# ============================================================================
# Chunking Constants
# ============================================================================

# Chunk size limits
DEFAULT_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4000

# Chunk overlap limits
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 1000

# ============================================================================
# Question Generation Constants
# ============================================================================

# Question limits
MIN_QUESTIONS_PER_SESSION = 1
MAX_QUESTIONS_PER_SESSION = 250
DEFAULT_QUESTIONS_COUNT = 20

# Question text limits
MIN_QUESTION_LENGTH = 10
MAX_QUESTION_LENGTH = 2000
MIN_OPTION_LENGTH = 1
MAX_OPTION_LENGTH = 500
MAX_EXPLANATION_LENGTH = 1000

# Options limits
MIN_OPTIONS_COUNT = 3
MAX_OPTIONS_COUNT = 6
MIN_CORRECT_ANSWERS = 1

# ============================================================================
# Quality and Validation Constants
# ============================================================================

# Quality thresholds
MIN_CONFIDENCE_SCORE = 0.6
MIN_VALIDATION_SCORE = 0.7
MIN_QUALITY_SCORE = 0.6
AUTO_APPROVE_THRESHOLD = 0.8

# Theme extraction thresholds
MIN_THEME_CONFIDENCE = 0.6
MAX_THEMES_PER_DOCUMENT = 10
THEME_KEYWORDS_LIMIT = 20

# Language detection threshold
LANGUAGE_DETECTION_THRESHOLD = 0.8

# ============================================================================
# Generation Parameters
# ============================================================================

# Default LLM parameters
DEFAULT_TEMPERATURE = 0.7
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

DEFAULT_MAX_TOKENS = 512
MIN_MAX_TOKENS = 50
MAX_MAX_TOKENS = 2000

DEFAULT_TOP_P = 0.9
MIN_TOP_P = 0.1
MAX_TOP_P = 1.0

# Progressive generation batches
DEFAULT_BATCH_SIZES = [1, 5, -1]  # -1 means "all remaining"

# ============================================================================
# Timeout Constants (in seconds)
# ============================================================================

# Processing timeouts
PDF_PROCESSING_TIMEOUT = 600  # 10 minutes
THEME_EXTRACTION_TIMEOUT = 300  # 5 minutes
EMBEDDING_TIMEOUT = 900  # 15 minutes

# Generation timeouts
GENERATION_TIMEOUT = 1800  # 30 minutes
MODEL_RESPONSE_TIMEOUT = 120  # 2 minutes
VALIDATION_TIMEOUT = 60  # 1 minute

# Network timeouts
HTTP_TIMEOUT = 30
API_TIMEOUT = 120

# ============================================================================
# Cache Constants
# ============================================================================

# Cache TTL (Time To Live) in seconds
DEFAULT_CACHE_TTL = 3600  # 1 hour
SHORT_CACHE_TTL = 300  # 5 minutes
LONG_CACHE_TTL = 86400  # 24 hours

# Cache size limits
MAX_MEMORY_CACHE_SIZE = 100  # Maximum number of items
MAX_FILE_CACHE_SIZE_MB = 500  # 500 MB

# ============================================================================
# Vector Store Constants
# ============================================================================

# ChromaDB settings
DEFAULT_CHROMA_COLLECTION = "qcm_documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Search settings
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 50
SIMILARITY_THRESHOLD = 0.7

# Embedding dimensions (model-specific)
EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# ============================================================================
# API Constants
# ============================================================================

# API versioning
API_V1_PREFIX = "/api/v1"

# Rate limiting
DEFAULT_RATE_LIMIT = 60  # requests per minute
BURST_RATE_LIMIT = 100  # burst limit

# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MIN_PAGE_SIZE = 1

# Request/Response limits
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_RESPONSE_SIZE = 50 * 1024 * 1024  # 50 MB

# ============================================================================
# Export Constants
# ============================================================================

# Export file limits
MAX_EXPORT_QUESTIONS = 1000
EXPORT_FILENAME_MAX_LENGTH = 255

# CSV format settings (Udemy-compatible)
CSV_DELIMITER = ","
CSV_ENCODING = "utf-8"
CSV_QUOTE_CHAR = '"'
CSV_ESCAPE_CHAR = "\\"

# Export TTL (how long export files are kept)
EXPORT_FILE_TTL = 7 * 24 * 3600  # 7 days

# ============================================================================
# UI Constants
# ============================================================================

# Gradio settings
DEFAULT_GRADIO_PORT = 7860
MAX_CONCURRENT_SESSIONS = 5

# File upload limits for UI
UI_MAX_FILE_SIZE_MB = 50
UI_ALLOWED_EXTENSIONS = [".pdf"]

# ============================================================================
# Logging Constants
# ============================================================================

# Log file settings
LOG_FILENAME = "qcm_generator.log"
MAX_LOG_SIZE_MB = 10
LOG_BACKUP_COUNT = 3

# Log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# ============================================================================
# Security Constants
# ============================================================================

# Password/Token settings
MIN_API_KEY_LENGTH = 32
SESSION_ID_LENGTH = 32

# CORS settings
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:7860",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:7860",
]

# ============================================================================
# Model-Specific Constants
# ============================================================================

# Local model settings
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_DEFAULT_HOST = "localhost"
OLLAMA_DEFAULT_URL = f"http://{OLLAMA_DEFAULT_HOST}:{OLLAMA_DEFAULT_PORT}"

# Recommended local models
RECOMMENDED_LOCAL_MODELS = [
    "mistral:7b-instruct",
    "llama3:8b-instruct",
    "phi3:medium",
]

# Model context lengths (in tokens)
MODEL_CONTEXT_LENGTHS = {
    "mistral:7b-instruct": 4096,
    "llama3:8b-instruct": 8192,
    "phi3:medium": 4096,
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "claude-3-haiku-20240307": 200000,
}

# ============================================================================
# Language-Specific Constants
# ============================================================================

# Language codes and names
LANGUAGE_NAMES = {
    "fr": "Français",
    "en": "English",
    "es": "Español",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
}

# spaCy model mappings
SPACY_MODELS = {
    "fr": "fr_core_news_md",
    "en": "en_core_web_md",
    "es": "es_core_news_md",
    "de": "de_core_news_md",
    "it": "it_core_news_md",
    "pt": "pt_core_news_md",
}

# ============================================================================
# Error Messages
# ============================================================================

# Validation error messages
ERROR_MESSAGES = {
    "FILE_TOO_LARGE": "File size exceeds maximum allowed size",
    "INVALID_FILE_TYPE": "File type not supported",
    "PROCESSING_FAILED": "Document processing failed",
    "GENERATION_FAILED": "Question generation failed",
    "INVALID_CONFIGURATION": "Invalid configuration provided",
    "MODEL_NOT_AVAILABLE": "Requested model is not available",
    "INSUFFICIENT_CONTENT": "Document content insufficient for question generation",
    "VALIDATION_FAILED": "Question validation failed",
    "EXPORT_FAILED": "Export generation failed",
    "DATABASE_ERROR": "Database operation failed",
    "NETWORK_ERROR": "Network connection failed",
}

# ============================================================================
# Environment-Specific Constants
# ============================================================================

# Development settings
DEV_RELOAD = True
DEV_DEBUG = True
DEV_LOG_LEVEL = "DEBUG"

# Production settings
PROD_RELOAD = False
PROD_DEBUG = False
PROD_LOG_LEVEL = "INFO"

# Testing settings
TEST_DATABASE_NAME = "test_qcm_generator.db"
TEST_CHUNK_SIZE = 100  # Smaller chunks for faster tests
TEST_QUESTIONS_LIMIT = 10  # Limit for test sessions

# ============================================================================
# Monitoring Constants
# ============================================================================

# Metrics collection
METRICS_ENABLED = False
PROMETHEUS_PORT = 8001

# Health check endpoints
HEALTH_CHECK_TIMEOUT = 5
HEALTH_CHECK_ENDPOINTS = [
    "database",
    "vector_store",
    "llm_service",
    "file_system",
]

# ============================================================================
# Performance Constants
# ============================================================================

# Concurrency limits
MAX_CONCURRENT_GENERATIONS = 3
MAX_CONCURRENT_PROCESSING = 5
MAX_WORKER_THREADS = 8

# Batch processing sizes
EMBEDDING_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 10
EXPORT_BATCH_SIZE = 50

# Memory limits (in MB)
MAX_MEMORY_USAGE_MB = 2048
EMBEDDING_MEMORY_LIMIT_MB = 512
