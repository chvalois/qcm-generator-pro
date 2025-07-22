"""
QCM Generator Pro - Application Configuration

This module provides centralized configuration management using Pydantic Settings
with environment variable support and validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings

from ..models.enums import (
    CacheStrategy,
    ChunkingStrategy,
    EmbeddingModel,
    Environment,
    Language,
    LogLevel,
    ModelType,
    ValidationMode,
)


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="sqlite:///./data/database/qcm_generator.db",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=5, description="Connection pool size")
    pool_pre_ping: bool = Field(default=True, description="Enable connection health checks")
    pool_recycle: int = Field(default=300, description="Connection recycle time in seconds")
    
    class Config:
        env_prefix = "DATABASE_"


class LLMSettings(BaseSettings):
    """LLM configuration settings."""
    
    # Default model configuration
    default_model: str = Field(default="mistral-local", description="Default LLM model to use")
    model_type: ModelType = Field(default=ModelType.LOCAL, description="Type of default model")
    
    # Local models configuration
    local_models_dir: Path = Field(default=Path("./models"), description="Directory for local models")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_timeout: int = Field(default=120, description="Ollama request timeout")
    
    # OpenAI configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    openai_model: str = Field(default="gpt-3.5-turbo", description="Default OpenAI model")
    openai_timeout: int = Field(default=120, description="OpenAI request timeout")
    
    # Anthropic configuration
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-3-haiku-20240307", description="Default Anthropic model")
    anthropic_timeout: int = Field(default=120, description="Anthropic request timeout")
    
    # GPU configuration
    cuda_visible_devices: str = Field(default="0", description="CUDA visible devices")
    gpu_memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0, description="GPU memory fraction")
    
    # Generation parameters
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Default temperature")
    default_max_tokens: int = Field(default=512, ge=50, le=2000, description="Default max tokens")
    default_top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Default top_p")
    
    class Config:
        env_prefix = "LLM_"


class VectorStoreSettings(BaseSettings):
    """Vector store configuration settings."""
    
    # ChromaDB configuration
    chroma_db_path: Path = Field(default=Path("./data/vectorstore"), description="ChromaDB storage path")
    chroma_collection_name: str = Field(default="qcm_documents", description="ChromaDB collection name")
    
    # Embedding configuration
    embedding_model: EmbeddingModel = Field(
        default=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2,
        description="Embedding model to use"
    )
    embedding_device: str = Field(default="cpu", description="Device for embedding computation")
    embedding_batch_size: int = Field(default=32, ge=1, le=128, description="Embedding batch size")
    
    # Search configuration
    default_search_limit: int = Field(default=5, ge=1, le=50, description="Default number of search results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity search threshold")
    
    class Config:
        env_prefix = "VECTOR_"


class ProcessingSettings(BaseSettings):
    """Document processing configuration settings."""
    
    # File size limits (in bytes)
    max_pdf_size_bytes: int = Field(default=50 * 1024 * 1024, description="Maximum PDF file size")
    max_upload_size_bytes: int = Field(default=100 * 1024 * 1024, description="Maximum upload size")
    
    # Chunking configuration
    default_chunk_size: int = Field(default=1000, ge=100, le=4000, description="Default chunk size")
    default_chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Default chunk overlap")
    min_chunk_size: int = Field(default=100, ge=50, le=500, description="Minimum chunk size")
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.SEMANTIC, description="Default chunking strategy")
    
    # OCR configuration
    ocr_enabled: bool = Field(default=False, description="Enable OCR for scanned documents")
    tesseract_path: Optional[Path] = Field(default=None, description="Path to Tesseract executable")
    
    # spaCy configuration
    spacy_model: str = Field(default="fr_core_news_md", description="spaCy model for text processing")
    spacy_disable_components: List[str] = Field(default=["ner", "parser"], description="spaCy components to disable")
    
    # Theme extraction
    min_theme_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum theme confidence")
    max_themes_per_document: int = Field(default=10, ge=1, le=50, description="Maximum themes per document")
    theme_keywords_limit: int = Field(default=20, ge=5, le=50, description="Maximum keywords per theme")
    
    # Language detection
    language_detection_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Language detection threshold")
    
    class Config:
        env_prefix = "PROCESSING_"


class GenerationSettings(BaseSettings):
    """QCM generation configuration settings."""
    
    # Generation limits
    max_questions_per_session: int = Field(default=250, ge=1, le=1000, description="Maximum questions per session")
    min_questions_per_session: int = Field(default=1, ge=1, le=10, description="Minimum questions per session")
    default_questions_count: int = Field(default=20, ge=1, le=100, description="Default number of questions")
    
    # Question type distribution (as percentages)
    multiple_choice_ratio: float = Field(default=0.7, ge=0.0, le=1.0, description="Multiple choice ratio")
    multiple_selection_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="Multiple selection ratio")
    
    # Difficulty distribution (as percentages)
    easy_difficulty_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="Easy difficulty ratio")
    medium_difficulty_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Medium difficulty ratio")
    hard_difficulty_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Hard difficulty ratio")
    
    # Progressive validation
    validation_mode: ValidationMode = Field(default=ValidationMode.PROGRESSIVE, description="Validation mode")
    validation_batches: List[int] = Field(default=[1, 5, -1], description="Progressive validation batch sizes")
    
    # Quality thresholds
    min_quality_score: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum quality score")
    auto_approve_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Auto-approval threshold")
    
    # Timeouts (in seconds)
    generation_timeout: int = Field(default=1800, ge=60, le=3600, description="Generation timeout")
    model_response_timeout: int = Field(default=120, ge=10, le=300, description="Model response timeout")
    
    @field_validator('multiple_selection_ratio')
    @classmethod
    def validate_question_type_ratios(cls, v, info):
        values = info.data if info else {}
        mc_ratio = values.get('multiple_choice_ratio', 0.7)
        total = mc_ratio + v
        if not 0.95 <= total <= 1.05:  # Allow small floating point errors
            raise ValueError('Question type ratios must sum to 1.0')
        return v
    
    @field_validator('hard_difficulty_ratio')
    @classmethod
    def validate_difficulty_ratios(cls, v, info):
        values = info.data if info else {}
        easy = values.get('easy_difficulty_ratio', 0.3)
        medium = values.get('medium_difficulty_ratio', 0.5)
        total = easy + medium + v
        if not 0.95 <= total <= 1.05:  # Allow small floating point errors
            raise ValueError('Difficulty ratios must sum to 1.0')
        return v
    
    class Config:
        env_prefix = "GENERATION_"


class UISettings(BaseSettings):
    """UI configuration settings."""
    
    # Gradio server configuration
    server_port: int = Field(default=7860, ge=1024, le=65535, description="Gradio server port")
    server_name: str = Field(default="127.0.0.1", description="Gradio server host")
    share: bool = Field(default=False, description="Enable Gradio sharing")
    debug: bool = Field(default=False, description="Enable Gradio debug mode")
    
    # UI limits
    max_file_size_mb: int = Field(default=50, ge=1, le=500, description="Maximum file size in MB")
    concurrent_sessions: int = Field(default=5, ge=1, le=20, description="Maximum concurrent sessions")
    
    class Config:
        env_prefix = "UI_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # API security
    api_key_enabled: bool = Field(default=False, description="Enable API key authentication")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    
    # CORS configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:7860"],
        description="CORS allowed origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: List[str] = Field(default=["*"], description="CORS allowed methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="CORS allowed headers")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000, description="Rate limit per minute")
    
    class Config:
        env_prefix = "SECURITY_"


class CacheSettings(BaseSettings):
    """Caching configuration settings."""
    
    # Cache strategy
    strategy: CacheStrategy = Field(default=CacheStrategy.FILE, description="Caching strategy")
    
    # Redis configuration (if using Redis)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Cache TTL in seconds")
    
    # File cache configuration
    file_cache_dir: Path = Field(default=Path("./data/cache"), description="File cache directory")
    file_cache_max_size_mb: int = Field(default=500, ge=10, le=10000, description="Max file cache size in MB")
    
    class Config:
        env_prefix = "CACHE_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    # Log levels and format
    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # File logging
    file: Optional[Path] = Field(default=Path("./logs/qcm_generator.log"), description="Log file path")
    max_size_mb: int = Field(default=10, ge=1, le=100, description="Max log file size in MB")
    backup_count: int = Field(default=3, ge=1, le=10, description="Number of log file backups")
    
    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application information
    app_name: str = Field(default="QCM Generator Pro", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    
    # Server configuration
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    reload: bool = Field(default=True, description="Enable auto-reload in development")
    
    # Default language
    default_language: Language = Field(default=Language.FR, description="Default application language")
    supported_languages: List[Language] = Field(
        default=[Language.FR, Language.EN, Language.ES, Language.DE],
        description="Supported languages"
    )
    
    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    ui: UISettings = Field(default_factory=UISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Computed properties
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING
    
    @property
    def base_dir(self) -> Path:
        """Get the base directory of the application."""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.base_dir / "data"
    
    @property
    def logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.base_dir / "logs"
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data_dir / "pdfs",
            self.data_dir / "vectorstore", 
            self.data_dir / "database",
            self.data_dir / "exports",
            self.data_dir / "cache",
            self.logs_dir,
            self.llm.local_models_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        
        # Custom environment variable parsing
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> any:
            # Handle comma-separated lists
            if field_name in ['supported_languages', 'cors_origins', 'cors_allow_methods', 'cors_allow_headers', 'spacy_disable_components']:
                return [item.strip() for item in raw_val.split(',') if item.strip()]
            # Handle JSON lists for validation_batches
            if field_name == 'validation_batches':
                import json
                try:
                    return json.loads(raw_val)
                except json.JSONDecodeError:
                    return [int(x.strip()) for x in raw_val.split(',') if x.strip()]
            return raw_val


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()