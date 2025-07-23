"""
QCM Generator Pro - Configuration Tests

Tests for configuration management, environment variables, and settings validation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.config import (
    CacheSettings,
    DatabaseSettings,
    GenerationSettings,
    LLMSettings,
    LoggingSettings,
    ProcessingSettings,
    SecuritySettings,
    Settings,
    UISettings,
    VectorStoreSettings,
)
from src.models.enums import (
    CacheStrategy,
    ChunkingStrategy,
    Environment,
    Language,
    LogLevel,
)


class TestDatabaseSettings:
    """Test database configuration settings."""

    def test_database_settings_defaults(self):
        """Test database settings with default values."""
        db_settings = DatabaseSettings()

        assert "sqlite:///" in db_settings.url
        assert db_settings.echo is False
        assert db_settings.pool_size == 5
        assert db_settings.pool_pre_ping is True
        assert db_settings.pool_recycle == 300

    def test_database_settings_from_env(self):
        """Test database settings from environment variables."""
        with patch.dict(os.environ, {
            'DATABASE_URL': 'postgresql://user:pass@localhost:5432/testdb',
            'DATABASE_ECHO': 'true',
            'DATABASE_POOL_SIZE': '10'
        }):
            db_settings = DatabaseSettings()

            assert db_settings.url == 'postgresql://user:pass@localhost:5432/testdb'
            assert db_settings.echo is True
            assert db_settings.pool_size == 10


class TestLLMSettings:
    """Test LLM configuration settings."""

    def test_llm_settings_defaults(self):
        """Test LLM settings with default values."""
        llm_settings = LLMSettings()

        assert llm_settings.default_model == "mistral-local"
        assert llm_settings.ollama_base_url == "http://localhost:11434"
        assert llm_settings.default_temperature == 0.7
        assert 0.0 <= llm_settings.gpu_memory_fraction <= 1.0

    def test_llm_settings_validation(self):
        """Test LLM settings validation."""
        # Valid settings
        llm_settings = LLMSettings(
            default_temperature=0.8,
            gpu_memory_fraction=0.95
        )
        assert llm_settings.default_temperature == 0.8
        assert llm_settings.gpu_memory_fraction == 0.95

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            LLMSettings(default_temperature=3.0)

        # Invalid GPU memory fraction
        with pytest.raises(ValidationError):
            LLMSettings(gpu_memory_fraction=1.5)

    def test_llm_settings_from_env(self):
        """Test LLM settings from environment variables."""
        with patch.dict(os.environ, {
            'LLM_DEFAULT_MODEL': 'custom-model',
            'LLM_OPENAI_API_KEY': 'sk-test123',
            'LLM_DEFAULT_TEMPERATURE': '0.5',
            'LLM_OLLAMA_TIMEOUT': '180'
        }):
            llm_settings = LLMSettings()

            assert llm_settings.default_model == 'custom-model'
            assert llm_settings.openai_api_key == 'sk-test123'
            assert llm_settings.default_temperature == 0.5
            assert llm_settings.ollama_timeout == 180


class TestGenerationSettings:
    """Test generation configuration settings."""

    def test_generation_settings_defaults(self):
        """Test generation settings with default values."""
        gen_settings = GenerationSettings()

        assert gen_settings.max_questions_per_session == 250
        assert gen_settings.default_questions_count == 20
        assert gen_settings.multiple_choice_ratio == 0.7
        assert gen_settings.multiple_selection_ratio == 0.3
        assert gen_settings.easy_difficulty_ratio == 0.3
        assert gen_settings.medium_difficulty_ratio == 0.5
        assert gen_settings.hard_difficulty_ratio == 0.2

    def test_generation_settings_ratio_validation(self):
        """Test ratio validation in generation settings."""
        # Valid ratios
        gen_settings = GenerationSettings(
            multiple_choice_ratio=0.6,
            multiple_selection_ratio=0.4
        )
        assert gen_settings.multiple_choice_ratio == 0.6
        assert gen_settings.multiple_selection_ratio == 0.4

        # Invalid ratios (don't sum to 1.0)
        with pytest.raises(ValidationError) as exc_info:
            GenerationSettings(
                multiple_choice_ratio=0.5,
                multiple_selection_ratio=0.4  # Sum = 0.9, not 1.0
            )
        assert "must sum to 1.0" in str(exc_info.value)

        # Test difficulty ratio validation
        with pytest.raises(ValidationError) as exc_info:
            GenerationSettings(
                easy_difficulty_ratio=0.2,
                medium_difficulty_ratio=0.3,
                hard_difficulty_ratio=0.4  # Sum = 0.9, not 1.0
            )
        assert "must sum to 1.0" in str(exc_info.value)

    def test_generation_settings_limits(self):
        """Test generation settings limits."""
        # Valid limits
        gen_settings = GenerationSettings(
            max_questions_per_session=100,
            min_questions_per_session=5
        )
        assert gen_settings.max_questions_per_session == 100
        assert gen_settings.min_questions_per_session == 5

        # Invalid limits
        with pytest.raises(ValidationError):
            GenerationSettings(max_questions_per_session=2000)  # Too high


class TestProcessingSettings:
    """Test processing configuration settings."""

    def test_processing_settings_defaults(self):
        """Test processing settings with default values."""
        proc_settings = ProcessingSettings()

        assert proc_settings.max_pdf_size_bytes == 50 * 1024 * 1024  # 50MB
        assert proc_settings.default_chunk_size == 1000
        assert proc_settings.default_chunk_overlap == 200
        assert proc_settings.chunking_strategy == ChunkingStrategy.SEMANTIC
        assert proc_settings.min_theme_confidence == 0.6

    def test_processing_settings_validation(self):
        """Test processing settings validation."""
        # Valid settings
        proc_settings = ProcessingSettings(
            default_chunk_size=800,
            default_chunk_overlap=100
        )
        assert proc_settings.default_chunk_size == 800
        assert proc_settings.default_chunk_overlap == 100

        # Invalid chunk size (too small)
        with pytest.raises(ValidationError):
            ProcessingSettings(default_chunk_size=50)

        # Invalid chunk overlap (too large)
        with pytest.raises(ValidationError):
            ProcessingSettings(default_chunk_overlap=2000)


class TestMainSettings:
    """Test main application settings."""

    def test_settings_defaults(self):
        """Test main settings with default values."""
        settings = Settings()

        assert settings.app_name == "QCM Generator Pro"
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.default_language == Language.FR
        assert Language.FR in settings.supported_languages
        assert Language.EN in settings.supported_languages

    def test_settings_properties(self):
        """Test settings computed properties."""
        # Development environment
        dev_settings = Settings(environment=Environment.DEVELOPMENT)
        assert dev_settings.is_development is True
        assert dev_settings.is_production is False
        assert dev_settings.is_testing is False

        # Production environment
        prod_settings = Settings(environment=Environment.PRODUCTION)
        assert prod_settings.is_development is False
        assert prod_settings.is_production is True
        assert prod_settings.is_testing is False

        # Testing environment
        test_settings = Settings(environment=Environment.TESTING)
        assert test_settings.is_development is False
        assert test_settings.is_production is False
        assert test_settings.is_testing is True

    def test_settings_paths(self):
        """Test settings path properties."""
        settings = Settings()

        assert isinstance(settings.base_dir, Path)
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.logs_dir, Path)

        # Paths should be absolute
        assert settings.base_dir.is_absolute()
        assert settings.data_dir.is_absolute()
        assert settings.logs_dir.is_absolute()

    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock base_dir to use temp directory
            settings = Settings()
            settings.data_dir = temp_path / "data"
            settings.logs_dir = temp_path / "logs"
            settings.llm.local_models_dir = temp_path / "models"

            # Ensure directories
            settings.ensure_directories()

            # Verify directories were created
            assert (temp_path / "data" / "pdfs").exists()
            assert (temp_path / "data" / "vectorstore").exists()
            assert (temp_path / "data" / "database").exists()
            assert (temp_path / "data" / "exports").exists()
            assert (temp_path / "data" / "cache").exists()
            assert (temp_path / "logs").exists()
            assert (temp_path / "models").exists()

    def test_nested_settings(self):
        """Test nested settings configuration."""
        settings = Settings()

        # Test that all nested settings are properly initialized
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.vector_store, VectorStoreSettings)
        assert isinstance(settings.processing, ProcessingSettings)
        assert isinstance(settings.generation, GenerationSettings)
        assert isinstance(settings.ui, UISettings)
        assert isinstance(settings.security, SecuritySettings)
        assert isinstance(settings.cache, CacheSettings)
        assert isinstance(settings.logging, LoggingSettings)

    def test_settings_from_env_file(self):
        """Test settings loading from environment variables."""
        env_vars = {
            'APP_NAME': 'Custom QCM Generator',
            'ENVIRONMENT': 'production',
            'DEBUG': 'false',
            'DEFAULT_LANGUAGE': 'en',
            'HOST': '0.0.0.0',
            'PORT': '9000'
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()

            assert settings.app_name == 'Custom QCM Generator'
            assert settings.environment == Environment.PRODUCTION
            assert settings.debug is False
            assert settings.default_language == Language.EN
            assert settings.host == '0.0.0.0'
            assert settings.port == 9000


class TestSettingsValidation:
    """Test settings validation and error handling."""

    def test_invalid_environment(self):
        """Test invalid environment value."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'invalid_env'}):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_language(self):
        """Test invalid language value."""
        with patch.dict(os.environ, {'DEFAULT_LANGUAGE': 'invalid_lang'}):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_port(self):
        """Test invalid port values."""
        # Port too low
        with patch.dict(os.environ, {'PORT': '500'}):
            with pytest.raises(ValidationError):
                Settings()

        # Port too high
        with patch.dict(os.environ, {'PORT': '70000'}):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_log_level(self):
        """Test invalid log level."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'INVALID'}):
            with pytest.raises(ValidationError):
                Settings()


class TestUISettings:
    """Test UI configuration settings."""

    def test_ui_settings_defaults(self):
        """Test UI settings with default values."""
        ui_settings = UISettings()

        assert ui_settings.server_port == 7860
        assert ui_settings.server_name == "127.0.0.1"
        assert ui_settings.share is False
        assert ui_settings.concurrent_sessions == 5

    def test_ui_settings_validation(self):
        """Test UI settings validation."""
        # Valid port
        ui_settings = UISettings(server_port=8080)
        assert ui_settings.server_port == 8080

        # Invalid port (too low)
        with pytest.raises(ValidationError):
            UISettings(server_port=500)

        # Invalid concurrent sessions (too high)
        with pytest.raises(ValidationError):
            UISettings(concurrent_sessions=50)


class TestSecuritySettings:
    """Test security configuration settings."""

    def test_security_settings_defaults(self):
        """Test security settings with default values."""
        sec_settings = SecuritySettings()

        assert sec_settings.api_key_enabled is False
        assert sec_settings.cors_allow_credentials is True
        assert sec_settings.rate_limit_enabled is True
        assert sec_settings.rate_limit_per_minute == 60

    def test_security_settings_cors_origins(self):
        """Test CORS origins configuration."""
        sec_settings = SecuritySettings()

        assert isinstance(sec_settings.cors_origins, list)
        assert len(sec_settings.cors_origins) > 0
        assert any("localhost" in origin for origin in sec_settings.cors_origins)


class TestCacheSettings:
    """Test cache configuration settings."""

    def test_cache_settings_defaults(self):
        """Test cache settings with default values."""
        cache_settings = CacheSettings()

        assert cache_settings.strategy == CacheStrategy.FILE
        assert cache_settings.ttl_seconds == 3600
        assert cache_settings.file_cache_max_size_mb == 500

    def test_cache_settings_validation(self):
        """Test cache settings validation."""
        # Valid settings
        cache_settings = CacheSettings(
            ttl_seconds=7200,
            file_cache_max_size_mb=1000
        )
        assert cache_settings.ttl_seconds == 7200
        assert cache_settings.file_cache_max_size_mb == 1000

        # Invalid TTL (too low)
        with pytest.raises(ValidationError):
            CacheSettings(ttl_seconds=30)

        # Invalid cache size (too small)
        with pytest.raises(ValidationError):
            CacheSettings(file_cache_max_size_mb=5)


class TestLoggingSettings:
    """Test logging configuration settings."""

    def test_logging_settings_defaults(self):
        """Test logging settings with default values."""
        log_settings = LoggingSettings()

        assert log_settings.level == LogLevel.INFO
        assert "%(asctime)s" in log_settings.format
        assert log_settings.max_size_mb == 10
        assert log_settings.backup_count == 3

    def test_logging_settings_validation(self):
        """Test logging settings validation."""
        # Valid settings
        log_settings = LoggingSettings(
            level=LogLevel.DEBUG,
            max_size_mb=20,
            backup_count=5
        )
        assert log_settings.level == LogLevel.DEBUG
        assert log_settings.max_size_mb == 20
        assert log_settings.backup_count == 5

        # Invalid max size (too small)
        with pytest.raises(ValidationError):
            LoggingSettings(max_size_mb=0)

        # Invalid backup count (too high)
        with pytest.raises(ValidationError):
            LoggingSettings(backup_count=20)


class TestSettingsIntegration:
    """Test settings integration and edge cases."""

    def test_settings_env_file_precedence(self):
        """Test that environment variables take precedence over defaults."""
        # Set environment variable
        with patch.dict(os.environ, {'APP_NAME': 'ENV_APP_NAME'}):
            settings = Settings()
            assert settings.app_name == 'ENV_APP_NAME'

    def test_settings_case_insensitive(self):
        """Test case-insensitive environment variable parsing."""
        # Environment variables are typically uppercase
        with patch.dict(os.environ, {
            'app_name': 'lowercase_name',  # Should work
            'DEBUG': 'True'
        }):
            settings = Settings()
            # The Settings class should handle case insensitivity
            assert settings.debug is True

    def test_settings_list_parsing(self):
        """Test parsing of list values from environment."""
        with patch.dict(os.environ, {
            'SUPPORTED_LANGUAGES': 'fr,en,es',
            'SECURITY_CORS_ORIGINS': 'http://localhost:3000,http://localhost:8000'
        }):
            settings = Settings()

            # Custom parsing should handle comma-separated lists
            if hasattr(settings, 'supported_languages'):
                assert Language.FR in settings.supported_languages
                assert Language.EN in settings.supported_languages
                assert Language.ES in settings.supported_languages

    def test_settings_with_missing_optional_values(self):
        """Test settings with missing optional configuration."""
        # Remove optional environment variables
        env_without_optional = {
            key: value for key, value in os.environ.items()
            if not key.startswith(('OPENAI_', 'ANTHROPIC_', 'REDIS_'))
        }

        with patch.dict(os.environ, env_without_optional, clear=True):
            settings = Settings()

            # Should still work with default values
            assert settings.llm.openai_api_key is None
            assert settings.llm.anthropic_api_key is None
            assert settings.cache.redis_url == "redis://localhost:6379/0"
