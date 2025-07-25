"""
QCM Generator Pro - Basic Tests

Simple tests to verify core functionality.
"""

import pytest
from datetime import datetime

from src.models.enums import Language, QuestionType, Difficulty, ValidationStatus
from src.models.schemas import QuestionOption, GenerationConfig


class TestBasicFunctionality:
    """Test basic functionality."""

    def test_imports(self):
        """Test that core modules can be imported."""
        from src.core.config import settings
        from src.models.database import Document, Question
        from src.models.schemas import DocumentCreate, QuestionCreate
        
        assert settings is not None
        assert Document is not None
        assert Question is not None
        assert DocumentCreate is not None
        assert QuestionCreate is not None

    def test_enums(self):
        """Test enums are working."""
        assert Language.FR == "fr"
        assert Language.EN == "en"
        assert QuestionType.MULTIPLE_CHOICE == "multiple-choice"
        assert Difficulty.EASY == "easy"
        assert ValidationStatus.PENDING == "pending"

    def test_question_option(self):
        """Test question option creation."""
        option = QuestionOption(
            text="Test option",
            is_correct=True
        )
        
        assert option.text == "Test option"
        assert option.is_correct is True
        assert option.explanation is None

    def test_generation_config(self):
        """Test generation config creation."""
        config = GenerationConfig(
            num_questions=10,
            language=Language.FR
        )
        
        assert config.num_questions == 10
        assert config.language == Language.FR
        assert config.model == "mistral-local"  # default value
        assert config.temperature == 0.7  # default value

    def test_generation_config_validation(self):
        """Test generation config validation."""
        # Should raise error for invalid num_questions
        with pytest.raises(ValueError):
            GenerationConfig(num_questions=0)
            
        with pytest.raises(ValueError):
            GenerationConfig(num_questions=300)  # exceeds max

    def test_datetime_handling(self):
        """Test datetime objects work correctly."""
        now = datetime.now()
        assert isinstance(now, datetime)
        assert now.year >= 2024