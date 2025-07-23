"""
QCM Generator Pro - Schemas Unit Tests

Tests for Pydantic schemas validation and serialization.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.enums import (
    ChunkingStrategy,
    Difficulty,
    EmbeddingModel,
    ExportFormat,
    Language,
    ProcessingStatus,
    QuestionType,
    ValidationStatus,
)
from src.models.schemas import (
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
    ErrorResponse,
    ExportRequest,
    GenerationConfig,
    GenerationSessionCreate,
    ProcessingConfig,
    QuestionCreate,
    QuestionOption,
    QuestionResponse,
    QuestionUpdate,
    SuccessResponse,
    ThemeCreate,
    ThemeResponse,
)


class TestDocumentSchemas:
    """Test document-related schemas."""

    def test_document_create_valid(self):
        """Test creating valid document."""
        data = {
            "filename": "test_document.pdf",
            "file_path": "/path/to/test_document.pdf",
            "file_size": 1024000,
            "language": Language.FR,
            "title": "Test Document",
            "author": "Test Author"
        }

        document = DocumentCreate(**data)
        assert document.filename == "test_document.pdf"
        assert document.language == Language.FR
        assert document.file_size == 1024000

    def test_document_create_validation_errors(self):
        """Test document creation validation errors."""
        # Empty filename
        with pytest.raises(ValidationError) as exc_info:
            DocumentCreate(filename="", file_path="/path/to/file.pdf")
        assert "ensure this value has at least 1 characters" in str(exc_info.value)

        # Invalid language
        with pytest.raises(ValidationError) as exc_info:
            DocumentCreate(filename="test.pdf", file_path="/path", language="invalid")

        # Filename too long
        long_filename = "a" * 300 + ".pdf"
        with pytest.raises(ValidationError) as exc_info:
            DocumentCreate(filename=long_filename, file_path="/path")

    def test_document_update_partial(self):
        """Test partial document updates."""
        # Only update title
        update = DocumentUpdate(title="New Title")
        assert update.title == "New Title"
        assert update.filename is None

        # Update multiple fields
        update = DocumentUpdate(
            title="Updated Title",
            author="Updated Author",
            language=Language.EN
        )
        assert update.title == "Updated Title"
        assert update.author == "Updated Author"
        assert update.language == Language.EN

    def test_document_response_serialization(self):
        """Test document response serialization."""
        now = datetime.now()
        data = {
            "id": 1,
            "filename": "test.pdf",
            "file_path": "/path/test.pdf",
            "language": Language.FR,
            "processing_status": ProcessingStatus.COMPLETED,
            "created_at": now,
            "updated_at": now,
            "upload_date": now,
            "themes_count": 3,
            "questions_count": 15
        }

        response = DocumentResponse(**data)
        assert response.id == 1
        assert response.themes_count == 3
        assert response.questions_count == 15


class TestQuestionSchemas:
    """Test question-related schemas."""

    def test_question_option_valid(self):
        """Test valid question option."""
        option = QuestionOption(
            text="Python is a programming language",
            is_correct=True,
            explanation="Python is indeed a programming language"
        )

        assert option.text == "Python is a programming language"
        assert option.is_correct is True
        assert option.explanation == "Python is indeed a programming language"

    def test_question_create_multiple_choice(self):
        """Test creating multiple choice question."""
        options = [
            QuestionOption(text="Option 1", is_correct=True),
            QuestionOption(text="Option 2", is_correct=False),
            QuestionOption(text="Option 3", is_correct=False),
        ]

        question = QuestionCreate(
            document_id=1,
            session_id="test_session",
            question_text="What is the correct answer?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            language=Language.EN,
            difficulty=Difficulty.MEDIUM,
            options=options,
            explanation="Option 1 is correct because...",
            generation_order=1
        )

        assert question.question_type == QuestionType.MULTIPLE_CHOICE
        assert len(question.options) == 3
        assert sum(opt.is_correct for opt in question.options) == 1

    def test_question_create_validation_errors(self):
        """Test question creation validation errors."""
        # Too few options
        with pytest.raises(ValidationError) as exc_info:
            QuestionCreate(
                document_id=1,
                session_id="test",
                question_text="Question?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                language=Language.EN,
                difficulty=Difficulty.EASY,
                options=[QuestionOption(text="Only one", is_correct=True)],
                generation_order=1
            )

        # Multiple choice with multiple correct answers
        options = [
            QuestionOption(text="Option 1", is_correct=True),
            QuestionOption(text="Option 2", is_correct=True),  # Invalid for multiple choice
            QuestionOption(text="Option 3", is_correct=False),
        ]

        with pytest.raises(ValidationError) as exc_info:
            QuestionCreate(
                document_id=1,
                session_id="test",
                question_text="Question?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                language=Language.EN,
                difficulty=Difficulty.EASY,
                options=options,
                generation_order=1
            )
        assert "Multiple choice questions must have exactly one correct answer" in str(exc_info.value)

        # Question text too short
        with pytest.raises(ValidationError) as exc_info:
            QuestionCreate(
                document_id=1,
                session_id="test",
                question_text="Short",  # Too short
                question_type=QuestionType.MULTIPLE_CHOICE,
                language=Language.EN,
                difficulty=Difficulty.EASY,
                options=[
                    QuestionOption(text="A", is_correct=True),
                    QuestionOption(text="B", is_correct=False),
                    QuestionOption(text="C", is_correct=False),
                ],
                generation_order=1
            )

    def test_question_create_multiple_selection(self):
        """Test creating multiple selection question."""
        options = [
            QuestionOption(text="Correct 1", is_correct=True),
            QuestionOption(text="Correct 2", is_correct=True),
            QuestionOption(text="Incorrect", is_correct=False),
        ]

        question = QuestionCreate(
            document_id=1,
            session_id="test_session",
            question_text="Select all correct answers:",
            question_type=QuestionType.MULTIPLE_SELECTION,
            language=Language.EN,
            difficulty=Difficulty.HARD,
            options=options,
            generation_order=1
        )

        assert question.question_type == QuestionType.MULTIPLE_SELECTION
        assert sum(opt.is_correct for opt in question.options) == 2

    def test_question_update_partial(self):
        """Test partial question updates."""
        update = QuestionUpdate(
            validation_status=ValidationStatus.APPROVED,
            validation_feedback="Question looks good"
        )

        assert update.validation_status == ValidationStatus.APPROVED
        assert update.validation_feedback == "Question looks good"
        assert update.question_text is None  # Not updated

    def test_question_response_complete(self):
        """Test complete question response."""
        now = datetime.now()
        options = [
            QuestionOption(text="Option 1", is_correct=True),
            QuestionOption(text="Option 2", is_correct=False),
        ]

        response = QuestionResponse(
            id=1,
            document_id=1,
            session_id="test_session",
            question_text="What is Python?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            language=Language.EN,
            difficulty=Difficulty.EASY,
            options=options,
            explanation="Python is a programming language",
            validation_status=ValidationStatus.APPROVED,
            generation_order=1,
            created_at=now,
            updated_at=now
        )

        assert response.id == 1
        assert response.validation_status == ValidationStatus.APPROVED
        assert len(response.options) == 2


class TestGenerationSchemas:
    """Test generation configuration schemas."""

    def test_generation_config_default(self):
        """Test generation config with defaults."""
        config = GenerationConfig()

        assert config.num_questions == 20  # Default
        assert config.language == Language.FR
        assert config.validation_mode.value == "progressive"
        assert sum(config.question_types.values()) == pytest.approx(1.0)
        assert sum(config.difficulty_distribution.values()) == pytest.approx(1.0)

    def test_generation_config_custom(self):
        """Test custom generation configuration."""
        config = GenerationConfig(
            num_questions=50,
            language=Language.EN,
            model="custom-model",
            temperature=0.8,
            question_types={
                QuestionType.MULTIPLE_CHOICE: 0.8,
                QuestionType.MULTIPLE_SELECTION: 0.2
            },
            difficulty_distribution={
                Difficulty.EASY: 0.2,
                Difficulty.MEDIUM: 0.5,
                Difficulty.HARD: 0.3
            },
            themes_filter=["Python", "Programming"]
        )

        assert config.num_questions == 50
        assert config.language == Language.EN
        assert config.model == "custom-model"
        assert config.temperature == 0.8
        assert config.themes_filter == ["Python", "Programming"]

    def test_generation_config_validation_errors(self):
        """Test generation config validation errors."""
        # Invalid question type distribution (doesn't sum to 1)
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(
                question_types={
                    QuestionType.MULTIPLE_CHOICE: 0.5,
                    QuestionType.MULTIPLE_SELECTION: 0.3
                }  # Sums to 0.8, not 1.0
            )
        assert "must sum to 1.0" in str(exc_info.value)

        # Invalid difficulty distribution
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(
                difficulty_distribution={
                    Difficulty.EASY: 0.5,
                    Difficulty.MEDIUM: 0.5,
                    Difficulty.HARD: 0.2
                }  # Sums to 1.2, not 1.0
            )
        assert "must sum to 1.0" in str(exc_info.value)

        # Invalid batch sizes (doesn't end with -1)
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(batch_sizes=[1, 5, 10])
        assert "Last batch size must be -1" in str(exc_info.value)

        # Too many questions
        with pytest.raises(ValidationError) as exc_info:
            GenerationConfig(num_questions=500)  # Exceeds MAX_QUESTIONS_PER_SESSION

    def test_generation_session_create(self):
        """Test generation session creation."""
        config = GenerationConfig(num_questions=10, language=Language.FR)
        session_create = GenerationSessionCreate(
            document_id=1,
            config=config
        )

        assert session_create.document_id == 1
        assert session_create.config.num_questions == 10
        assert session_create.config.language == Language.FR


class TestExportSchemas:
    """Test export-related schemas."""

    def test_export_request_with_session_id(self):
        """Test export request with session ID."""
        request = ExportRequest(
            session_id="session_123",
            format=ExportFormat.CSV,
            include_explanations=True
        )

        assert request.session_id == "session_123"
        assert request.format == ExportFormat.CSV
        assert request.include_explanations is True
        assert request.question_ids is None

    def test_export_request_with_question_ids(self):
        """Test export request with specific question IDs."""
        request = ExportRequest(
            question_ids=[1, 2, 3, 4, 5],
            format=ExportFormat.JSON,
            include_metadata=True
        )

        assert request.question_ids == [1, 2, 3, 4, 5]
        assert request.format == ExportFormat.JSON
        assert request.include_metadata is True
        assert request.session_id is None

    def test_export_request_validation_errors(self):
        """Test export request validation errors."""
        # Neither session_id nor question_ids provided
        with pytest.raises(ValidationError) as exc_info:
            ExportRequest(format=ExportFormat.CSV)
        assert "Either session_id or question_ids must be provided" in str(exc_info.value)

        # Both session_id and question_ids provided
        with pytest.raises(ValidationError) as exc_info:
            ExportRequest(
                session_id="session_123",
                question_ids=[1, 2, 3],
                format=ExportFormat.CSV
            )
        assert "Only one of session_id or question_ids should be provided" in str(exc_info.value)


class TestProcessingSchemas:
    """Test processing configuration schemas."""

    def test_processing_config_default(self):
        """Test processing config with defaults."""
        config = ProcessingConfig()

        assert config.chunking_strategy == ChunkingStrategy.SEMANTIC
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2
        assert config.extract_themes is True

    def test_processing_config_custom(self):
        """Test custom processing configuration."""
        config = ProcessingConfig(
            chunking_strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=800,
            chunk_overlap=100,
            embedding_model=EmbeddingModel.OPENAI_TEXT_EMBEDDING_ADA_002,
            extract_themes=False,
            max_themes=15
        )

        assert config.chunking_strategy == ChunkingStrategy.FIXED_SIZE
        assert config.chunk_size == 800
        assert config.chunk_overlap == 100
        assert config.extract_themes is False
        assert config.max_themes == 15

    def test_processing_config_validation_errors(self):
        """Test processing config validation errors."""
        # Chunk overlap >= chunk size
        with pytest.raises(ValidationError) as exc_info:
            ProcessingConfig(chunk_size=500, chunk_overlap=500)
        assert "Chunk overlap must be less than chunk size" in str(exc_info.value)

        # Invalid chunk size
        with pytest.raises(ValidationError) as exc_info:
            ProcessingConfig(chunk_size=50)  # Below minimum


class TestResponseSchemas:
    """Test response schemas."""

    def test_error_response(self):
        """Test error response schema."""
        now = datetime.now()
        response = ErrorResponse(
            error="Validation failed",
            details=[
                {"code": "INVALID_FIELD", "message": "Field is required", "field": "title"}
            ],
            request_id="req_123",
            timestamp=now
        )

        assert response.error == "Validation failed"
        assert len(response.details) == 1
        assert response.details[0]["field"] == "title"
        assert response.request_id == "req_123"
        assert response.timestamp == now

    def test_success_response(self):
        """Test success response schema."""
        response = SuccessResponse(
            message="Operation completed successfully",
            data={"id": 1, "status": "created"}
        )

        assert response.message == "Operation completed successfully"
        assert response.data["id"] == 1
        assert isinstance(response.timestamp, datetime)

    def test_success_response_with_list_data(self):
        """Test success response with list data."""
        response = SuccessResponse(
            message="Items retrieved",
            data=[{"id": 1}, {"id": 2}]
        )

        assert response.message == "Items retrieved"
        assert len(response.data) == 2
        assert response.data[0]["id"] == 1
