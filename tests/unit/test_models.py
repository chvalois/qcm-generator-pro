"""
QCM Generator Pro - Models Unit Tests

Tests for SQLAlchemy database models and their relationships.
"""

import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from src.models.database import Document, DocumentTheme, DocumentChunk, Question, GenerationSession
from src.models.enums import ProcessingStatus, QuestionType, ValidationStatus, Language, Difficulty


class TestDocument:
    """Test Document model."""
    
    def test_create_document(self, db_session, sample_document_data):
        """Test creating a document."""
        document = Document(**sample_document_data)
        db_session.add(document)
        db_session.commit()
        
        assert document.id is not None
        assert document.filename == sample_document_data["filename"]
        assert document.language == sample_document_data["language"]
        assert document.processing_status == sample_document_data["processing_status"]
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.updated_at, datetime)
    
    def test_document_relationships(self, db_session, clean_db):
        """Test document relationships with themes, chunks, and questions."""
        # Create document
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create theme
        theme = DocumentTheme(
            document_id=document.id,
            theme_name="Test Theme",
            confidence_score=0.8
        )
        db_session.add(theme)
        
        # Create chunk
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_text="Test chunk content",
            chunk_order=1,
            char_count=18
        )
        db_session.add(chunk)
        
        # Create question
        question = Question(
            document_id=document.id,
            session_id="test_session",
            question_text="Test question?",
            question_type="multiple-choice",
            language="fr",
            difficulty="easy",
            options=[{"text": "Option 1", "is_correct": True}],
            correct_answers=[0],
            generation_order=1,
            validation_status="pending"
        )
        db_session.add(question)
        db_session.commit()
        
        # Test relationships
        db_session.refresh(document)
        assert len(document.themes) == 1
        assert len(document.chunks) == 1  
        assert len(document.questions) == 1
        assert document.themes[0].theme_name == "Test Theme"
        assert document.chunks[0].chunk_text == "Test chunk content"
        assert document.questions[0].question_text == "Test question?"
    
    def test_document_required_fields(self, db_session):
        """Test that required fields are enforced."""
        with pytest.raises(Exception):  # Should raise integrity error
            document = Document()  # Missing required filename and file_path
            db_session.add(document)
            db_session.commit()
    
    def test_document_repr(self, db_session, clean_db):
        """Test document string representation."""
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr",
            processing_status="completed"
        )
        db_session.add(document)
        db_session.commit()
        
        repr_str = repr(document)
        assert "Document" in repr_str
        assert "test.pdf" in repr_str
        assert "completed" in repr_str


class TestDocumentTheme:
    """Test DocumentTheme model."""
    
    def test_create_theme(self, db_session, clean_db):
        """Test creating a theme."""
        # Create document first
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create theme
        theme = DocumentTheme(
            document_id=document.id,
            theme_name="Python Basics",
            description="Introduction to Python programming",
            start_page=1,
            end_page=5,
            confidence_score=0.85,
            keywords=["python", "programming", "basics"],
            concepts=["variables", "functions"]
        )
        db_session.add(theme)
        db_session.commit()
        
        assert theme.id is not None
        assert theme.document_id == document.id
        assert theme.theme_name == "Python Basics"
        assert theme.confidence_score == 0.85
        assert "python" in theme.keywords
        assert "variables" in theme.concepts
    
    def test_theme_document_relationship(self, db_session, clean_db):
        """Test theme-document relationship."""
        # Create document
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create theme
        theme = DocumentTheme(
            document_id=document.id,
            theme_name="Test Theme",
            confidence_score=0.7
        )
        db_session.add(theme)
        db_session.commit()
        
        # Test relationship
        assert theme.document.filename == "test.pdf"
        assert document.themes[0].theme_name == "Test Theme"
    
    def test_theme_repr(self, db_session, clean_db):
        """Test theme string representation."""
        document = Document(filename="test.pdf", file_path="/tmp/test.pdf", language="fr")
        db_session.add(document)
        db_session.commit()
        
        theme = DocumentTheme(
            document_id=document.id,
            theme_name="Test Theme",
            confidence_score=0.8
        )
        db_session.add(theme)
        db_session.commit()
        
        repr_str = repr(theme)
        assert "DocumentTheme" in repr_str
        assert "Test Theme" in repr_str
        assert "0.8" in repr_str


class TestDocumentChunk:
    """Test DocumentChunk model."""
    
    def test_create_chunk(self, db_session, clean_db):
        """Test creating a document chunk."""
        # Create document first
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create chunk
        chunk_text = "This is a test chunk of document content."
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_text=chunk_text,
            chunk_order=1,
            start_page=1,
            end_page=1,
            word_count=9,
            char_count=len(chunk_text),
            vector_id="test_vector_123"
        )
        db_session.add(chunk)
        db_session.commit()
        
        assert chunk.id is not None
        assert chunk.document_id == document.id
        assert chunk.chunk_text == chunk_text
        assert chunk.chunk_order == 1
        assert chunk.char_count == len(chunk_text)
        assert chunk.vector_id == "test_vector_123"
    
    def test_chunk_document_relationship(self, db_session, clean_db):
        """Test chunk-document relationship."""
        # Create document
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create chunk
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_text="Test content",
            chunk_order=1,
            char_count=12
        )
        db_session.add(chunk)
        db_session.commit()
        
        # Test relationship
        assert chunk.document.filename == "test.pdf"
        assert document.chunks[0].chunk_text == "Test content"


class TestQuestion:
    """Test Question model."""
    
    def test_create_question(self, db_session, clean_db):
        """Test creating a question."""
        # Create document first
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create question
        options = [
            {"text": "Option 1", "is_correct": True},
            {"text": "Option 2", "is_correct": False},
            {"text": "Option 3", "is_correct": False}
        ]
        
        question = Question(
            document_id=document.id,
            session_id="test_session_001",
            question_text="What is the correct answer?",
            question_type="multiple-choice",
            language="en",
            difficulty="medium",
            options=options,
            correct_answers=[0],
            explanation="Option 1 is correct because...",
            generation_order=1,
            validation_status="approved",
            model_used="test-model"
        )
        db_session.add(question)
        db_session.commit()
        
        assert question.id is not None
        assert question.document_id == document.id
        assert question.question_type == "multiple-choice"
        assert len(question.options) == 3
        assert question.options[0]["is_correct"] is True
        assert question.correct_answers == [0]
        assert question.validation_status == "approved"
    
    def test_question_relationships(self, db_session, clean_db):
        """Test question relationships with document and theme."""
        # Create document
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create theme
        theme = DocumentTheme(
            document_id=document.id,
            theme_name="Test Theme",
            confidence_score=0.8
        )
        db_session.add(theme)
        db_session.commit()
        
        # Create question
        question = Question(
            document_id=document.id,
            theme_id=theme.id,
            session_id="test_session",
            question_text="Test question?",
            question_type="multiple-choice",
            language="fr",
            difficulty="easy",
            options=[{"text": "Answer", "is_correct": True}],
            correct_answers=[0],
            generation_order=1,
            validation_status="pending"
        )
        db_session.add(question)
        db_session.commit()
        
        # Test relationships
        assert question.document.filename == "test.pdf"
        assert question.theme_obj.theme_name == "Test Theme"
        assert document.questions[0].question_text == "Test question?"
        assert theme.questions[0].question_text == "Test question?"
    
    def test_question_validation_statuses(self, db_session, clean_db):
        """Test different validation statuses."""
        document = Document(filename="test.pdf", file_path="/tmp/test.pdf", language="fr")
        db_session.add(document)
        db_session.commit()
        
        statuses = ["pending", "approved", "rejected", "needs_review"]
        
        for i, status in enumerate(statuses):
            question = Question(
                document_id=document.id,
                session_id=f"session_{i}",
                question_text=f"Question {i}?",
                question_type="multiple-choice",
                language="fr",
                difficulty="easy",
                options=[{"text": "Answer", "is_correct": True}],
                correct_answers=[0],
                generation_order=i + 1,
                validation_status=status
            )
            db_session.add(question)
        
        db_session.commit()
        
        # Verify all questions were created with correct statuses
        questions = db_session.query(Question).all()
        assert len(questions) == 4
        for i, question in enumerate(questions):
            assert question.validation_status == statuses[i]


class TestGenerationSession:
    """Test GenerationSession model."""
    
    def test_create_session(self, db_session, clean_db):
        """Test creating a generation session."""
        # Create document first
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create session
        config = {
            "num_questions": 10,
            "language": "fr",
            "difficulty": "mixed",
            "question_types": {"multiple-choice": 0.7, "multiple-selection": 0.3}
        }
        
        session = GenerationSession(
            session_id="session_test_001",
            document_id=document.id,
            total_questions_requested=10,
            language="fr",
            model_used="test-model",
            generation_config=config,
            batch_sizes=[1, 5, -1],
            status="running"
        )
        db_session.add(session)
        db_session.commit()
        
        assert session.id is not None
        assert session.session_id == "session_test_001"
        assert session.document_id == document.id
        assert session.total_questions_requested == 10
        assert session.generation_config["num_questions"] == 10
        assert session.batch_sizes == [1, 5, -1]
        assert session.status == "running"
    
    def test_session_progress_tracking(self, db_session, clean_db):
        """Test session progress tracking fields."""
        document = Document(filename="test.pdf", file_path="/tmp/test.pdf", language="fr")
        db_session.add(document)
        db_session.commit()
        
        session = GenerationSession(
            session_id="progress_test",
            document_id=document.id,
            total_questions_requested=20,
            language="fr",
            model_used="test-model",
            generation_config={"num_questions": 20},
            batch_sizes=[1, 5, -1],
            current_batch=2,
            batches_completed=1,
            questions_generated=6,
            questions_validated=5,
            questions_approved=4,
            questions_rejected=1,
            status="running"
        )
        db_session.add(session)
        db_session.commit()
        
        assert session.current_batch == 2
        assert session.batches_completed == 1
        assert session.questions_generated == 6
        assert session.questions_validated == 5
        assert session.questions_approved == 4
        assert session.questions_rejected == 1
    
    def test_session_document_relationship(self, db_session, clean_db):
        """Test session-document relationship."""
        document = Document(filename="test.pdf", file_path="/tmp/test.pdf", language="fr")
        db_session.add(document)
        db_session.commit()
        
        session = GenerationSession(
            session_id="relationship_test",
            document_id=document.id,
            total_questions_requested=5,
            language="fr",
            model_used="test-model",
            generation_config={},
            batch_sizes=[1, -1],
            status="pending"
        )
        db_session.add(session)
        db_session.commit()
        
        # Test relationship
        assert session.document.filename == "test.pdf"