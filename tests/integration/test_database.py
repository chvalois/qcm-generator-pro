"""
QCM Generator Pro - Database Integration Tests

Tests for database operations, transactions, and data integrity.
"""

from datetime import datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from src.core.database import (
    DatabaseManager,
    check_database_health,
    init_database,
    reset_database,
)
from src.models.database import (
    Document,
    DocumentChunk,
    DocumentTheme,
    GenerationSession,
    Question,
)


class TestDatabaseConnection:
    """Test database connection and basic operations."""

    def test_database_manager_initialization(self, test_database_url):
        """Test database manager initialization."""
        db_manager = DatabaseManager(test_database_url)

        assert db_manager.database_url == test_database_url
        assert db_manager._engine is None  # Lazy initialization

    def test_engine_creation(self, test_db_manager):
        """Test database engine creation."""
        engine = test_db_manager.engine

        assert engine is not None
        assert engine.url.database is not None

    def test_session_creation(self, test_db_manager):
        """Test database session creation."""
        session_maker = test_db_manager.session_maker

        assert session_maker is not None

        with test_db_manager.get_session() as session:
            assert session is not None
            # Test basic query
            result = session.execute("SELECT 1").scalar()
            assert result == 1

    def test_connection_test(self, test_db_manager):
        """Test database connection health check."""
        is_healthy = test_db_manager.test_connection()
        assert is_healthy is True

    def test_table_creation(self, test_db_manager):
        """Test database table creation."""
        # Tables should already be created by setup, but test the method
        test_db_manager.create_tables()

        # Verify tables exist by querying them
        with test_db_manager.get_session() as session:
            # Try to query each main table
            session.query(Document).first()  # Should not raise error
            session.query(DocumentTheme).first()
            session.query(DocumentChunk).first()
            session.query(Question).first()
            session.query(GenerationSession).first()


class TestDatabaseOperations:
    """Test CRUD operations and data integrity."""

    def test_document_crud_operations(self, db_session, clean_db):
        """Test document CRUD operations."""
        # Create
        document = Document(
            filename="test_crud.pdf",
            file_path="/tmp/test_crud.pdf",
            file_size=2048,
            language="fr",
            title="CRUD Test Document",
            processing_status="pending"
        )
        db_session.add(document)
        db_session.commit()

        document_id = document.id
        assert document_id is not None

        # Read
        retrieved = db_session.query(Document).filter(Document.id == document_id).first()
        assert retrieved is not None
        assert retrieved.filename == "test_crud.pdf"
        assert retrieved.title == "CRUD Test Document"

        # Update
        retrieved.title = "Updated CRUD Test Document"
        retrieved.processing_status = "completed"
        db_session.commit()

        updated = db_session.query(Document).filter(Document.id == document_id).first()
        assert updated.title == "Updated CRUD Test Document"
        assert updated.processing_status == "completed"

        # Delete
        db_session.delete(updated)
        db_session.commit()

        deleted = db_session.query(Document).filter(Document.id == document_id).first()
        assert deleted is None

    def test_foreign_key_constraints(self, db_session, clean_db):
        """Test foreign key constraint enforcement."""
        # Try to create a theme without a document (should fail)
        theme = DocumentTheme(
            document_id=999,  # Non-existent document
            theme_name="Orphan Theme",
            confidence_score=0.5
        )
        db_session.add(theme)

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

    def test_cascade_deletion(self, db_session, clean_db):
        """Test cascade deletion of related records."""
        # Create document with related records
        document = Document(
            filename="cascade_test.pdf",
            file_path="/tmp/cascade_test.pdf",
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
            session_id="cascade_test",
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

        document_id = document.id
        theme_id = theme.id
        chunk_id = chunk.id
        question_id = question.id

        # Delete document (should cascade)
        db_session.delete(document)
        db_session.commit()

        # Verify related records were deleted
        assert db_session.query(Document).filter(Document.id == document_id).first() is None
        assert db_session.query(DocumentTheme).filter(DocumentTheme.id == theme_id).first() is None
        assert db_session.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first() is None
        assert db_session.query(Question).filter(Question.id == question_id).first() is None

    def test_unique_constraints(self, db_session, clean_db):
        """Test unique constraint enforcement where applicable."""
        # Create generation session
        session1 = GenerationSession(
            session_id="unique_test_001",
            document_id=1,  # Will fail due to FK, but testing unique constraint
            total_questions_requested=10,
            language="fr",
            model_used="test-model",
            generation_config={},
            batch_sizes=[1, -1]
        )
        db_session.add(session1)

        # Due to FK constraint, this will fail, but let's test with a valid document
        document = Document(
            filename="unique_test.pdf",
            file_path="/tmp/unique_test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()

        session1.document_id = document.id
        db_session.commit()

        # Try to create another session with the same session_id
        session2 = GenerationSession(
            session_id="unique_test_001",  # Duplicate session_id
            document_id=document.id,
            total_questions_requested=5,
            language="en",
            model_used="other-model",
            generation_config={},
            batch_sizes=[1, -1]
        )
        db_session.add(session2)

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()


class TestDatabaseTransactions:
    """Test transaction handling and rollback scenarios."""

    def test_successful_transaction(self, db_session, clean_db):
        """Test successful transaction with multiple operations."""
        document = Document(
            filename="transaction_test.pdf",
            file_path="/tmp/transaction_test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.flush()  # Get ID without committing

        theme = DocumentTheme(
            document_id=document.id,
            theme_name="Transaction Theme",
            confidence_score=0.7
        )
        db_session.add(theme)

        question = Question(
            document_id=document.id,
            session_id="transaction_test",
            question_text="Transaction question?",
            question_type="multiple-choice",
            language="fr",
            difficulty="medium",
            options=[{"text": "Yes", "is_correct": True}, {"text": "No", "is_correct": False}],
            correct_answers=[0],
            generation_order=1,
            validation_status="pending"
        )
        db_session.add(question)

        # Commit all together
        db_session.commit()

        # Verify all records exist
        assert db_session.query(Document).filter(Document.filename == "transaction_test.pdf").first() is not None
        assert db_session.query(DocumentTheme).filter(DocumentTheme.theme_name == "Transaction Theme").first() is not None
        assert db_session.query(Question).filter(Question.question_text == "Transaction question?").first() is not None

    def test_transaction_rollback(self, db_session, clean_db):
        """Test transaction rollback on error."""
        document = Document(
            filename="rollback_test.pdf",
            file_path="/tmp/rollback_test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.flush()

        # Add valid theme
        theme = DocumentTheme(
            document_id=document.id,
            theme_name="Valid Theme",
            confidence_score=0.8
        )
        db_session.add(theme)

        # Add invalid question (intentionally cause error)
        invalid_question = Question(
            document_id=999999,  # Invalid document_id
            session_id="rollback_test",
            question_text="This should fail",
            question_type="multiple-choice",
            language="fr",
            difficulty="easy",
            options=[{"text": "Answer", "is_correct": True}],
            correct_answers=[0],
            generation_order=1,
            validation_status="pending"
        )
        db_session.add(invalid_question)

        # This should fail and rollback the entire transaction
        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

        # Verify nothing was committed
        assert db_session.query(Document).filter(Document.filename == "rollback_test.pdf").first() is None
        assert db_session.query(DocumentTheme).filter(DocumentTheme.theme_name == "Valid Theme").first() is None


class TestDatabasePerformance:
    """Test database performance and optimization."""

    def test_bulk_insert_performance(self, db_session, clean_db):
        """Test bulk insert operations."""
        import time

        # Create document first
        document = Document(
            filename="bulk_test.pdf",
            file_path="/tmp/bulk_test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.commit()

        # Bulk insert chunks
        start_time = time.time()
        chunks = []
        for i in range(100):
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_text=f"This is test chunk number {i} with some content.",
                chunk_order=i + 1,
                char_count=45 + len(str(i))
            )
            chunks.append(chunk)

        db_session.bulk_save_objects(chunks)
        db_session.commit()
        bulk_time = time.time() - start_time

        # Verify all chunks were inserted
        chunk_count = db_session.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).count()
        assert chunk_count == 100

        # Performance should be reasonable (less than 5 seconds for 100 records)
        assert bulk_time < 5.0

    def test_query_performance_with_relationships(self, db_session, clean_db):
        """Test query performance with eager loading."""
        import time

        # Create test data
        document = Document(
            filename="performance_test.pdf",
            file_path="/tmp/performance_test.pdf",
            language="fr"
        )
        db_session.add(document)
        db_session.flush()

        # Create multiple themes and questions
        for i in range(10):
            theme = DocumentTheme(
                document_id=document.id,
                theme_name=f"Theme {i}",
                confidence_score=0.8
            )
            db_session.add(theme)

            for j in range(5):
                question = Question(
                    document_id=document.id,
                    theme_id=theme.id if i < 5 else None,  # Some questions without themes
                    session_id=f"perf_session_{i}",
                    question_text=f"Question {i}-{j}?",
                    question_type="multiple-choice",
                    language="fr",
                    difficulty="medium",
                    options=[
                        {"text": f"Option A-{j}", "is_correct": True},
                        {"text": f"Option B-{j}", "is_correct": False}
                    ],
                    correct_answers=[0],
                    generation_order=j + 1,
                    validation_status="pending"
                )
                db_session.add(question)

        db_session.commit()

        # Test query performance with eager loading
        start_time = time.time()
        from sqlalchemy.orm import joinedload

        documents_with_relations = db_session.query(Document).options(
            joinedload(Document.themes),
            joinedload(Document.questions)
        ).filter(Document.id == document.id).all()

        query_time = time.time() - start_time

        # Verify data was loaded
        doc = documents_with_relations[0]
        assert len(doc.themes) == 10
        assert len(doc.questions) == 50

        # Query should be reasonable fast (less than 1 second)
        assert query_time < 1.0


class TestDatabaseHealthCheck:
    """Test database health monitoring."""

    def test_health_check_function(self):
        """Test database health check function."""
        health_status = check_database_health()

        assert "status" in health_status
        assert health_status["status"] in ["healthy", "unhealthy", "error"]

        if health_status["status"] == "healthy":
            assert "database_url" in health_status

    def test_database_utility_functions(self, test_database_url):
        """Test database utility functions."""
        # Test initialization
        db_manager = DatabaseManager(test_database_url)

        # Test reset functionality
        db_manager.reset_database()

        # Verify tables were recreated
        with db_manager.get_session() as session:
            # Should not raise errors
            session.query(Document).first()
            session.query(Question).first()

        db_manager.close()


class TestDatabaseMigrationScenarios:
    """Test scenarios that might occur during database migrations."""

    def test_handling_existing_data_during_reset(self, db_session, clean_db):
        """Test database reset with existing data."""
        # Create some test data
        document = Document(
            filename="migration_test.pdf",
            file_path="/tmp/migration_test.pdf",
            language="fr"
        )
        db_session.add(document)

        theme = DocumentTheme(
            document_id=document.id,
            theme_name="Migration Theme",
            confidence_score=0.9
        )
        db_session.add(theme)
        db_session.commit()

        # Verify data exists
        doc_count_before = db_session.query(Document).count()
        theme_count_before = db_session.query(DocumentTheme).count()
        assert doc_count_before > 0
        assert theme_count_before > 0

        # Reset database using clean_db fixture (which clears data)
        # This is handled by the fixture, so we just verify it worked
        doc_count_after = db_session.query(Document).count()
        theme_count_after = db_session.query(DocumentTheme).count()

        # Data should be cleared by clean_db fixture
        assert doc_count_after == 0
        assert theme_count_after == 0
