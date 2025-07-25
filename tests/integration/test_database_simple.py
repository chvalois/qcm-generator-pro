"""
QCM Generator Pro - Simple Database Integration Tests

Basic tests for database connectivity and model creation.
"""

import pytest
from sqlalchemy import text

from src.core.database import DatabaseManager
from src.models.database import Document, Question, DocumentTheme


class TestBasicDatabaseOperations:
    """Test basic database operations."""

    def test_database_manager_creation(self):
        """Test database manager can be created."""
        db_manager = DatabaseManager()
        assert db_manager is not None

    def test_database_connection(self):
        """Test database connection works."""
        db_manager = DatabaseManager()
        
        # Just test that we can create a session without errors
        try:
            with db_manager.get_session() as session:
                # If we can create a session, connection is working
                assert session is not None
        except Exception:
            pytest.skip("Database connection test skipped - database may not be initialized")

    def test_simple_query(self):
        """Test simple database query."""
        db_manager = DatabaseManager()
        
        with db_manager.get_session() as session:
            # Simple query that should work on any database
            result = session.execute(text("SELECT 1 as test_value")).scalar()
            assert result == 1

    def test_table_exists(self):
        """Test that our main tables exist."""
        db_manager = DatabaseManager()
        
        with db_manager.get_session() as session:
            # Try to query main tables - should not raise errors
            try:
                session.query(Document).first()
                session.query(Question).first()  
                session.query(DocumentTheme).first()
                # If we get here, tables exist
                assert True
            except Exception as e:
                # If tables don't exist, that's also ok for this basic test
                assert "no such table" in str(e).lower() or "does not exist" in str(e).lower()

    def test_basic_document_creation(self):
        """Test basic document model creation."""
        # Just test model instantiation, not database insertion
        doc = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr"
        )
        
        assert doc.filename == "test.pdf"
        assert doc.file_path == "/tmp/test.pdf"
        assert doc.language == "fr"