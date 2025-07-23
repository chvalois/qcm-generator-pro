"""
QCM Generator Pro - Pytest Configuration and Fixtures

This module provides common test configuration, fixtures, and utilities
for the test suite.
"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import Settings
from src.core.database import DatabaseManager
from src.models.database import Base
from src.models.enums import Difficulty, Language, QuestionType, ValidationStatus


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings with temporary database."""
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    test_db_path = Path(temp_dir) / "test_qcm_generator.db"

    # Override settings for testing
    os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Create settings instance
    settings = Settings()
    settings.database.url = f"sqlite:///{test_db_path}"

    return settings


@pytest.fixture(scope="session")
def test_database_url(test_settings: Settings) -> str:
    """Get test database URL."""
    return test_settings.database.url


@pytest.fixture(scope="session")
def test_db_manager(test_database_url: str) -> DatabaseManager:
    """Create database manager for testing."""
    return DatabaseManager(database_url=test_database_url)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database(test_db_manager: DatabaseManager):
    """Set up test database with tables."""
    # Create all tables
    test_db_manager.create_tables()

    yield

    # Clean up after all tests
    test_db_manager.close()


@pytest.fixture
def db_session(test_db_manager: DatabaseManager):
    """Create a database session for testing with cleanup."""
    with test_db_manager.get_session() as session:
        yield session
        # Rollback any uncommitted changes
        session.rollback()


@pytest.fixture
def clean_db(test_db_manager: DatabaseManager):
    """Provide a clean database for each test."""
    # Clear all tables
    with test_db_manager.get_session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()

    yield

    # Clean up after test
    with test_db_manager.get_session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "filename": "test_document.pdf",
        "file_path": "/path/to/test_document.pdf",
        "file_size": 1024 * 1024,  # 1MB
        "file_hash": "abc123def456",
        "title": "Test Document",
        "author": "Test Author",
        "subject": "Computer Science",
        "total_pages": 10,
        "language": Language.FR,
        "processing_status": "completed",
        "doc_metadata": {"key": "value"},
    }


@pytest.fixture
def sample_theme_data():
    """Sample theme data for testing."""
    return {
        "theme_name": "Introduction to Python",
        "description": "Basic Python concepts and syntax",
        "start_page": 1,
        "end_page": 3,
        "confidence_score": 0.85,
        "importance_score": 0.9,
        "keywords": ["python", "programming", "syntax"],
        "concepts": ["variables", "functions", "loops"],
    }


@pytest.fixture
def sample_question_data():
    """Sample question data for testing."""
    return {
        "question_text": "What is the output of print('Hello, World!')?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "language": Language.EN,
        "difficulty": Difficulty.EASY,
        "options": [
            {"text": "Hello, World!", "is_correct": True},
            {"text": "Hello World", "is_correct": False},
            {"text": "print('Hello, World!')", "is_correct": False},
            {"text": "Error", "is_correct": False},
        ],
        "explanation": "The print() function outputs the string exactly as provided.",
        "estimated_time_seconds": 30,
        "session_id": "test_session_001",
        "batch_number": 1,
        "generation_order": 1,
        "validation_status": ValidationStatus.PENDING,
        "tags": ["python", "basic", "print"],
    }


@pytest.fixture
def sample_generation_config():
    """Sample generation configuration for testing."""
    return {
        "num_questions": 10,
        "language": Language.FR,
        "model": "test-model",
        "question_types": {
            QuestionType.MULTIPLE_CHOICE: 0.7,
            QuestionType.MULTIPLE_SELECTION: 0.3,
        },
        "difficulty_distribution": {
            Difficulty.EASY: 0.3,
            Difficulty.MEDIUM: 0.5,
            Difficulty.HARD: 0.2,
        },
        "validation_mode": "progressive",
        "batch_sizes": [1, 5, -1],
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "min_quality_score": 0.6,
        "require_explanation": True,
        "auto_approve_threshold": 0.8,
    }


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_pdf_content():
    """Mock PDF content for testing."""
    return """
    Chapter 1: Introduction to Programming

    Programming is the process of creating instructions for computers to follow.
    Python is a popular programming language known for its simplicity and readability.

    Basic Concepts:
    1. Variables - store data values
    2. Functions - reusable code blocks
    3. Loops - repeat code execution
    4. Conditionals - make decisions in code

    Example:
    def greet(name):
        return f"Hello, {name}!"

    Chapter 2: Data Types

    Python supports several built-in data types:
    - Integers (int)
    - Floating-point numbers (float)
    - Strings (str)
    - Booleans (bool)
    - Lists (list)
    - Dictionaries (dict)
    """


@pytest.fixture
def mock_extracted_themes():
    """Mock extracted themes for testing."""
    return [
        {
            "theme_name": "Introduction to Programming",
            "description": "Basic programming concepts and Python introduction",
            "start_page": 1,
            "end_page": 1,
            "confidence_score": 0.9,
            "keywords": ["programming", "python", "computer", "instructions"],
            "concepts": ["variables", "functions", "loops", "conditionals"],
        },
        {
            "theme_name": "Data Types",
            "description": "Python built-in data types and their usage",
            "start_page": 2,
            "end_page": 2,
            "confidence_score": 0.85,
            "keywords": ["data types", "int", "float", "string", "boolean"],
            "concepts": ["integers", "strings", "lists", "dictionaries"],
        },
    ]


@pytest.fixture
def mock_document_chunks():
    """Mock document chunks for testing."""
    return [
        {
            "chunk_text": "Programming is the process of creating instructions for computers to follow.",
            "chunk_order": 1,
            "start_page": 1,
            "end_page": 1,
            "word_count": 12,
            "char_count": 75,
        },
        {
            "chunk_text": "Python is a popular programming language known for its simplicity.",
            "chunk_order": 2,
            "start_page": 1,
            "end_page": 1,
            "word_count": 11,
            "char_count": 66,
        },
        {
            "chunk_text": "Python supports several built-in data types including integers, strings, and lists.",
            "chunk_order": 3,
            "start_page": 2,
            "end_page": 2,
            "word_count": 13,
            "char_count": 83,
        },
    ]


# ============================================================================
# Utility Functions for Tests
# ============================================================================

def create_test_document(session, **kwargs):
    """Create a test document in the database."""
    from src.models.database import Document

    document_data = {
        "filename": "test.pdf",
        "file_path": "/tmp/test.pdf",
        "language": "fr",
        "processing_status": "completed",
    }
    document_data.update(kwargs)

    document = Document(**document_data)
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def create_test_theme(session, document_id, **kwargs):
    """Create a test theme in the database."""
    from src.models.database import DocumentTheme

    theme_data = {
        "document_id": document_id,
        "theme_name": "Test Theme",
        "confidence_score": 0.8,
    }
    theme_data.update(kwargs)

    theme = DocumentTheme(**theme_data)
    session.add(theme)
    session.commit()
    session.refresh(theme)
    return theme


def create_test_question(session, document_id, **kwargs):
    """Create a test question in the database."""
    from src.models.database import Question

    question_data = {
        "document_id": document_id,
        "session_id": "test_session",
        "question_text": "What is Python?",
        "question_type": "multiple-choice",
        "language": "en",
        "difficulty": "easy",
        "options": [{"text": "A snake", "is_correct": False}, {"text": "A programming language", "is_correct": True}],
        "correct_answers": [1],
        "generation_order": 1,
        "validation_status": "pending",
    }
    question_data.update(kwargs)

    question = Question(**question_data)
    session.add(question)
    session.commit()
    session.refresh(question)
    return question


# ============================================================================
# Test Markers and Configuration
# ============================================================================

# Custom markers for test organization
pytest_plugins = []

# Mark tests that require database
requires_database = pytest.mark.database

# Mark tests that are slow
slow = pytest.mark.slow

# Mark integration tests
integration = pytest.mark.integration

# Mark tests that require external services
requires_external = pytest.mark.requires_external


# ============================================================================
# Test Environment Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Performance and Memory Testing Utilities
# ============================================================================

@pytest.fixture
def memory_profiler():
    """Simple memory usage profiler for tests."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss

    yield

    end_memory = process.memory_info().rss
    memory_diff = end_memory - start_memory

    # Warn if memory usage increased significantly (>10MB)
    if memory_diff > 10 * 1024 * 1024:
        print(f"Warning: Test increased memory usage by {memory_diff / 1024 / 1024:.2f} MB")


# ============================================================================
# Temporary File Management
# ============================================================================

@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Write minimal PDF content
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello, World!) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000053 00000 n
0000000125 00000 n
0000000185 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
279
%%EOF"""
        f.write(pdf_content)
        temp_path = f.name

    yield temp_path

    # Clean up
    if os.path.exists(temp_path):
        os.unlink(temp_path)
