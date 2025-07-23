#!/usr/bin/env python3
"""
QCM Generator Pro - Setup Verification Script

This script performs comprehensive testing of the application setup,
verifying that all components are working correctly.

Usage:
    python scripts/test_setup.py [--verbose] [--quick]

Options:
    --verbose    Show detailed output
    --quick      Run only quick tests (skip database operations)
"""

import sys
import os
import argparse
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


class TestResult:
    """Container for test results."""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[Tuple[str, str]] = []

    def add_pass(self, test_name: str):
        self.total += 1
        self.passed += 1

    def add_fail(self, test_name: str, error: str):
        self.total += 1
        self.failed += 1
        self.failures.append((test_name, error))

    def add_skip(self, test_name: str):
        self.total += 1
        self.skipped += 1

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100


class SetupTester:
    """Main test runner for setup verification."""

    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.results = TestResult()

    def print_header(self, text: str):
        """Print a formatted header."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

    def print_success(self, text: str):
        """Print success message."""
        print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

    def print_failure(self, text: str):
        """Print failure message."""
        print(f"{Colors.RED}✗ {text}{Colors.RESET}")

    def print_warning(self, text: str):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

    def print_info(self, text: str):
        """Print info message."""
        if self.verbose:
            print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")

    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test and handle results."""
        try:
            self.print_info(f"Running: {test_name}")
            test_func(*args, **kwargs)
            self.print_success(test_name)
            self.results.add_pass(test_name)
        except Exception as e:
            error_msg = str(e)
            if self.verbose:
                error_msg = traceback.format_exc()
            self.print_failure(f"{test_name}: {error_msg}")
            self.results.add_fail(test_name, error_msg)

    def test_imports(self):
        """Test that all core modules can be imported."""
        self.print_header("Testing Core Module Imports")

        imports_to_test = [
            ("Core Configuration", "from src.core.config import Settings"),
            ("Core Constants", "from src.core.constants import APP_NAME"),
            ("Core Exceptions", "from src.core.exceptions import QCMGeneratorException"),
            ("Database Models", "from src.models.database import Document, Question"),
            ("Pydantic Schemas", "from src.models.schemas import DocumentCreate, QuestionCreate"),
            ("Enums", "from src.models.enums import Language, QuestionType"),
            ("Database Manager", "from src.core.database import DatabaseManager"),
        ]

        for test_name, import_statement in imports_to_test:
            self.run_test(test_name, self._test_import, import_statement)

    def _test_import(self, import_statement: str):
        """Execute an import statement."""
        exec(import_statement)

    def test_configuration(self):
        """Test configuration system."""
        self.print_header("Testing Configuration System")

        self.run_test("Load Default Settings", self._test_default_settings)
        self.run_test("Environment Variable Override", self._test_env_override)
        self.run_test("Nested Settings", self._test_nested_settings)
        self.run_test("Settings Properties", self._test_settings_properties)
        self.run_test("Directory Creation", self._test_directory_creation)

    def _test_default_settings(self):
        """Test loading default settings."""
        from src.core.config import Settings

        settings = Settings()
        assert settings.app_name == "QCM Generator Pro"
        assert settings.port >= 1024
        assert len(settings.supported_languages) >= 2

    def _test_env_override(self):
        """Test environment variable override."""
        import os
        from src.core.config import Settings

        # Set temporary environment variable
        original_value = os.environ.get('APP_NAME')
        os.environ['APP_NAME'] = 'Test Override'

        try:
            settings = Settings()
            assert settings.app_name == 'Test Override'
        finally:
            # Restore original value
            if original_value:
                os.environ['APP_NAME'] = original_value
            else:
                os.environ.pop('APP_NAME', None)

    def _test_nested_settings(self):
        """Test nested settings configuration."""
        from src.core.config import Settings

        settings = Settings()
        assert hasattr(settings, 'database')
        assert hasattr(settings, 'llm')
        assert hasattr(settings, 'generation')
        assert settings.database.pool_size > 0
        assert settings.llm.default_temperature >= 0.0

    def _test_settings_properties(self):
        """Test settings computed properties."""
        from src.core.config import Settings
        from src.models.enums import Environment

        settings = Settings(environment=Environment.DEVELOPMENT)
        assert settings.is_development is True
        assert settings.is_production is False

        # Test paths
        assert settings.base_dir.exists()
        assert settings.data_dir.name == "data"

    def _test_directory_creation(self):
        """Test directory creation."""
        from src.core.config import Settings

        settings = Settings()
        settings.ensure_directories()

        # Check that data directories exist
        assert settings.data_dir.exists()
        assert (settings.data_dir / "pdfs").exists()
        assert (settings.data_dir / "database").exists()

    def test_models_and_schemas(self):
        """Test data models and schemas."""
        self.print_header("Testing Models and Schemas")

        self.run_test("Enum Validation", self._test_enums)
        self.run_test("Database Model Creation", self._test_model_creation)
        self.run_test("Pydantic Schema Validation", self._test_schema_validation)
        self.run_test("Schema Serialization", self._test_schema_serialization)

    def _test_enums(self):
        """Test enum definitions."""
        from src.models.enums import Language, QuestionType, Difficulty

        # Test enum values
        assert Language.FR == "fr"
        assert Language.EN == "en"
        assert QuestionType.MULTIPLE_CHOICE == "multiple-choice"
        assert Difficulty.EASY == "easy"

        # Test enum iteration
        languages = list(Language)
        assert len(languages) >= 4  # FR, EN, ES, DE minimum

    def _test_model_creation(self):
        """Test database model creation."""
        from src.models.database import Document, Question
        from datetime import datetime

        # Test document creation
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            language="fr",
            upload_date=datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        assert document.filename == "test.pdf"
        assert document.language == "fr"

        # Test question creation
        question = Question(
            document_id=1,
            session_id="test",
            question_text="Test question?",
            question_type="multiple-choice",
            language="fr",
            difficulty="easy",
            options=[{"text": "Answer", "is_correct": True}],
            correct_answers=[0],
            generation_order=1,
            validation_status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        assert question.question_text == "Test question?"
        assert len(question.options) == 1

    def _test_schema_validation(self):
        """Test Pydantic schema validation."""
        from src.models.schemas import DocumentCreate, QuestionCreate, QuestionOption, GenerationConfig
        from src.models.enums import Language, QuestionType, Difficulty

        # Test document schema
        doc_data = {
            "filename": "test.pdf",
            "file_path": "/tmp/test.pdf",
            "language": Language.FR
        }
        document = DocumentCreate(**doc_data)
        assert document.filename == "test.pdf"

        # Test question schema with validation
        options = [
            QuestionOption(text="Option 1", is_correct=True),
            QuestionOption(text="Option 2", is_correct=False),
            QuestionOption(text="Option 3", is_correct=False),
        ]

        question = QuestionCreate(
            document_id=1,
            session_id="test",
            question_text="What is the answer?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            language=Language.EN,
            difficulty=Difficulty.MEDIUM,
            options=options,
            generation_order=1
        )
        assert question.question_type == QuestionType.MULTIPLE_CHOICE
        assert len(question.options) == 3

        # Test generation config
        config = GenerationConfig(num_questions=10)
        assert config.num_questions == 10
        assert sum(config.question_types.values()) == 1.0

    def _test_schema_serialization(self):
        """Test schema serialization."""
        from src.models.schemas import SuccessResponse, ErrorResponse
        from datetime import datetime

        # Test success response
        success = SuccessResponse(
            message="Test successful",
            data={"key": "value"}
        )
        success_dict = success.dict()
        assert success_dict["message"] == "Test successful"
        assert "timestamp" in success_dict

        # Test error response
        error = ErrorResponse(
            error="Test error",
            details=[{"code": "TEST", "message": "Test error"}]
        )
        error_dict = error.dict()
        assert error_dict["error"] == "Test error"
        assert len(error_dict["details"]) == 1

    def test_database_connection(self):
        """Test database connection and operations."""
        if self.quick:
            self.results.add_skip("Database Connection (Quick Mode)")
            return

        self.print_header("Testing Database Connection")

        self.run_test("Database Manager Creation", self._test_db_manager)
        self.run_test("Database Connection", self._test_db_connection)
        self.run_test("Table Creation", self._test_table_creation)
        self.run_test("Basic CRUD Operations", self._test_crud_operations)
        self.run_test("Database Health Check", self._test_db_health)

    def _test_db_manager(self):
        """Test database manager creation."""
        from src.core.database import DatabaseManager

        db_manager = DatabaseManager()
        assert db_manager.database_url is not None
        assert "sqlite" in db_manager.database_url.lower()

    def _test_db_connection(self):
        """Test database connection."""
        from src.core.database import DatabaseManager

        db_manager = DatabaseManager()
        is_connected = db_manager.test_connection()
        assert is_connected is True

    def _test_table_creation(self):
        """Test database table creation."""
        from src.core.database import DatabaseManager
        from src.models.database import Base

        db_manager = DatabaseManager()
        db_manager.create_tables()

        # Verify tables exist by attempting to query them
        with db_manager.get_session() as session:
            from src.models.database import Document, Question
            session.query(Document).first()  # Should not raise error
            session.query(Question).first()

    def _test_crud_operations(self):
        """Test basic CRUD operations."""
        from src.core.database import DatabaseManager
        from src.models.database import Document
        from datetime import datetime

        db_manager = DatabaseManager()

        with db_manager.get_session() as session:
            # Create
            document = Document(
                filename="crud_test.pdf",
                file_path="/tmp/crud_test.pdf",
                language="fr",
                processing_status="pending",
                upload_date=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            session.add(document)
            session.commit()

            document_id = document.id
            assert document_id is not None

            # Read
            retrieved = session.query(Document).filter(Document.id == document_id).first()
            assert retrieved is not None
            assert retrieved.filename == "crud_test.pdf"

            # Update
            retrieved.processing_status = "completed"
            session.commit()

            updated = session.query(Document).filter(Document.id == document_id).first()
            assert updated.processing_status == "completed"

            # Delete
            session.delete(updated)
            session.commit()

            deleted = session.query(Document).filter(Document.id == document_id).first()
            assert deleted is None

    def _test_db_health(self):
        """Test database health check."""
        from src.core.database import check_database_health

        health = check_database_health()
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy", "error"]

    def test_constants_and_exceptions(self):
        """Test constants and exception system."""
        self.print_header("Testing Constants and Exceptions")

        self.run_test("Constants Access", self._test_constants)
        self.run_test("Exception Creation", self._test_exceptions)
        self.run_test("Exception Hierarchy", self._test_exception_hierarchy)

    def _test_constants(self):
        """Test constants access."""
        from src.core.constants import APP_NAME, DEFAULT_CHUNK_SIZE, MAX_QUESTIONS_PER_SESSION

        assert APP_NAME == "QCM Generator Pro"
        assert DEFAULT_CHUNK_SIZE > 0
        assert MAX_QUESTIONS_PER_SESSION > 0

    def _test_exceptions(self):
        """Test custom exception creation."""
        from src.core.exceptions import QCMGeneratorException, ValidationError, DatabaseError

        # Test base exception
        base_exc = QCMGeneratorException("Test error", error_code="TEST_ERROR")
        assert base_exc.message == "Test error"
        assert base_exc.error_code == "TEST_ERROR"

        # Test exception serialization
        exc_dict = base_exc.to_dict()
        assert exc_dict["message"] == "Test error"
        assert exc_dict["error_code"] == "TEST_ERROR"

    def _test_exception_hierarchy(self):
        """Test exception inheritance hierarchy."""
        from src.core.exceptions import (
            QCMGeneratorException,
            ValidationError,
            DatabaseError,
            FileError
        )

        # Test inheritance
        validation_error = ValidationError("Validation failed", field="test_field")
        assert isinstance(validation_error, QCMGeneratorException)
        assert validation_error.field == "test_field"

        db_error = DatabaseError("DB error")
        assert isinstance(db_error, QCMGeneratorException)

        file_error = FileError("File error")
        assert isinstance(file_error, QCMGeneratorException)

    def run_all_tests(self):
        """Run all tests and print summary."""
        print(f"{Colors.BOLD}{Colors.MAGENTA}")
        print("QCM Generator Pro - Setup Verification")
        print("=" * 60)
        print(f"{Colors.RESET}")

        # Run test suites
        self.test_imports()
        self.test_configuration()
        self.test_models_and_schemas()
        self.test_database_connection()
        self.test_constants_and_exceptions()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test results summary."""
        self.print_header("Test Results Summary")

        print(f"Total Tests: {self.results.total}")
        print(f"{Colors.GREEN}Passed: {self.results.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.results.failed}{Colors.RESET}")
        print(f"{Colors.YELLOW}Skipped: {self.results.skipped}{Colors.RESET}")
        print(f"Success Rate: {self.results.success_rate:.1f}%")

        if self.results.failures:
            print(f"\n{Colors.RED}Failed Tests:{Colors.RESET}")
            for test_name, error in self.results.failures:
                print(f"  • {test_name}: {error}")

        if self.results.success_rate >= 90:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Setup verification completed successfully!{Colors.RESET}")
            print(f"{Colors.GREEN}Your QCM Generator Pro installation is ready to use.{Colors.RESET}")
        elif self.results.success_rate >= 70:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Setup verification completed with warnings.{Colors.RESET}")
            print(f"{Colors.YELLOW}Most components are working, but some issues were found.{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ Setup verification failed.{Colors.RESET}")
            print(f"{Colors.RED}Significant issues were found. Please review the failures above.{Colors.RESET}")

        print(f"\n{Colors.CYAN}Next steps:{Colors.RESET}")
        print(f"  • Run 'make test' for comprehensive testing")
        print(f"  • Run 'make run' to start the development server")
        print(f"  • Check CLAUDE.md for development guidelines")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QCM Generator Pro Setup Verification")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--quick", "-q", action="store_true", help="Run only quick tests")

    args = parser.parse_args()

    tester = SetupTester(verbose=args.verbose, quick=args.quick)
    tester.run_all_tests()

    # Exit with appropriate code
    if tester.results.success_rate >= 90:
        sys.exit(0)  # Success
    elif tester.results.success_rate >= 70:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Failure


if __name__ == "__main__":
    main()