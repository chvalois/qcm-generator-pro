#!/usr/bin/env python3
"""
QCM Generator Pro - Integration Test

Simple integration test to verify that the core system works end-to-end.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all core modules can be imported."""
    logger.info("🔗 Testing core imports...")
    
    try:
        from core.config import settings
        logger.info("  ✅ Configuration loaded")
        
        from models.database import Document, Question
        logger.info("  ✅ Database models imported")
        
        from models.schemas import GenerationConfig, QuestionCreate
        logger.info("  ✅ Pydantic schemas imported")
        
        from services.pdf_processor import process_pdf
        logger.info("  ✅ PDF processor imported")
        
        from services.llm_manager import get_llm_manager
        logger.info("  ✅ LLM manager imported")
        
        from ui.streamlit_app import create_streamlit_interface
        logger.info("  ✅ Streamlit interface imported")
        
        from api.main import app
        logger.info("  ✅ FastAPI app imported")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Import failed: {e}")
        return False


def test_configuration():
    """Test that configuration is loaded correctly."""
    logger.info("⚙️  Testing configuration...")
    
    try:
        from core.config import settings
        
        logger.info(f"  ✅ App name: {settings.app_name}")
        logger.info(f"  ✅ Debug mode: {settings.debug}")
        logger.info(f"  ✅ Default language: {settings.default_language}")
        logger.info(f"  ✅ Data directory: {settings.data_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Configuration test failed: {e}")
        return False


def test_database_models():
    """Test that database models can be created."""
    logger.info("🗄️  Testing database models...")
    
    try:
        from models.database import Document, Question, DocumentTheme
        from models.enums import ProcessingStatus, Language
        from datetime import datetime
        
        # Test document creation
        doc = Document(
            filename="test.pdf",
            file_path="/test/path.pdf",
            upload_date=datetime.now(),
            total_pages=10,
            language=Language.FR,
            processing_status=ProcessingStatus.PENDING
        )
        
        logger.info("  ✅ Document model creation works")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Database model test failed: {e}")
        return False


def test_schema_validation():
    """Test that Pydantic schemas validate correctly."""
    logger.info("📋 Testing schema validation...")
    
    try:
        from models.schemas import GenerationConfig, QuestionCreate
        from models.enums import Language, Difficulty, QuestionType
        
        # Test generation config
        config = GenerationConfig(
            num_questions=5,
            language=Language.FR,
            difficulty_distribution={
                Difficulty.EASY: 0.3,
                Difficulty.MEDIUM: 0.5,
                Difficulty.HARD: 0.2
            }
        )
        
        logger.info("  ✅ GenerationConfig validation works")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Schema validation test failed: {e}")
        return False


def test_basic_functionality():
    """Test basic service functionality."""
    logger.info("🔧 Testing basic functionality...")
    
    try:
        from services.llm_manager import get_llm_manager
        
        # Test LLM manager creation
        llm_manager = get_llm_manager()
        logger.info("  ✅ LLM manager created")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Basic functionality test failed: {e}")
        return False


def main():
    """Run integration tests."""
    logger.info("🎯 QCM Generator Pro - Integration Test")
    logger.info("=" * 50)
    
    tests = [
        ("Core Imports", test_imports),
        ("Configuration", test_configuration), 
        ("Database Models", test_database_models),
        ("Schema Validation", test_schema_validation),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} test passed")
            else:
                logger.error(f"❌ {test_name} test failed")
        except Exception as e:
            logger.error(f"❌ {test_name} test error: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed!")
        logger.info("\n💡 System is ready to run:")
        logger.info("  • make run-app        # Full application")
        logger.info("  • make run-api-only   # API server only")
        logger.info("  • make run-ui-only    # UI interface only")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())