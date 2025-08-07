#!/usr/bin/env python3
"""
QCM Generator Pro - Setup Validation Script

This script validates that the development environment is properly set up
with all required dependencies and configurations.
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.10+)"


def check_required_packages() -> Tuple[bool, List[str]]:
    """Check if all required packages are installed."""
    packages = [
        # Core packages
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("sqlalchemy", "SQLAlchemy"),
        ("python_dotenv", "python-dotenv"),
        
        # Processing packages
        ("pypdf", "PyPDF"),
        ("httpx", "HTTPX"),
        
        # UI packages
        ("streamlit", "Streamlit"),
        
        # LLM packages
        ("langchain", "Langchain"),
        ("openai", "OpenAI"),
        ("chromadb", "ChromaDB"),
    ]
    
    results = []
    all_installed = True
    
    for package_name, display_name in packages:
        try:
            importlib.import_module(package_name)
            results.append(f"✅ {display_name}")
        except ImportError:
            results.append(f"❌ {display_name} (not installed)")
            all_installed = False
    
    return all_installed, results


def check_project_structure() -> Tuple[bool, List[str]]:
    """Check if project structure is correct."""
    project_root = Path(__file__).parent.parent
    
    required_paths = [
        "src/",
        "src/api/",
        "src/core/",
        "src/models/",
        "src/services/",
        "src/ui/",
        "scripts/",
        "tests/",
        "pyproject.toml",
        "Makefile",
        "CLAUDE.md"
    ]
    
    results = []
    all_exist = True
    
    for path_str in required_paths:
        path = project_root / path_str
        if path.exists():
            results.append(f"✅ {path_str}")
        else:
            results.append(f"❌ {path_str} (missing)")
            all_exist = False
    
    return all_exist, results


def check_data_directories() -> Tuple[bool, List[str]]:
    """Check if data directories exist or can be created."""
    project_root = Path(__file__).parent.parent
    
    data_dirs = [
        "data/",
        "data/pdfs/",
        "data/vectorstore/",
        "data/database/",
        "data/exports/",
        "data/cache/"
    ]
    
    results = []
    all_ready = True
    
    for dir_str in data_dirs:
        dir_path = project_root / dir_str
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            results.append(f"✅ {dir_str} (ready)")
        except Exception as e:
            results.append(f"❌ {dir_str} (error: {e})")
            all_ready = False
    
    return all_ready, results


def check_imports() -> Tuple[bool, List[str]]:
    """Check if core modules can be imported."""
    modules = [
        ("src.core.config", "Core configuration"),
        ("src.models.database", "Database models"),
        ("src.models.schemas", "Pydantic schemas"),
        ("src.services.document.pdf_processor", "PDF processor"),
        ("src.services.llm.llm_manager", "LLM manager"),
    ]
    
    results = []
    all_imported = True
    
    for module_name, display_name in modules:
        try:
            importlib.import_module(module_name)
            results.append(f"✅ {display_name}")
        except Exception as e:
            results.append(f"❌ {display_name} (error: {str(e)[:60]}...)")
            all_imported = False
    
    return all_imported, results


def check_optional_features() -> Tuple[bool, List[str]]:
    """Check optional features availability."""
    results = []
    
    # Check if .env file exists
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        results.append("✅ .env configuration file found")
    else:
        results.append("⚠️  .env file not found (copy from .env.example)")
    
    # Check if models directory exists
    models_dir = Path(__file__).parent.parent / "models"
    if models_dir.exists():
        results.append("✅ Local models directory found")
    else:
        results.append("ℹ️  Local models directory not found (optional)")
    
    return True, results


def main() -> None:
    """Main validation function."""
    logger.info("🎯 QCM Generator Pro - Setup Validation")
    logger.info("=" * 50)
    
    all_checks_passed = True
    
    # Python version check
    python_ok, python_msg = check_python_version()
    logger.info(f"\n📋 Python Version:\n{python_msg}")
    if not python_ok:
        all_checks_passed = False
    
    # Required packages check
    packages_ok, package_results = check_required_packages()
    logger.info(f"\n📦 Required Packages:")
    for result in package_results:
        logger.info(f"  {result}")
    if not packages_ok:
        all_checks_passed = False
    
    # Project structure check
    structure_ok, structure_results = check_project_structure()
    logger.info(f"\n📁 Project Structure:")  
    for result in structure_results:
        logger.info(f"  {result}")
    if not structure_ok:
        all_checks_passed = False
    
    # Data directories check
    dirs_ok, dirs_results = check_data_directories()
    logger.info(f"\n🗂️  Data Directories:")
    for result in dirs_results:
        logger.info(f"  {result}")
    if not dirs_ok:
        all_checks_passed = False
    
    # Core imports check
    imports_ok, import_results = check_imports()
    logger.info(f"\n🔗 Core Imports:")
    for result in import_results:
        logger.info(f"  {result}")
    if not imports_ok:
        all_checks_passed = False
    
    # Optional features check
    optional_ok, optional_results = check_optional_features()
    logger.info(f"\n⚙️  Optional Features:")
    for result in optional_results:
        logger.info(f"  {result}")
    
    # Final result
    logger.info("\n" + "=" * 50)
    if all_checks_passed:
        logger.info("🎉 All critical checks passed! Setup is ready.")
        logger.info("\n💡 Next steps:")
        logger.info("  1. Copy .env.example to .env and configure")
        logger.info("  2. Run: make run-app")
        logger.info("  3. Open http://127.0.0.1:7860 for UI")
        logger.info("  4. API docs at http://127.0.0.1:8000/docs")
        sys.exit(0)
    else:
        logger.info("❌ Some critical checks failed. Please fix the issues above.")
        logger.info("\n🔧 Common fixes:")
        logger.info("  • Install missing packages: pip install -e .[ui,full]")
        logger.info("  • Check Python version: python3 --version")
        logger.info("  • Run: make dev-setup")
        sys.exit(1)


if __name__ == "__main__":
    main()