#!/usr/bin/env python3
"""
Test script for document persistence and ChromaDB integration.
"""

import sys
from pathlib import Path
import logging
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.services.infrastructure.rag_engine import get_rag_engine, switch_rag_engine
from src.services.document.document_manager import get_document_manager
from src.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rag_engine_switching():
    """Test switching between RAG engines."""
    print("🔄 Testing RAG Engine Switching")
    print("=" * 50)
    
    try:
        # Test SimpleRAGEngine
        print("📋 Testing SimpleRAGEngine...")
        simple_rag = get_rag_engine("simple")
        print(f"   ✅ SimpleRAGEngine: {type(simple_rag).__name__}")
        
        # Test ChromaDB RAG Engine
        print("📋 Testing ChromaDBRAGEngine...")
        try:
            chroma_rag = get_rag_engine("chromadb")
            print(f"   ✅ ChromaDBRAGEngine: {type(chroma_rag).__name__}")
            
            # Test ChromaDB functionality
            stats = chroma_rag.get_collection_stats()
            print(f"   📊 ChromaDB Stats: {stats}")
            
        except Exception as e:
            print(f"   ❌ ChromaDB failed: {e}")
            return False
        
        # Test switching
        print("📋 Testing switching...")
        success = switch_rag_engine("chromadb")
        print(f"   Switch to ChromaDB: {'✅' if success else '❌'}")
        
        success = switch_rag_engine("simple") 
        print(f"   Switch to Simple: {'✅' if success else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG engine switching test failed: {e}")
        return False


def test_document_manager():
    """Test document manager functionality."""
    print("\n📋 Testing Document Manager")
    print("=" * 50)
    
    try:
        # Initialize document manager
        doc_manager = get_document_manager()
        print("✅ Document manager initialized")
        
        # Test listing documents (should be empty initially)
        documents = doc_manager.list_documents()
        print(f"📄 Found {len(documents)} existing documents")
        
        # Test getting all themes
        themes = doc_manager.get_all_themes()
        print(f"🎯 Found {len(themes)} unique themes")
        
        if themes:
            print("   Sample themes:")
            for theme in themes[:3]:
                print(f"      - {theme['name']}: {theme['document_count']} docs, confidence: {theme['avg_confidence']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration values."""
    print("\n⚙️  Testing Configuration")
    print("=" * 50)
    
    try:
        print(f"📊 Vector Engine Type: {settings.vector_store.engine_type}")
        print(f"📊 ChromaDB Path: {settings.vector_store.chroma_db_path}")
        print(f"📊 Collection Name: {settings.vector_store.chroma_collection_name}")
        print(f"📊 Database URL: {settings.database.url}")
        print(f"📊 Chunk Size: {settings.processing.default_chunk_size}")
        print(f"📊 Default LLM: {settings.llm.default_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_database_connection():
    """Test database connection and tables."""
    print("\n🗄️  Testing Database Connection")
    print("=" * 50)
    
    try:
        doc_manager = get_document_manager()
        
        # Test session creation
        with doc_manager.get_session() as session:
            print("✅ Database connection successful")
            
            # Check if tables exist by running a simple query
            from src.models.database import Document
            from sqlalchemy import select
            
            stmt = select(Document).limit(1)
            result = session.execute(stmt).first()
            print("✅ Document table accessible")
            
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_current_data():
    """Show current data in the system."""
    print("\n📊 Current System Data")
    print("=" * 50)
    
    try:
        # Simple RAG Engine data
        simple_rag = get_rag_engine("simple")
        if hasattr(simple_rag, 'document_chunks'):
            print(f"📄 SimpleRAG Documents: {len(simple_rag.document_chunks)}")
            for doc_id, chunks in simple_rag.document_chunks.items():
                print(f"   - {doc_id}: {len(chunks)} chunks")
        
        # ChromaDB data
        try:
            chroma_rag = get_rag_engine("chromadb")
            stats = chroma_rag.get_collection_stats()
            print(f"📊 ChromaDB Stats: {stats}")
            
            documents = chroma_rag.list_documents()
            print(f"📄 ChromaDB Documents: {len(documents)}")
            for doc in documents:
                print(f"   - {doc['document_id']}: {doc['chunk_count']} chunks, themes: {doc['themes']}")
                
        except Exception as e:
            print(f"❌ ChromaDB data check failed: {e}")
        
        # Database data
        doc_manager = get_document_manager()
        db_documents = doc_manager.list_documents()
        print(f"🗄️  Database Documents: {len(db_documents)}")
        for doc in db_documents:
            print(f"   - {doc['filename']}: {doc['total_pages']} pages, {len(doc['themes'])} themes")
        
    except Exception as e:
        print(f"❌ Data overview failed: {e}")


def main():
    """Run all tests."""
    print("🧪 QCM Generator Persistence Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Database Connection", test_database_connection), 
        ("RAG Engine Switching", test_rag_engine_switching),
        ("Document Manager", test_document_manager)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Show current data
    show_current_data()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The persistence system is ready.")
        print("💡 Next steps:")
        print("   1. Upload a document through the UI")
        print("   2. Run migration script: python scripts/migrate_to_chromadb.py")
        print("   3. Test document reusability")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    main()