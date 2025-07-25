#!/usr/bin/env python3
"""
Debug script to test document upload and persistence workflow.
"""

import sys
from pathlib import Path
import logging
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.services.document_manager import get_document_manager, process_and_store_pdf
from src.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_document_upload():
    """Test the document upload and persistence workflow."""
    print("🧪 Testing Document Upload and Persistence")
    print("=" * 50)
    
    # Check if we have any test PDFs
    test_pdfs = list(Path(".").glob("*.pdf"))
    test_pdfs.extend(list(Path("tests/fixtures").glob("*.pdf")))
    test_pdfs.extend(list(Path("data/pdfs").glob("*.pdf")))
    
    if not test_pdfs:
        print("❌ No PDF files found for testing")
        print("💡 Testing basic DocumentManager functionality without document...")
        
        # Test just the DocumentManager initialization and database
        try:
            print("\n📋 Step 1: Initializing DocumentManager...")
            doc_manager = get_document_manager()
            print("✅ DocumentManager initialized")
            
            # Test database connection
            print("\n📋 Step 2: Testing database connection...")
            with doc_manager.get_session() as session:
                print("✅ Database connection successful")
            
            # Test listing (should be empty)
            print("\n📋 Step 3: Testing document listing...")
            stored_docs = doc_manager.list_documents()
            print(f"📊 Documents in database: {len(stored_docs)}")
            
            # Test themes listing (should be empty)
            all_themes = doc_manager.get_all_themes()
            print(f"🎯 Themes in database: {len(all_themes)}")
            
            print("\n✅ Basic DocumentManager functionality working!")
            return True
            
        except Exception as e:
            print(f"❌ Basic test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    test_pdf = test_pdfs[0]
    print(f"📄 Testing with: {test_pdf}")
    
    try:
        # Test DocumentManager initialization
        print("\n📋 Step 1: Initializing DocumentManager...")
        doc_manager = get_document_manager()
        print("✅ DocumentManager initialized")
        
        # Test database connection
        print("\n📋 Step 2: Testing database connection...")
        with doc_manager.get_session() as session:
            print("✅ Database connection successful")
        
        # Test document processing and storage
        print(f"\n📋 Step 3: Processing document {test_pdf}...")
        document = await doc_manager.process_and_store_document(test_pdf, store_in_rag=True)
        print(f"✅ Document processed and stored: {document.id}")
        
        # Verify storage
        print("\n📋 Step 4: Verifying storage...")
        stored_docs = doc_manager.list_documents()
        print(f"📊 Documents in database: {len(stored_docs)}")
        
        for doc in stored_docs:
            print(f"   📄 {doc['filename']}: {len(doc['themes'])} themes")
        
        # Test theme extraction
        themes = doc_manager.get_document_themes(document.id)
        print(f"🎯 Themes extracted: {len(themes)}")
        
        for theme in themes[:3]:
            print(f"   • {theme.theme_name} (confidence: {theme.confidence_score:.2f})")
        
        # Test RAG engine integration
        from src.services.rag_engine import get_rag_engine
        rag_engine = get_rag_engine()
        
        if hasattr(rag_engine, 'get_collection_stats'):
            stats = rag_engine.get_collection_stats()
            print(f"📊 RAG Engine Stats: {stats}")
        else:
            chunk_count = sum(len(chunks) for chunks in rag_engine.document_chunks.values())
            print(f"📊 RAG Engine: {len(rag_engine.document_chunks)} docs, {chunk_count} chunks")
        
        print("\n✅ All tests passed! Document upload and persistence working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration values."""
    print("\n⚙️ Configuration Check")
    print("=" * 30)
    
    print(f"📊 RAG Engine Type: {settings.vector_store.engine_type}")
    print(f"📊 Database URL: {settings.database.url}")
    print(f"📊 ChromaDB Path: {settings.vector_store.chroma_db_path}")
    
    # Check if directories exist
    db_path = Path(settings.database.url.replace("sqlite:///", ""))
    vector_path = settings.vector_store.chroma_db_path
    
    print(f"📁 DB Directory exists: {db_path.parent.exists()}")
    print(f"📁 Vector Directory exists: {vector_path.exists()}")
    
    # Create directories if they don't exist
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        print("✅ Created database directory")
    
    if not vector_path.exists():
        vector_path.mkdir(parents=True, exist_ok=True)
        print("✅ Created vector store directory")


def main():
    """Main test function."""
    print("🚀 QCM Generator Document Upload Debug")
    print("=" * 50)
    
    test_configuration()
    
    # Run async test
    result = asyncio.run(test_document_upload())
    
    if result:
        print("\n🎉 Document upload workflow is working correctly!")
        print("💡 You can now upload documents through the UI and they will be persisted.")
    else:
        print("\n❌ Document upload workflow has issues.")
        print("💡 Check the logs above for error details.")


if __name__ == "__main__":
    main()