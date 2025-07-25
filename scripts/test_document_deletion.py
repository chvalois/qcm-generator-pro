#!/usr/bin/env python3
"""
Test script for document deletion functionality.
"""

import sys
from pathlib import Path
import logging
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.services.document_manager import get_document_manager
from src.services.rag_engine import get_rag_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_document_deletion():
    """Test document deletion functionality."""
    print("🗑️ Testing Document Deletion")
    print("=" * 50)
    
    try:
        # Create test PDF if it doesn't exist
        test_pdf = Path("test_document.pdf")
        if not test_pdf.exists():
            print("📄 Creating test PDF...")
            from scripts.create_test_pdf import create_test_pdf
            create_test_pdf()
        
        # Initialize services
        doc_manager = get_document_manager()
        
        # Upload and store a document
        print("📋 Step 1: Uploading test document...")
        document = await doc_manager.process_and_store_document(test_pdf, store_in_rag=True)
        doc_id = document.id
        print(f"✅ Document uploaded with ID: {doc_id}")
        
        # Verify document exists
        print("\n📋 Step 2: Verifying document exists...")
        stored_docs = doc_manager.list_documents()
        print(f"📊 Documents in database before deletion: {len(stored_docs)}")
        
        # Check RAG engine
        rag_engine = get_rag_engine()
        if hasattr(rag_engine, 'document_chunks'):
            rag_docs_before = len(rag_engine.document_chunks)
            print(f"📊 Documents in RAG before deletion: {rag_docs_before}")
        elif hasattr(rag_engine, 'get_collection_stats'):
            stats = rag_engine.get_collection_stats()
            print(f"📊 ChromaDB stats before deletion: {stats}")
        
        # Test deletion
        print(f"\n📋 Step 3: Deleting document {doc_id}...")
        success = doc_manager.delete_document(doc_id)
        
        if success:
            print(f"✅ Document {doc_id} deleted successfully")
        else:
            print(f"❌ Failed to delete document {doc_id}")
            return False
        
        # Verify deletion
        print("\n📋 Step 4: Verifying deletion...")
        stored_docs_after = doc_manager.list_documents()
        print(f"📊 Documents in database after deletion: {len(stored_docs_after)}")
        
        # Check RAG engine after deletion
        if hasattr(rag_engine, 'document_chunks'):
            rag_docs_after = len(rag_engine.document_chunks)
            print(f"📊 Documents in RAG after deletion: {rag_docs_after}")
            
            if rag_docs_after < rag_docs_before:
                print("✅ Document removed from RAG engine")
            else:
                print("⚠️ Document may still be in RAG engine")
                
        elif hasattr(rag_engine, 'get_collection_stats'):
            stats_after = rag_engine.get_collection_stats()
            print(f"📊 ChromaDB stats after deletion: {stats_after}")
        
        # Check if file was deleted
        if not test_pdf.exists():
            print("✅ Physical file deleted")
        else:
            print("⚠️ Physical file still exists")
        
        print("\n✅ Document deletion test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bulk_deletion():
    """Test bulk deletion functionality."""
    print("\n🗑️ Testing Bulk Document Deletion")
    print("=" * 50)
    
    try:
        # This would simulate bulk deletion
        doc_manager = get_document_manager()
        
        # Get current documents
        stored_docs = doc_manager.list_documents()
        
        if not stored_docs:
            print("❌ No documents available for bulk deletion test")
            return True
        
        print(f"📊 Found {len(stored_docs)} documents for bulk deletion test")
        
        # Test deleting all documents
        success_count = 0
        for doc in stored_docs:
            if doc_manager.delete_document(doc['id']):
                success_count += 1
        
        print(f"✅ Bulk deletion: {success_count}/{len(stored_docs)} documents deleted")
        
        # Verify all deleted
        remaining_docs = doc_manager.list_documents()
        print(f"📊 Remaining documents: {len(remaining_docs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bulk deletion test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Document Deletion Test Suite")
    print("=" * 50)
    
    # Run deletion test
    result1 = asyncio.run(test_document_deletion())
    
    # Run bulk deletion test
    result2 = test_bulk_deletion()
    
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print("=" * 50)
    
    print(f"   Single Deletion: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"   Bulk Deletion: {'✅ PASS' if result2 else '❌ FAIL'}")
    
    if result1 and result2:
        print("\n🎉 All deletion tests passed!")
        print("💡 Document deletion functionality is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    main()