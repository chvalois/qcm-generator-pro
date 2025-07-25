#!/usr/bin/env python3
"""
Script pour vérifier les données stockées dans le RAG engine en mémoire
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
from src.services.rag_engine import SimpleRAGEngine, get_rag_engine


def check_rag_data():
    """Check what data is stored in the RAG engine."""
    print("🔍 Checking RAG Engine Data")
    print("=" * 50)
    
    try:
        # Get the RAG engine instance
        rag_engine = get_rag_engine()
        
        if not hasattr(rag_engine, 'document_chunks'):
            print("❌ RAG engine doesn't have document_chunks attribute")
            return
        
        chunks_data = rag_engine.document_chunks
        
        if not chunks_data:
            print("❌ No documents stored in RAG engine memory")
            print("\n💡 This means:")
            print("   - No documents have been processed and added to RAG")
            print("   - You need to upload and process a PDF through the UI first")
            print("   - The RAG engine stores data in memory, not in ChromaDB")
            return
        
        print(f"✅ Found {len(chunks_data)} document(s) in RAG memory")
        
        total_chunks = 0
        for doc_id, chunks in chunks_data.items():
            print(f"\n📄 Document ID: {doc_id}")
            print(f"   Chunks: {len(chunks)}")
            total_chunks += len(chunks)
            
            if chunks:
                # Show first chunk as example
                first_chunk = chunks[0]
                print(f"   First chunk content: {first_chunk.content[:200]}...")
                print(f"   First chunk metadata: {first_chunk.metadata}")
        
        print(f"\n📊 Total chunks across all documents: {total_chunks}")
        
        # Show sample search
        if total_chunks > 0:
            print(f"\n🔍 Testing search functionality...")
            context = rag_engine.get_question_context(
                topic="test",
                document_ids=list(chunks_data.keys()),
                max_chunks=3
            )
            print(f"   Search returned {len(context.source_chunks)} chunks")
            if context.source_chunks:
                print(f"   First result: {context.source_chunks[0].content[:100]}...")
    
    except Exception as e:
        print(f"❌ Error checking RAG data: {e}")
        import traceback
        traceback.print_exc()


def show_rag_architecture():
    """Show the current RAG architecture."""
    print("\n🏗️ Current RAG Architecture")
    print("=" * 50)
    
    print("📋 Current setup:")
    print("   - RAG Engine: SimpleRAGEngine (in-memory)")
    print("   - Storage: Python dict in memory")
    print("   - Persistence: None (data lost on restart)")
    print("   - Vector Store: Not used")
    print("   - Embeddings: Text similarity only")
    
    print("\n📋 ChromaDB Configuration (unused):")
    print("   - Path: ./data/vectorstore")
    print("   - Collection: qcm_documents")
    print("   - Status: Configured but not implemented")
    
    print("\n💡 To see the actual data:")
    print("   1. Upload a PDF through the Streamlit UI")
    print("   2. Process the document")
    print("   3. Run this script again")


if __name__ == "__main__":
    check_rag_data()
    show_rag_architecture()