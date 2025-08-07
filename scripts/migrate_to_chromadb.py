#!/usr/bin/env python3
"""
Migration script to move from SimpleRAGEngine to ChromaDB persistence.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.services.infrastructure.rag_engine import get_rag_engine, switch_rag_engine
from src.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_simple_to_chromadb():
    """Migrate data from SimpleRAGEngine to ChromaDB."""
    print("🔄 Migration: SimpleRAGEngine → ChromaDB")
    print("=" * 50)
    
    try:
        # Get current Simple RAG engine
        print("📋 Step 1: Getting current SimpleRAGEngine data...")
        simple_rag = get_rag_engine("simple")
        
        if not hasattr(simple_rag, 'document_chunks') or not simple_rag.document_chunks:
            print("❌ No data found in SimpleRAGEngine to migrate")
            print("💡 Upload documents through the UI first, then run migration")
            return False
        
        print(f"✅ Found {len(simple_rag.document_chunks)} document(s) in SimpleRAGEngine")
        
        # Initialize ChromaDB RAG engine
        print("\n📋 Step 2: Initializing ChromaDB RAG engine...")
        chroma_rag = get_rag_engine("chromadb")
        
        if chroma_rag is None:
            print("❌ Failed to initialize ChromaDB RAG engine")
            return False
        
        print("✅ ChromaDB RAG engine initialized")
        
        # Migrate data
        print("\n📋 Step 3: Migrating document chunks...")
        total_migrated = 0
        
        for doc_id, chunks in simple_rag.document_chunks.items():
            print(f"📄 Migrating document {doc_id}: {len(chunks)} chunks")
            
            try:
                # Extract text and metadata from chunks
                full_text = " ".join(chunk.content for chunk in chunks)
                metadata = chunks[0].metadata if chunks else {}
                themes = chunks[0].themes if chunks else []
                
                # Add to ChromaDB
                chroma_rag.add_document(doc_id, full_text, metadata, themes)
                total_migrated += len(chunks)
                print(f"   ✅ Migrated successfully")
                
            except Exception as e:
                print(f"   ❌ Migration failed: {e}")
                continue
        
        print(f"\n✅ Migration completed: {total_migrated} chunks migrated")
        
        # Verify migration
        print("\n📋 Step 4: Verifying migration...")
        stats = chroma_rag.get_collection_stats()
        print(f"📊 ChromaDB Stats: {stats}")
        
        if stats.get("total_chunks", 0) > 0:
            print("✅ Migration verification successful")
            return True
        else:
            print("❌ Migration verification failed")
            return False
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chromadb_functionality():
    """Test ChromaDB RAG functionality after migration."""
    print("\n🧪 Testing ChromaDB Functionality")
    print("=" * 50)
    
    try:
        # Get ChromaDB engine
        chroma_rag = get_rag_engine("chromadb")
        
        # Get stats
        stats = chroma_rag.get_collection_stats()
        print(f"📊 Collection Stats: {stats}")
        
        if stats.get('total_chunks', 0) > 0:
            # Test search
            print("\n🔍 Testing search functionality...")
            context = chroma_rag.get_question_context(
                topic="test",
                context_size=3
            )
            
            print(f"   Found {len(context.source_chunks)} relevant chunks")
            print(f"   Confidence: {context.confidence_score:.2f}")
            print(f"   Context length: {len(context.context_text)} characters")
            
            if context.source_chunks:
                print("\n📄 Sample results:")
                for i, chunk_info in enumerate(context.source_chunks[:2]):
                    print(f"   Chunk {i+1}:")
                    print(f"      Document: {chunk_info.get('document_id')}")
                    print(f"      Similarity: {chunk_info.get('similarity', 0):.3f}")
            
            # Test document listing
            print("\n📋 Testing document listing...")
            documents = chroma_rag.list_documents()
            print(f"   Found {len(documents)} documents:")
            for doc in documents:
                print(f"      - {doc['document_id']}: {doc['chunk_count']} chunks, themes: {doc['themes']}")
            
            print("\n✅ All tests passed!")
            return True
        else:
            print("❌ No data in ChromaDB to test")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def update_configuration():
    """Update configuration to use ChromaDB by default."""
    print("\n⚙️ Updating Configuration")
    print("=" * 50)
    
    env_file = project_root / ".env"
    
    try:
        # Read existing .env file
        env_content = ""
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_content = f.read()
        
        # Check if VECTOR_ENGINE_TYPE is already set
        if "VECTOR_ENGINE_TYPE" in env_content:
            print("⚠️  VECTOR_ENGINE_TYPE already configured in .env")
            return
        
        # Add ChromaDB configuration
        with open(env_file, 'a') as f:
            f.write("\n# Vector Store Configuration\n")
            f.write("VECTOR_ENGINE_TYPE=chromadb\n")
        
        print("✅ Updated .env to use ChromaDB by default")
        print("💡 Restart the application to apply the new configuration")
        
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")


def main():
    """Main migration process."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "migrate":
            success = migrate_simple_to_chromadb()
            if success:
                test_chromadb_functionality()
                update_configuration()
        elif command == "test":
            test_chromadb_functionality()
        elif command == "config":
            update_configuration()
        else:
            print("Usage:")
            print("  python migrate_to_chromadb.py migrate   # Full migration")
            print("  python migrate_to_chromadb.py test     # Test ChromaDB")
            print("  python migrate_to_chromadb.py config   # Update config only")
    else:
        # Run full migration by default
        print("🚀 Starting Full Migration Process")
        print("=" * 50)
        
        success = migrate_simple_to_chromadb()
        if success:
            test_chromadb_functionality()
            update_configuration()
            print("\n🎉 Migration completed successfully!")
            print("💡 Restart the application to use ChromaDB persistence")
        else:
            print("\n❌ Migration failed. Check logs for details.")


if __name__ == "__main__":
    main()