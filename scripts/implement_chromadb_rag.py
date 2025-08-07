#!/usr/bin/env python3
"""
Script pour implémenter un RAG engine avec ChromaDB persistant
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional
from src.models.schemas import DocumentChunk, QuestionContext
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)


class ChromaDBRAGEngine:
    """
    RAG engine using ChromaDB for persistent vector storage.
    """
    
    def __init__(self):
        """Initialize ChromaDB RAG engine."""
        # Initialize ChromaDB client
        self.db_path = settings.vector_store.chroma_db_path
        self.collection_name = settings.vector_store.chroma_collection_name
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "QCM document chunks for RAG"}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
    
    def add_document_chunks(self, document_id: str, chunks: List[DocumentChunk]) -> bool:
        """
        Add document chunks to ChromaDB.
        
        Args:
            document_id: Unique document identifier
            chunks: List of document chunks to add
            
        Returns:
            True if successful
        """
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk.content)
                
                # Prepare metadata
                metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "word_count": len(chunk.content.split()),
                    "chunk_size": len(chunk.content)
                }
                
                # Add original metadata if present
                if chunk.metadata:
                    metadata.update(chunk.metadata)
                
                metadatas.append(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks for document {document_id} to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            return False
    
    def get_question_context(
        self,
        topic: str,
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 5,
        themes_filter: Optional[List[str]] = None
    ) -> QuestionContext:
        """
        Get relevant context for question generation using ChromaDB similarity search.
        
        Args:
            topic: Topic or query for context retrieval
            document_ids: Filter by specific document IDs
            max_chunks: Maximum number of chunks to return
            themes_filter: Filter by specific themes
            
        Returns:
            QuestionContext with relevant chunks
        """
        try:
            # Build where clause for filtering
            where_clause = {}
            if document_ids:
                where_clause["document_id"] = {"$in": document_ids}
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[topic],
                n_results=max_chunks,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to DocumentChunks
            source_chunks = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    chunk = DocumentChunk(
                        content=doc,
                        metadata=metadata or {},
                        similarity_score=1 - distance  # Convert distance to similarity
                    )
                    source_chunks.append(chunk)
            
            # Calculate overall confidence
            confidence_score = 0.8 if source_chunks else 0.0
            if source_chunks:
                avg_similarity = sum(chunk.similarity_score for chunk in source_chunks) / len(source_chunks)
                confidence_score = min(0.95, avg_similarity + 0.1)
            
            return QuestionContext(
                topic=topic,
                source_chunks=source_chunks,
                confidence_score=confidence_score,
                retrieval_method="chromadb_similarity"
            )
            
        except Exception as e:
            logger.error(f"Failed to get context from ChromaDB: {e}")
            # Return empty context on error
            return QuestionContext(
                topic=topic,
                source_chunks=[],
                confidence_score=0.0,
                retrieval_method="chromadb_error"
            )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            count = self.collection.count()
            
            if count == 0:
                return {
                    "total_chunks": 0,
                    "documents": {},
                    "status": "empty"
                }
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(
                limit=min(1000, count),
                include=['metadatas']
            )
            
            # Analyze document distribution
            doc_counts = {}
            for metadata in sample_results['metadatas']:
                if metadata and 'document_id' in metadata:
                    doc_id = metadata['document_id']
                    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            
            return {
                "total_chunks": count,
                "documents": doc_counts,
                "collection_name": self.collection_name,
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"status": "error", "error": str(e)}


def migrate_simple_to_chromadb():
    """Migrate data from SimpleRAGEngine to ChromaDB."""
    print("🔄 Migrating from SimpleRAGEngine to ChromaDB")
    print("=" * 50)
    
    try:
        # Import current RAG engine
        from src.services.infrastructure.rag_engine import get_rag_engine
        
        simple_rag = get_rag_engine()
        
        if not hasattr(simple_rag, 'document_chunks') or not simple_rag.document_chunks:
            print("❌ No data found in SimpleRAGEngine to migrate")
            print("💡 Upload documents through the UI first, then run migration")
            return
        
        # Initialize ChromaDB RAG
        chroma_rag = ChromaDBRAGEngine()
        
        # Migrate data
        total_migrated = 0
        for doc_id, chunks in simple_rag.document_chunks.items():
            print(f"📄 Migrating document {doc_id}: {len(chunks)} chunks")
            
            if chroma_rag.add_document_chunks(doc_id, chunks):
                total_migrated += len(chunks)
                print(f"   ✅ Migrated successfully")
            else:
                print(f"   ❌ Migration failed")
        
        print(f"\n✅ Migration completed: {total_migrated} chunks migrated")
        
        # Show stats
        stats = chroma_rag.get_collection_stats()
        print(f"📊 ChromaDB Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()


def test_chromadb_rag():
    """Test ChromaDB RAG functionality."""
    print("🧪 Testing ChromaDB RAG Engine")
    print("=" * 50)
    
    try:
        # Initialize ChromaDB RAG
        chroma_rag = ChromaDBRAGEngine()
        
        # Get stats
        stats = chroma_rag.get_collection_stats()
        print(f"📊 Collection Stats: {stats}")
        
        if stats['total_chunks'] > 0:
            # Test search
            print(f"\n🔍 Testing search...")
            context = chroma_rag.get_question_context(
                topic="architecture",
                max_chunks=3
            )
            
            print(f"   Found {len(context.source_chunks)} relevant chunks")
            print(f"   Confidence: {context.confidence_score:.2f}")
            
            if context.source_chunks:
                for i, chunk in enumerate(context.source_chunks[:2]):
                    print(f"\n   📄 Chunk {i+1}:")
                    print(f"      Content: {chunk.content[:150]}...")
                    print(f"      Similarity: {chunk.similarity_score:.3f}")
                    print(f"      Metadata: {chunk.metadata}")
        else:
            print("❌ No data in ChromaDB. Run migration first.")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "migrate":
            migrate_simple_to_chromadb()
        elif command == "test":
            test_chromadb_rag()
        else:
            print("Usage:")
            print("  python implement_chromadb_rag.py migrate")
            print("  python implement_chromadb_rag.py test")
    else:
        test_chromadb_rag()