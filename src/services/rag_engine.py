"""
QCM Generator Pro - RAG Engine Service

This module handles Retrieval-Augmented Generation for QCM creation,
providing context-aware question generation through document retrieval.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, List, Dict, Optional

from ..core.config import settings
from ..models.schemas import DocumentChunk, QuestionContext

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGEngineError(Exception):
    """Exception raised when RAG operations fail."""
    pass


class SimpleRAGEngine:
    """
    Simple RAG engine using text similarity for document retrieval.
    
    This is a basic implementation that doesn't require ChromaDB
    for the simplified version. Can be enhanced later with vector databases.
    """
    
    def __init__(self):
        """Initialize RAG engine."""
        self.chunk_size = settings.processing.default_chunk_size
        self.chunk_overlap = settings.processing.default_chunk_overlap
        self.max_results = settings.vector_store.default_search_limit
        
        # In-memory storage for document chunks
        self.document_chunks: dict[str, list[DocumentChunk]] = {}
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using improved word overlap.
        
        Args:
            text1: First text (query)
            text2: Second text (document content)
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1.strip() or not text2.strip():
            return 0.0
            
        # Simple tokenization and normalization
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate overlap ratio based on query terms (more forgiving for theme searches)
        intersection = len(words1.intersection(words2))
        
        # Use query length as denominator for better theme matching
        # This gives higher scores when query terms appear in content
        query_coverage = intersection / len(words1) if words1 else 0.0
        
        # Also calculate traditional Jaccard for comparison
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Return the maximum of both approaches for better recall
        return max(query_coverage * 0.3, jaccard)
        
    def add_document(
        self, 
        document_id: str, 
        text: str, 
        metadata: dict[str, Any] | None = None,
        themes: list[str] | None = None
    ) -> None:
        """
        Add a document to the RAG engine.
        
        Args:
            document_id: Unique document identifier
            text: Document text
            metadata: Document metadata
            themes: Document themes
        """
        logger.info(f"Adding document to RAG engine: {document_id}")
        
        # Split text into chunks
        chunks = self._create_chunks(text)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Flatten metadata to ensure Pydantic compatibility
            flattened_metadata = {}
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        # Convert complex types to strings for Pydantic compatibility
                        flattened_metadata[key] = json.dumps(value) if value else ""
                    elif value is not None:
                        flattened_metadata[key] = str(value)
                    else:
                        flattened_metadata[key] = ""
            
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                content=chunk_text,
                chunk_index=i,
                start_char=text.find(chunk_text) if chunk_text in text else 0,
                end_char=text.find(chunk_text) + len(chunk_text) if chunk_text in text else len(chunk_text),
                metadata=flattened_metadata,
                themes=themes or []
            )
            document_chunks.append(chunk)
            
        self.document_chunks[document_id] = document_chunks
        logger.info(f"Added {len(document_chunks)} chunks for document {document_id}")
        
    def _create_chunks(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at a sentence boundary
            if end < text_length:
                # Look for sentence ending within the last 200 characters
                search_start = max(start + self.chunk_size - 200, start)
                sentence_endings = ['.', '!', '?', '\n\n']
                
                best_break = -1
                for ending in sentence_endings:
                    pos = text.rfind(ending, search_start, end)
                    if pos > best_break:
                        best_break = pos
                        
                if best_break > start:
                    end = best_break + 1
                else:
                    # Fall back to word boundary
                    word_break = text.rfind(' ', start, end)
                    if word_break > start:
                        end = word_break
                        
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
            
        return chunks
        
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        document_ids: list[str] | None = None,
        themes_filter: list[str] | None = None,
        limit: int | None = None
    ) -> list[DocumentChunk]:
        """
        Retrieve document chunks relevant to the query.
        
        Args:
            query: Search query
            document_ids: Filter by specific document IDs
            themes_filter: Filter by themes
            limit: Maximum number of results
            
        Returns:
            List of relevant document chunks
        """
        if not query.strip():
            return []
            
        limit = limit or self.max_results
        relevant_chunks = []
        
        # Search through all documents
        search_docs = document_ids or list(self.document_chunks.keys())
        
        for doc_id in search_docs:
            if doc_id not in self.document_chunks:
                continue
                
            for chunk in self.document_chunks[doc_id]:
                # Calculate similarity
                similarity = self.calculate_text_similarity(query, chunk.content)
                
                # Apply theme filter if specified
                if themes_filter and chunk.themes:
                    if not any(theme in chunk.themes for theme in themes_filter):
                        continue
                        
                if similarity > 0.01:  # Lower minimum similarity threshold for better recall
                    chunk_with_score = chunk.model_copy()
                    chunk_with_score.metadata = {
                        **chunk.metadata,
                        "similarity_score": similarity
                    }
                    relevant_chunks.append(chunk_with_score)
                    
        # Sort by similarity and return top results
        relevant_chunks.sort(
            key=lambda x: x.metadata.get("similarity_score", 0.0), 
            reverse=True
        )
        
        return relevant_chunks[:limit]
        
    def get_question_context(
        self, 
        topic: str, 
        document_ids: list[str] | None = None,
        themes_filter: list[str] | None = None,
        context_size: int = 3
    ) -> QuestionContext:
        """
        Get context for question generation on a specific topic.
        
        Args:
            topic: Topic for question generation
            document_ids: Filter by document IDs
            themes_filter: Filter by themes
            context_size: Number of chunks to include
            
        Returns:
            Question context with relevant information
        """
        logger.debug(f"Getting question context for topic: {topic}")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(
            topic, 
            document_ids=document_ids,
            themes_filter=themes_filter,
            limit=context_size
        )
        
        if not relevant_chunks:
            logger.warning(f"No relevant context found for topic: {topic}")
            return QuestionContext(
                topic=topic,
                context_text="",
                source_chunks=[],
                themes=themes_filter or [],
                confidence_score=0.0,
                metadata={}
            )
            
        # Combine chunk texts for context
        context_texts = []
        source_info = []
        total_similarity = 0.0
        
        for chunk in relevant_chunks:
            context_texts.append(chunk.content)
            source_info.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "similarity": chunk.metadata.get("similarity_score", 0.0)
            })
            total_similarity += chunk.metadata.get("similarity_score", 0.0)
            
        # Calculate average confidence
        confidence = total_similarity / len(relevant_chunks) if relevant_chunks else 0.0
        
        context = QuestionContext(
            topic=topic,
            context_text="\n\n".join(context_texts),
            source_chunks=source_info,
            themes=themes_filter or [],
            confidence_score=confidence,
            metadata={
                "chunks_count": len(relevant_chunks),
                "total_chars": sum(len(chunk.content) for chunk in relevant_chunks)
            }
        )
        
        logger.debug(f"Context created with {len(relevant_chunks)} chunks, confidence: {confidence:.2f}")
        return context
        
    def search_documents(
        self, 
        query: str, 
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Search documents with optional filters.
        
        Args:
            query: Search query
            filters: Optional search filters
            
        Returns:
            List of search results with metadata
        """
        filters = filters or {}
        
        # Get relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(
            query,
            document_ids=filters.get("document_ids"),
            themes_filter=filters.get("themes"),
            limit=filters.get("limit", self.max_results * 2)
        )
        
        # Group results by document
        doc_results = {}
        for chunk in relevant_chunks:
            doc_id = chunk.document_id
            similarity = chunk.metadata.get("similarity_score", 0.0)
            
            if doc_id not in doc_results:
                doc_results[doc_id] = {
                    "document_id": doc_id,
                    "max_similarity": similarity,
                    "total_similarity": similarity,
                    "chunk_count": 1,
                    "relevant_chunks": [chunk.chunk_id],
                    "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                }
            else:
                doc_results[doc_id]["max_similarity"] = max(doc_results[doc_id]["max_similarity"], similarity)
                doc_results[doc_id]["total_similarity"] += similarity
                doc_results[doc_id]["chunk_count"] += 1
                doc_results[doc_id]["relevant_chunks"].append(chunk.chunk_id)
                
        # Sort by relevance
        results = list(doc_results.values())
        results.sort(key=lambda x: x["max_similarity"], reverse=True)
        
        return results[:filters.get("limit", self.max_results)]
        
    def get_document_summary(self, document_id: str) -> dict[str, Any]:
        """
        Get summary information about a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document summary
        """
        if document_id not in self.document_chunks:
            return {"error": "Document not found"}
            
        chunks = self.document_chunks[document_id]
        total_text = " ".join(chunk.content for chunk in chunks)
        
        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "total_characters": len(total_text),
            "word_count": len(total_text.split()),
            "themes": list(set(theme for chunk in chunks for theme in chunk.themes)),
            "first_chunk_preview": chunks[0].content[:200] + "..." if chunks and len(chunks[0].content) > 200 else chunks[0].content if chunks else ""
        }
        
    def save_index(self, file_path: Path) -> None:
        """
        Save the document index to a file.
        
        Args:
            file_path: Path to save the index
        """
        try:
            # Convert DocumentChunk objects to dictionaries for JSON serialization
            serializable_data = {}
            for doc_id, chunks in self.document_chunks.items():
                serializable_data[doc_id] = [chunk.model_dump() for chunk in chunks]
                
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"RAG index saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save RAG index: {e}")
            raise RAGEngineError(f"Failed to save index: {e}")
            
    def load_index(self, file_path: Path) -> None:
        """
        Load the document index from a file.
        
        Args:
            file_path: Path to load the index from
        """
        try:
            if not file_path.exists():
                logger.warning(f"Index file not found: {file_path}")
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert dictionaries back to DocumentChunk objects
            self.document_chunks = {}
            for doc_id, chunks_data in data.items():
                chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
                self.document_chunks[doc_id] = chunks
                
            logger.info(f"RAG index loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load RAG index: {e}")
            raise RAGEngineError(f"Failed to load index: {e}")


class ChromaDBRAGEngine:
    """
    RAG engine using ChromaDB for persistent vector storage.
    
    This provides persistent storage of document chunks with semantic similarity search
    using ChromaDB vector database.
    """
    
    def __init__(self):
        """Initialize ChromaDB RAG engine."""
        if not CHROMADB_AVAILABLE:
            raise RAGEngineError("ChromaDB not available. Install with: pip install chromadb")
            
        # Initialize ChromaDB client
        self.db_path = Path(settings.vector_store.chroma_db_path)
        self.collection_name = settings.vector_store.chroma_collection_name
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
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
    
    def add_document(
        self, 
        document_id: str, 
        text: str, 
        metadata: Dict[str, Any] | None = None,
        themes: List[str] | None = None
    ) -> None:
        """
        Add a document to ChromaDB with persistence.
        
        Args:
            document_id: Unique document identifier
            text: Document text
            metadata: Document metadata
            themes: Document themes
        """
        logger.info(f"Adding document to ChromaDB RAG engine: {document_id}")
        
        # Split text into chunks
        chunks = self._create_chunks(text)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk_text)
            
            # Prepare metadata
            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": i,
                "word_count": len(chunk_text.split()),
                "chunk_size": len(chunk_text),
                "themes": ",".join(themes) if themes else ""
            }
            
            # Add original metadata if present
            if metadata:
                # Convert complex objects to strings for ChromaDB
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        chunk_metadata[f"meta_{key}"] = json.dumps(value)
                    else:
                        chunk_metadata[f"meta_{key}"] = str(value)
            
            metadatas.append(chunk_metadata)
        
        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks for document {document_id} to ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            raise RAGEngineError(f"Failed to add document to ChromaDB: {e}")
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks (same logic as SimpleRAGEngine).
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        chunk_size = settings.processing.default_chunk_size
        chunk_overlap = settings.processing.default_chunk_overlap
        
        while start < text_length:
            # Calculate end position
            end = start + chunk_size
            
            # If we're not at the end, try to break at a sentence boundary
            if end < text_length:
                # Look for sentence ending within the last 200 characters
                search_start = max(start + chunk_size - 200, start)
                sentence_endings = ['.', '!', '?', '\n\n']
                
                best_break = -1
                for ending in sentence_endings:
                    pos = text.rfind(ending, search_start, end)
                    if pos > best_break:
                        best_break = pos
                        
                if best_break > start:
                    end = best_break + 1
                else:
                    # Fall back to word boundary
                    word_break = text.rfind(' ', start, end)
                    if word_break > start:
                        end = word_break
                        
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Move start position with overlap
            start = max(start + chunk_size - chunk_overlap, end)
            
        return chunks
    
    def get_question_context(
        self,
        topic: str,
        document_ids: Optional[List[str]] = None,
        themes_filter: Optional[List[str]] = None,
        context_size: int = 3
    ) -> QuestionContext:
        """
        Get context for question generation using ChromaDB similarity search.
        
        Args:
            topic: Topic or query for context retrieval
            document_ids: Filter by specific document IDs
            themes_filter: Filter by specific themes
            context_size: Number of chunks to include
            
        Returns:
            QuestionContext with relevant chunks
        """
        try:
            logger.debug(f"Getting question context for topic: {topic}")
            
            # Build where clause for filtering
            where_clause = {}
            if document_ids:
                where_clause["document_id"] = {"$in": document_ids}
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[topic],
                n_results=context_size,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to source chunks info
            source_chunks = []
            context_texts = []
            total_similarity = 0.0
            
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                ):
                    similarity_score = 1 - distance  # Convert distance to similarity
                    total_similarity += similarity_score
                    
                    context_texts.append(doc)
                    source_chunks.append({
                        "chunk_id": f"{metadata.get('document_id', 'unknown')}_chunk_{metadata.get('chunk_index', 0)}",
                        "document_id": metadata.get('document_id', 'unknown'),
                        "similarity": similarity_score
                    })
            
            # Calculate confidence
            confidence_score = total_similarity / len(source_chunks) if source_chunks else 0.0
            
            return QuestionContext(
                topic=topic,
                context_text="\n\n".join(context_texts),
                source_chunks=source_chunks,
                themes=themes_filter or [],
                confidence_score=confidence_score,
                metadata={
                    "chunks_count": len(source_chunks),
                    "total_chars": sum(len(text) for text in context_texts),
                    "retrieval_method": "chromadb_similarity"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get context from ChromaDB: {e}")
            # Return empty context on error
            return QuestionContext(
                topic=topic,
                context_text="",
                source_chunks=[],
                themes=themes_filter or [],
                confidence_score=0.0,
                metadata={"error": str(e), "retrieval_method": "chromadb_error"}
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
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents stored in ChromaDB."""
        try:
            # Get all documents metadata
            results = self.collection.get(include=['metadatas'])
            
            # Group by document_id
            documents = {}
            for metadata in results['metadatas']:
                if metadata and 'document_id' in metadata:
                    doc_id = metadata['document_id']
                    if doc_id not in documents:
                        documents[doc_id] = {
                            "document_id": doc_id,
                            "chunk_count": 0,
                            "themes": set(),
                            "word_count": 0
                        }
                    
                    documents[doc_id]["chunk_count"] += 1
                    documents[doc_id]["word_count"] += metadata.get("word_count", 0)
                    
                    # Extract themes
                    themes_str = metadata.get("themes", "")
                    if themes_str:
                        documents[doc_id]["themes"].update(themes_str.split(","))
            
            # Convert sets to lists
            for doc in documents.values():
                doc["themes"] = list(doc["themes"])
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a specific document from ChromaDB.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if deletion successful
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=['metadatas']
            )
            
            if not results['ids']:
                logger.warning(f"No chunks found for document {document_id} in ChromaDB")
                return True  # Nothing to delete is considered success
            
            # Delete all chunks for this document
            self.collection.delete(
                where={"document_id": document_id}
            )
            
            deleted_count = len(results['ids'])
            logger.info(f"Deleted {deleted_count} chunks for document {document_id} from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id} from ChromaDB: {e}")
            return False


# Global RAG engine instances
_simple_rag_engine: SimpleRAGEngine | None = None
_chromadb_rag_engine: ChromaDBRAGEngine | None = None


def get_rag_engine(engine_type: str | None = None):
    """
    Get the RAG engine instance based on configuration.
    
    Args:
        engine_type: Override engine type ('simple' or 'chromadb')
        
    Returns:
        RAG engine instance (SimpleRAGEngine or ChromaDBRAGEngine)
    """
    global _simple_rag_engine, _chromadb_rag_engine
    
    # Use provided type or get from settings
    engine_type = engine_type or settings.vector_store.engine_type
    
    if engine_type.lower() == "chromadb":
        if _chromadb_rag_engine is None:
            if not CHROMADB_AVAILABLE:
                logger.warning("ChromaDB not available, falling back to SimpleRAGEngine")
                engine_type = "simple"
            else:
                try:
                    _chromadb_rag_engine = ChromaDBRAGEngine()
                    logger.info("Initialized ChromaDB RAG engine")
                except Exception as e:
                    logger.error(f"Failed to initialize ChromaDB RAG engine: {e}")
                    logger.warning("Falling back to SimpleRAGEngine")
                    engine_type = "simple"
        
        if engine_type.lower() == "chromadb":
            return _chromadb_rag_engine
    
    # Default to SimpleRAGEngine
    if _simple_rag_engine is None:
        _simple_rag_engine = SimpleRAGEngine()
        logger.info("Initialized Simple RAG engine")
    
    return _simple_rag_engine


def switch_rag_engine(engine_type: str) -> bool:
    """
    Switch to a different RAG engine type.
    
    Args:
        engine_type: Engine type to switch to ('simple' or 'chromadb')
        
    Returns:
        True if successfully switched
    """
    try:
        engine = get_rag_engine(engine_type)
        logger.info(f"Successfully switched to {engine_type} RAG engine")
        return True
    except Exception as e:
        logger.error(f"Failed to switch to {engine_type} RAG engine: {e}")
        return False


# Convenience functions
def add_document_to_rag(
    document_id: str, 
    text: str, 
    metadata: dict[str, Any] | None = None,
    themes: list[str] | None = None
) -> None:
    """Add a document to the RAG engine."""
    engine = get_rag_engine()
    engine.add_document(document_id, text, metadata, themes)


def get_question_context(
    topic: str, 
    document_ids: list[str] | None = None,
    themes_filter: list[str] | None = None
) -> QuestionContext:
    """Get context for question generation."""
    engine = get_rag_engine()
    return engine.get_question_context(topic, document_ids, themes_filter)