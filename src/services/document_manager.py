"""
QCM Generator Pro - Document Manager Service

This module handles document persistence, theme management, and integration
between RAG engines and the SQLite database for document reusability.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session

from ..core.config import settings
from ..models.database import Document, DocumentTheme, DocumentChunk, Base
from ..models.schemas import DocumentCreate, ThemeDetection, ProcessingConfig
from .pdf_processor import process_pdf
from .theme_extractor import extract_document_themes
from .rag_engine import get_rag_engine

logger = logging.getLogger(__name__)


class DocumentManagerError(Exception):
    """Exception raised when document management operations fail."""
    pass


class DocumentManager:
    """
    Service for managing document persistence and reusability.
    
    Handles document storage in both SQLite database and RAG engines,
    enabling document and theme reusability across sessions.
    """
    
    def __init__(self):
        """Initialize document manager."""
        # Setup database connection
        self.engine = create_engine(
            settings.database.url,
            echo=settings.database.echo,
            pool_pre_ping=settings.database.pool_pre_ping,
            pool_recycle=settings.database.pool_recycle
        )
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    async def process_and_store_document(
        self,
        file_path: Path,
        config: Optional[ProcessingConfig] = None,
        store_in_rag: bool = True
    ) -> Document:
        """
        Process a PDF document and store it persistently.
        
        Args:
            file_path: Path to PDF file
            config: Processing configuration
            store_in_rag: Whether to store in RAG engine
            
        Returns:
            Document database model
            
        Raises:
            DocumentManagerError: If processing or storage fails
        """
        try:
            logger.info(f"Processing and storing document: {file_path}")
            
            # Process PDF
            document_data = process_pdf(file_path, config)
            
            # Extract themes
            themes = await extract_document_themes(
                document_data.full_text,
                metadata=document_data.doc_metadata,
                pages_data=document_data.pages_data,
                language=document_data.language
            )
            
            # Store in database
            with self.get_session() as session:
                # Create document record (let DB generate ID)
                document = Document(
                    filename=document_data.filename,
                    file_path=document_data.file_path,
                    upload_date=datetime.utcnow(),
                    total_pages=document_data.total_pages,
                    language=document_data.language,
                    processing_status=document_data.processing_status.value if hasattr(document_data.processing_status, 'value') else str(document_data.processing_status),
                    doc_metadata=document_data.doc_metadata
                )
                session.add(document)
                session.flush()  # Flush to get the generated ID
                
                document_id = document.id
                
                # Store themes
                for theme in themes:
                    doc_theme = DocumentTheme(
                        document_id=document_id,
                        theme_name=theme.theme_name,
                        start_page=theme.start_page,
                        end_page=theme.end_page,
                        confidence_score=theme.confidence_score,
                        keywords=theme.keywords
                    )
                    session.add(doc_theme)
                
                # Store chunks
                for i, chunk_text in enumerate(document_data.chunks):
                    doc_chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_text=chunk_text,
                        chunk_order=i,
                        word_count=len(chunk_text.split()),
                        char_count=len(chunk_text)
                    )
                    session.add(doc_chunk)
                
                session.commit()
                session.refresh(document)
            
            # Store in RAG engine if requested
            if store_in_rag:
                try:
                    rag_engine = get_rag_engine()
                    theme_names = [theme.theme_name for theme in themes]
                    rag_engine.add_document(
                        document_id=str(document_id),  # Convert to string for RAG
                        text=document_data.full_text,
                        metadata=document_data.doc_metadata,
                        themes=theme_names
                    )
                    logger.info(f"Document {document_id} added to RAG engine")
                except Exception as e:
                    logger.error(f"Failed to add document to RAG engine: {e}")
                    # Don't fail the entire operation for RAG errors
            
            logger.info(f"Document processed and stored successfully: {document_id}")
            return document
            
        except Exception as e:
            logger.error(f"Failed to process and store document: {e}")
            raise DocumentManagerError(f"Document processing failed: {e}")
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all stored documents with their metadata.
        
        Returns:
            List of document information dictionaries
        """
        try:
            with self.get_session() as session:
                stmt = select(Document).order_by(Document.upload_date.desc())
                documents = session.execute(stmt).scalars().all()
                
                result = []
                for doc in documents:
                    # Get themes for this document
                    themes_stmt = select(DocumentTheme).where(DocumentTheme.document_id == doc.id)
                    themes = session.execute(themes_stmt).scalars().all()
                    
                    # Get chunk count
                    chunks_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
                    chunk_count = len(session.execute(chunks_stmt).scalars().all())
                    
                    doc_info = {
                        "id": doc.id,
                        "filename": doc.filename,
                        "upload_date": doc.upload_date.isoformat(),
                        "total_pages": doc.total_pages,
                        "language": doc.language,
                        "processing_status": doc.processing_status,
                        "chunk_count": chunk_count,
                        "themes": [
                            {
                                "name": theme.theme_name,
                                "confidence": theme.confidence_score,
                                "keywords": theme.keywords
                            }
                            for theme in themes
                        ]
                    }
                    result.append(doc_info)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a specific document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document model or None if not found
        """
        try:
            with self.get_session() as session:
                stmt = select(Document).where(Document.id == document_id)
                return session.execute(stmt).scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def get_document_themes(self, document_id: str) -> List[DocumentTheme]:
        """
        Get themes for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of document themes
        """
        try:
            with self.get_session() as session:
                stmt = select(DocumentTheme).where(
                    DocumentTheme.document_id == document_id
                ).order_by(DocumentTheme.confidence_score.desc())
                return session.execute(stmt).scalars().all()
        except Exception as e:
            logger.error(f"Failed to get themes for document {document_id}: {e}")
            return []

    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of document chunks with metadata
        """
        try:
            with self.get_session() as session:
                stmt = select(DocumentChunk).where(
                    DocumentChunk.document_id == document_id
                ).order_by(DocumentChunk.chunk_order)
                chunks = session.execute(stmt).scalars().all()
                
                result = []
                for chunk in chunks:
                    chunk_info = {
                        "id": chunk.id,
                        "chunk_order": chunk.chunk_order,
                        "chunk_text": chunk.chunk_text,
                        "word_count": chunk.word_count,
                        "char_count": chunk.char_count,
                        "metadata": {
                            "start_char": getattr(chunk, 'start_char', None),
                            "end_char": getattr(chunk, 'end_char', None),
                            "page_number": getattr(chunk, 'page_number', None)
                        }
                    }
                    result.append(chunk_info)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []
    
    def get_all_themes(self) -> List[Dict[str, Any]]:
        """
        Get all unique themes across all documents.
        
        Returns:
            List of theme information with usage statistics
        """
        try:
            with self.get_session() as session:
                stmt = select(DocumentTheme).order_by(DocumentTheme.confidence_score.desc())
                all_themes = session.execute(stmt).scalars().all()
                
                # Group themes by name
                theme_groups = {}
                for theme in all_themes:
                    name = theme.theme_name
                    if name not in theme_groups:
                        theme_groups[name] = {
                            "name": name,
                            "document_count": 0,
                            "avg_confidence": 0.0,
                            "keywords": set(),
                            "documents": []
                        }
                    
                    theme_groups[name]["document_count"] += 1
                    theme_groups[name]["avg_confidence"] += theme.confidence_score
                    theme_groups[name]["keywords"].update(theme.keywords or [])
                    theme_groups[name]["documents"].append(theme.document_id)
                
                # Calculate averages and convert sets to lists
                result = []
                for theme_info in theme_groups.values():
                    theme_info["avg_confidence"] /= theme_info["document_count"]
                    theme_info["keywords"] = list(theme_info["keywords"])
                    result.append(theme_info)
                
                # Sort by usage (document count) and confidence
                result.sort(key=lambda x: (x["document_count"], x["avg_confidence"]), reverse=True)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get all themes: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated data.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            with self.get_session() as session:
                # Delete document (cascades to themes and chunks)
                stmt = select(Document).where(Document.id == document_id)
                document = session.execute(stmt).scalar_one_or_none()
                
                if document:
                    # Store file path before deletion
                    file_path = Path(document.file_path)
                    
                    # Delete from database
                    session.delete(document)
                    session.commit()
                    logger.info(f"Document {document_id} deleted from database")
                    
                    # Try to remove from file system
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            logger.info(f"File {file_path} deleted")
                    except Exception as e:
                        logger.warning(f"Failed to delete file: {e}")
                    
                    # Try to remove from RAG engine
                    try:
                        from .rag_engine import get_rag_engine
                        rag_engine = get_rag_engine()
                        doc_id_str = str(document_id)
                        
                        # For SimpleRAGEngine
                        if hasattr(rag_engine, 'document_chunks') and doc_id_str in rag_engine.document_chunks:
                            del rag_engine.document_chunks[doc_id_str]
                            logger.info(f"Document {document_id} removed from SimpleRAGEngine")
                        
                        # For ChromaDBRAGEngine
                        elif hasattr(rag_engine, 'delete_document'):
                            if rag_engine.delete_document(doc_id_str):
                                logger.info(f"Document {document_id} removed from ChromaDBRAGEngine")
                            else:
                                logger.warning(f"Failed to remove document {document_id} from ChromaDBRAGEngine")
                            
                    except Exception as e:
                        logger.warning(f"Failed to remove document from RAG engine: {e}")
                    
                    return True
                else:
                    logger.warning(f"Document {document_id} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False


# Global document manager instance
_document_manager: Optional[DocumentManager] = None


def get_document_manager() -> DocumentManager:
    """Get the global document manager instance."""
    global _document_manager
    if _document_manager is None:
        _document_manager = DocumentManager()
    return _document_manager


# Convenience functions
async def process_and_store_pdf(
    file_path: Path,
    config: Optional[ProcessingConfig] = None
) -> Document:
    """Process and store a PDF document."""
    manager = get_document_manager()
    return await manager.process_and_store_document(file_path, config)


def list_stored_documents() -> List[Dict[str, Any]]:
    """List all stored documents."""
    manager = get_document_manager()
    return manager.list_documents()


def get_stored_document(document_id: str) -> Optional[Document]:
    """Get a stored document by ID."""
    manager = get_document_manager()
    return manager.get_document(document_id)


def get_available_themes() -> List[Dict[str, Any]]:
    """Get all available themes across documents."""
    manager = get_document_manager()
    return manager.get_all_themes()

def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Get chunks for a specific document."""
    manager = get_document_manager()
    return manager.get_document_chunks(document_id)
