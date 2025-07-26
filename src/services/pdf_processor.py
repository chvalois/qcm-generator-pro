"""
QCM Generator Pro - PDF Processing Service

This module handles PDF document processing including text extraction,
metadata parsing, and document chunking for QCM generation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..core.config import settings
from ..models.enums import ProcessingStatus
from ..models.schemas import DocumentCreate, ProcessingConfig

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Exception raised when PDF processing fails."""
    pass


class PDFProcessor:
    """Service for processing PDF documents."""
    
    def __init__(self):
        """Initialize PDF processor."""
        self.max_file_size = settings.processing.max_pdf_size_bytes
        
    def validate_pdf(self, file_path: Path) -> bool:
        """
        Validate PDF file before processing.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            PDFProcessingError: If validation fails
        """
        if not file_path.exists():
            raise PDFProcessingError(f"PDF file not found: {file_path}")
            
        if not file_path.suffix.lower() == '.pdf':
            raise PDFProcessingError(f"File is not a PDF: {file_path}")
            
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise PDFProcessingError(
                f"PDF file too large: {file_size} bytes (max: {self.max_file_size})"
            )
            
        # Try to read the PDF to validate it's not corrupted
        try:
            reader = PdfReader(str(file_path))
            if len(reader.pages) == 0:
                raise PDFProcessingError("PDF has no pages")
        except PdfReadError as e:
            raise PDFProcessingError(f"Invalid or corrupted PDF: {e}")
            
        return True
        
    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            reader = PdfReader(str(file_path))
            metadata = reader.metadata or {}
            
            # Extract standard metadata
            extracted_metadata = {
                "title": getattr(metadata, "title", None) or file_path.stem,
                "author": getattr(metadata, "author", None),
                "subject": getattr(metadata, "subject", None),
                "creator": getattr(metadata, "creator", None),
                "producer": getattr(metadata, "producer", None),
                "creation_date": None,
                "modification_date": None,
                "total_pages": len(reader.pages),
                "file_size_bytes": file_path.stat().st_size,
                "language": "auto-detect",  # To be determined by language detection
            }
            
            # Handle dates
            if hasattr(metadata, "creation_date") and metadata.creation_date:
                extracted_metadata["creation_date"] = metadata.creation_date.isoformat()
                
            if hasattr(metadata, "modification_date") and metadata.modification_date:
                extracted_metadata["modification_date"] = metadata.modification_date.isoformat()
                
            return extracted_metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return {
                "title": file_path.stem,
                "author": None,
                "subject": None,
                "creator": None,
                "producer": None,
                "creation_date": None,
                "modification_date": None,
                "total_pages": 0,
                "file_size_bytes": file_path.stat().st_size,
                "language": "auto-detect",
            }
            
    def extract_text(self, file_path: Path) -> str:
        """
        Extract all text from PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            PDFProcessingError: If text extraction fails
        """
        try:
            reader = PdfReader(str(file_path))
            text_content = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
                    
            if not text_content:
                raise PDFProcessingError("No text could be extracted from PDF")
                
            return "\n\n".join(text_content)
            
        except PdfReadError as e:
            raise PDFProcessingError(f"Failed to read PDF: {e}")
        except Exception as e:
            raise PDFProcessingError(f"Text extraction failed: {e}")
            
    def extract_text_by_pages(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Extract text from PDF with page-level granularity.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of page dictionaries with text and metadata
        """
        try:
            reader = PdfReader(str(file_path))
            pages_data = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    page_data = {
                        "page_number": page_num,
                        "text": page_text.strip(),
                        "char_count": len(page_text.strip()),
                        "word_count": len(page_text.strip().split()) if page_text.strip() else 0,
                    }
                    pages_data.append(page_data)
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num}: {e}")
                    pages_data.append({
                        "page_number": page_num,
                        "text": "",
                        "char_count": 0,
                        "word_count": 0,
                        "error": str(e),
                    })
                    
            return pages_data
            
        except Exception as e:
            raise PDFProcessingError(f"Page-level extraction failed: {e}")
            
    def chunk_text(self, text: str, config: ProcessingConfig | None = None) -> list[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            config: Processing configuration
            
        Returns:
            List of text chunks
        """
        if config is None:
            chunk_size = settings.processing.default_chunk_size
            chunk_overlap = settings.processing.default_chunk_overlap
        else:
            chunk_size = config.chunk_size
            chunk_overlap = config.chunk_overlap
            
        if not text.strip():
            return []
            
        # Simple character-based chunking
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + chunk_size
            
            # If we're not at the end, try to break at a sentence or word boundary
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
        
    def process_document(
        self, 
        file_path: Path, 
        config: ProcessingConfig | None = None
    ) -> DocumentCreate:
        """
        Process PDF document completely.
        
        Args:
            file_path: Path to PDF file
            config: Processing configuration
            
        Returns:
            Document creation schema with all extracted data
            
        Raises:
            PDFProcessingError: If processing fails
        """
        logger.info(f"Starting PDF processing: {file_path}")
        
        # Validate PDF
        self.validate_pdf(file_path)
        
        # Extract metadata
        metadata = self.extract_metadata(file_path)
        
        # Extract text
        full_text = self.extract_text(file_path)
        
        # Extract page-level data
        pages_data = self.extract_text_by_pages(file_path)
        
        # Chunk text for processing
        chunks = self.chunk_text(full_text, config)
        
        # Detect title hierarchy
        title_candidates = []
        try:
            from .title_detector import detect_document_titles
            title_candidates = detect_document_titles(full_text, pages_data)
            logger.info(f"Detected {len(title_candidates)} title candidates")
        except Exception as e:
            logger.warning(f"Title detection failed: {e}")
        
        # Add title information to metadata
        metadata["detected_titles"] = [
            {
                "text": title.text,
                "level": title.level,
                "confidence": title.confidence,
                "page_number": title.page_number,
                "features": title.features
            }
            for title in title_candidates
        ]
        
        # Create document creation schema
        document_data = DocumentCreate(
            filename=file_path.name,
            file_path=str(file_path.absolute()),
            total_pages=metadata["total_pages"],
            language="fr",  # Default, will be detected later
            processing_status=ProcessingStatus.COMPLETED,
            doc_metadata=metadata,
            full_text=full_text,
            pages_data=pages_data,
            chunks=chunks,
        )
        
        logger.info(
            f"PDF processing completed: {file_path} "
            f"({metadata['total_pages']} pages, {len(chunks)} chunks)"
        )
        
        return document_data


# Convenience functions
def process_pdf(file_path: Path, config: ProcessingConfig | None = None) -> DocumentCreate:
    """
    Process a PDF document.
    
    Args:
        file_path: Path to PDF file
        config: Processing configuration
        
    Returns:
        Document creation schema
    """
    processor = PDFProcessor()
    return processor.process_document(file_path, config)


def validate_pdf_file(file_path: Path) -> bool:
    """
    Validate a PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        True if valid
    """
    processor = PDFProcessor()
    return processor.validate_pdf(file_path)