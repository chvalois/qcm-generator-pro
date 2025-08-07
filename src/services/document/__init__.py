"""Document processing services.

This module contains services responsible for document processing,
including PDF text extraction, theme detection, and document lifecycle management.
"""

from .document_manager import (
    get_document_manager,
    get_stored_document,
    list_stored_documents,
    get_available_themes,
    get_document_chunks,
)
from .pdf_processor import process_pdf, validate_pdf_file
from .theme_extractor import extract_document_themes_sync
from .title_detector import detect_document_titles, build_chunk_title_hierarchies

__all__ = [
    # Document Manager
    "get_document_manager",
    "get_stored_document", 
    "list_stored_documents",
    "get_available_themes",
    "get_document_chunks",
    # PDF Processing
    "process_pdf",
    "validate_pdf_file",
    # Theme Extraction
    "extract_document_themes_sync",
    # Title Detection
    "detect_document_titles",
    "build_chunk_title_hierarchies",
]