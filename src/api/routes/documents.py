"""
QCM Generator Pro - Document Management API Routes

This module provides FastAPI routes for document upload, processing,
and management operations.
"""

import logging
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ...core.config import settings
from ...models.database import Document as DocumentModel
from ...models.schemas import (
    DocumentCreate,
    DocumentResponse,
    ProcessingConfig,
    SuccessResponse,
)
from ...services.pdf_processor import PDFProcessor
from ...services.rag_engine import add_document_to_rag
from ...services.theme_extractor import extract_document_themes_sync
from ...services.document_manager import get_document_chunks
from ..dependencies import (
    get_db_session,
    get_pdf_processor_service,
    validate_file_upload,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor_service),
    config: ProcessingConfig = None
) -> DocumentResponse:
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to upload
        db: Database session
        pdf_processor: PDF processor service
        config: Processing configuration
        
    Returns:
        Processed document information
        
    Raises:
        HTTPException: If upload or processing fails
    """
    logger.info(f"Starting document upload: {file.filename}")
    
    try:
        # Validate file
        validate_file_upload(file.size or 0, file.content_type or "")
        
        # Ensure upload directory exists
        upload_dir = settings.data_dir / "pdfs"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        logger.info(f"File saved: {file_path}")
        
        # Process document
        document_data = pdf_processor.process_document(file_path, config)
        
        # Extract themes using LLM
        try:
            themes = extract_document_themes_sync(
                text=document_data.full_text,
                metadata=document_data.doc_metadata,
                pages_data=document_data.pages_data
            )
            logger.info(f"Extracted {len(themes)} themes")
        except Exception as e:
            logger.warning(f"Theme extraction failed: {e}")
            themes = []
            
        # Save to database
        db_document = DocumentModel(
            filename=document_data.filename,
            file_path=document_data.file_path,
            total_pages=document_data.total_pages,
            language=document_data.language,
            processing_status=document_data.processing_status,
            doc_metadata=document_data.doc_metadata
        )
        
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Add to RAG engine
        try:
            theme_names = [theme.theme_name for theme in themes]
            add_document_to_rag(
                document_id=str(db_document.id),
                text=document_data.full_text,
                metadata=document_data.doc_metadata,
                themes=theme_names
            )
            logger.info(f"Document added to RAG engine: {db_document.id}")
        except Exception as e:
            logger.warning(f"Failed to add document to RAG: {e}")
            
        # Prepare response
        response = DocumentResponse(
            id=db_document.id,
            filename=db_document.filename,
            file_path=db_document.file_path,
            upload_date=db_document.upload_date,
            total_pages=db_document.total_pages,
            language=db_document.language,
            processing_status=db_document.processing_status,
            doc_metadata=db_document.doc_metadata,
            themes=themes,
            processing_stats={
                "chunks_created": len(document_data.chunks),
                "themes_detected": len(themes),
                "total_characters": len(document_data.full_text),
                "processing_time": "completed"
            }
        )
        
        logger.info(f"Document upload completed: {db_document.id}")
        return response
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        # Clean up file if it was created
        if 'file_path' in locals() and Path(file_path).exists():
            Path(file_path).unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db_session)
) -> List[DocumentResponse]:
    """
    List uploaded documents.
    
    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        db: Database session
        
    Returns:
        List of documents
    """
    logger.debug(f"Listing documents: skip={skip}, limit={limit}")
    
    try:
        documents = db.query(DocumentModel).offset(skip).limit(limit).all()
        
        response_documents = []
        for doc in documents:
            response = DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                file_path=doc.file_path,
                upload_date=doc.upload_date,
                total_pages=doc.total_pages,
                language=doc.language,
                processing_status=doc.processing_status,
                doc_metadata=doc.doc_metadata or {},
                themes=[],  # Could load themes here if needed
                processing_stats={}
            )
            response_documents.append(response)
            
        logger.debug(f"Retrieved {len(response_documents)} documents")
        return response_documents
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db_session)
) -> DocumentResponse:
    """
    Get a specific document by ID.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        Document information
        
    Raises:
        HTTPException: If document not found
    """
    logger.debug(f"Getting document: {document_id}")
    
    try:
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
        response = DocumentResponse(
            id=document.id,
            filename=document.filename,
            file_path=document.file_path,
            upload_date=document.upload_date,
            total_pages=document.total_pages,
            language=document.language,
            processing_status=document.processing_status,
            doc_metadata=document.doc_metadata or {},
            themes=[],  # Could load themes here
            processing_stats={}
        )
        
        logger.debug(f"Retrieved document: {document_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.get("/{document_id}/chunks", response_model=List[dict])
async def get_document_chunks_api(
    document_id: int,
    db: Session = Depends(get_db_session)
) -> List[dict]:
    """
    Get chunks for a specific document.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        List of document chunks with metadata
        
    Raises:
        HTTPException: If document not found
    """
    logger.debug(f"Getting chunks for document: {document_id}")
    
    try:
        # First verify the document exists
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
        # Get chunks using the document manager
        chunks = get_document_chunks(str(document_id))
        
        logger.debug(f"Retrieved {len(chunks)} chunks for document {document_id}")
        return chunks
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunks for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document chunks"
        )


@router.delete("/{document_id}", response_model=SuccessResponse)
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db_session)
) -> SuccessResponse:
    """
    Delete a document and its associated files.
    
    Args:
        document_id: Document ID to delete
        db: Database session
        
    Returns:
        Success response
        
    Raises:
        HTTPException: If document not found or deletion fails
    """
    logger.info(f"Deleting document: {document_id}")
    
    try:
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
        # Delete physical file
        file_path = Path(document.file_path)
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
                
        # Delete from database
        db.delete(document)
        db.commit()
        
        logger.info(f"Document deleted: {document_id}")
        return SuccessResponse(
            message=f"Document {document_id} deleted successfully",
            details={"document_id": document_id, "filename": document.filename}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: int,
    db: Session = Depends(get_db_session),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor_service),
    config: ProcessingConfig = None
) -> DocumentResponse:
    """
    Reprocess an existing document with new configuration.
    
    Args:
        document_id: Document ID to reprocess
        db: Database session
        pdf_processor: PDF processor service
        config: New processing configuration
        
    Returns:
        Updated document information
        
    Raises:
        HTTPException: If document not found or reprocessing fails
    """
    logger.info(f"Reprocessing document: {document_id}")
    
    try:
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
        file_path = Path(document.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document file not found: {file_path}"
            )
            
        # Reprocess document
        document_data = pdf_processor.process_document(file_path, config)
        
        # Extract themes
        try:
            themes = extract_document_themes_sync(
                text=document_data.full_text,
                metadata=document_data.doc_metadata,
                pages_data=document_data.pages_data
            )
        except Exception as e:
            logger.warning(f"Theme extraction failed during reprocessing: {e}")
            themes = []
            
        # Update database
        document.processing_status = document_data.processing_status
        document.doc_metadata = document_data.doc_metadata
        db.commit()
        db.refresh(document)
        
        # Update RAG engine
        try:
            theme_names = [theme.theme_name for theme in themes]
            add_document_to_rag(
                document_id=str(document.id),
                text=document_data.full_text,
                metadata=document_data.doc_metadata,
                themes=theme_names
            )
        except Exception as e:
            logger.warning(f"Failed to update RAG during reprocessing: {e}")
            
        response = DocumentResponse(
            id=document.id,
            filename=document.filename,
            file_path=document.file_path,
            upload_date=document.upload_date,
            total_pages=document.total_pages,
            language=document.language,
            processing_status=document.processing_status,
            doc_metadata=document.doc_metadata,
            themes=themes,
            processing_stats={
                "reprocessed": True,
                "chunks_created": len(document_data.chunks),
                "themes_detected": len(themes)
            }
        )
        
        logger.info(f"Document reprocessed: {document_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reprocess document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document reprocessing failed"
        )