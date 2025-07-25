"""
QCM Generator Pro - FastAPI Dependencies

This module provides dependency injection for FastAPI routes,
including database sessions and service instances.
"""

import logging
from typing import Any

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ..core.database import get_async_db_session, get_db_session as get_database_session
from ..services.llm_manager import get_llm_manager
from ..services.pdf_processor import PDFProcessor
from ..services.qcm_generator import get_qcm_generator
from ..services.rag_engine import get_rag_engine
from ..services.theme_extractor import LLMThemeExtractor
from ..services.validator import get_question_validator

logger = logging.getLogger(__name__)


# Database Dependencies
def get_db_session() -> Session:
    """
    Get database session dependency.
    
    Returns:
        Database session
    """
    try:
        session = get_database_session()
        yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )
    finally:
        session.close()


async def get_async_db_session() -> AsyncSession:
    """
    Get async database session dependency.
    
    Returns:
        Async database session
    """
    try:
        session = get_async_db_session()
        yield session
    except Exception as e:
        logger.error(f"Async database session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )
    finally:
        await session.close()


# Service Dependencies
def get_pdf_processor_service() -> PDFProcessor:
    """
    Get PDF processor service dependency.
    
    Returns:
        PDF processor instance
    """
    return PDFProcessor()


def get_theme_extractor_service() -> LLMThemeExtractor:
    """
    Get theme extractor service dependency.
    
    Returns:
        Theme extractor instance
    """
    return LLMThemeExtractor()


def get_llm_manager_service():
    """
    Get LLM manager service dependency.
    
    Returns:
        LLM manager instance
    """
    return get_llm_manager()


def get_rag_engine_service():
    """
    Get RAG engine service dependency.
    
    Returns:
        RAG engine instance
    """
    return get_rag_engine()


def get_qcm_generator_service():
    """
    Get QCM generator service dependency.
    
    Returns:
        QCM generator instance
    """
    return get_qcm_generator()


def get_validator_service():
    """
    Get question validator service dependency.
    
    Returns:
        Question validator instance
    """
    return get_question_validator()


# Authentication Dependencies (placeholder for future implementation)
async def get_current_user() -> dict[str, Any]:
    """
    Get current user dependency (placeholder).
    
    Returns:
        User information
        
    Note:
        This is a placeholder for future authentication implementation.
        Currently returns a default user.
    """
    return {
        "id": "default_user",
        "username": "developer", 
        "role": "admin"
    }


def verify_api_key(api_key: str = None) -> bool:
    """
    Verify API key dependency (placeholder).
    
    Args:
        api_key: API key to verify
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If API key is invalid
        
    Note:
        This is a placeholder for future API key authentication.
    """
    # For now, accept any key or no key
    return True


# Rate Limiting Dependency (placeholder)
async def rate_limit_dependency() -> None:
    """
    Rate limiting dependency (placeholder).
    
    Note:
        This is a placeholder for future rate limiting implementation.
    """
    pass


# Request Validation Dependencies
def validate_file_upload(file_size: int, file_type: str) -> bool:
    """
    Validate file upload dependency.
    
    Args:
        file_size: File size in bytes
        file_type: File MIME type
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If file is invalid
    """
    # Maximum file size (50MB)
    max_size = 50 * 1024 * 1024
    
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
        )
        
    # Allowed file types
    allowed_types = [
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "applications/vnd.pdf",
        "text/pdf",
        "text/x-pdf"
    ]
    
    if file_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported"
        )
        
    return True


# Service Health Check Dependencies
async def check_llm_health():
    """
    Check LLM service health dependency.
    
    Returns:
        Health status
        
    Raises:
        HTTPException: If LLM service is unavailable
    """
    try:
        llm_manager = get_llm_manager()
        health_results = await llm_manager.test_connection()
        
        # Check if at least one provider is working
        working_providers = [
            provider for provider, result in health_results.items()
            if result.get("status") == "success"
        ]
        
        if not working_providers:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No LLM providers available"
            )
            
        return {
            "status": "healthy",
            "providers": working_providers,
            "details": health_results
        }
        
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service health check failed"
        )


# Common Response Headers
def add_response_headers():
    """
    Add common response headers dependency.
    
    Returns:
        Headers dictionary
    """
    return {
        "X-API-Version": "1.0.0",
        "X-Service": "QCM-Generator-Pro"
    }


# Dependency combinations for common use cases
def get_document_processing_deps():
    """
    Get all dependencies needed for document processing.
    
    Returns:
        Dependency tuple
    """
    return Depends(get_db_session), Depends(get_pdf_processor_service)


def get_qcm_generation_deps():
    """
    Get all dependencies needed for QCM generation.
    
    Returns:
        Dependency tuple
    """
    return (
        Depends(get_db_session),
        Depends(get_qcm_generator_service),
        Depends(get_rag_engine_service),
        Depends(get_validator_service)
    )