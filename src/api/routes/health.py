"""
QCM Generator Pro - Health and Documentation API Routes

This module provides health check, status, and API documentation endpoints.
"""

import logging
import platform
import sys
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...core.config import settings
from ...core.database import check_database_health
from ...services.llm_manager import get_llm_manager
from ..dependencies import get_db_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "QCM Generator Pro",
        "version": "0.1.0"
    }


@router.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Detailed health check including all services.
    
    Args:
        db: Database session
        
    Returns:
        Comprehensive health status
    """
    logger.debug("Performing detailed health check")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "QCM Generator Pro",
        "version": "0.1.0",
        "checks": {}
    }
    
    # Database health
    try:
        db_health = check_database_health()
        health_status["checks"]["database"] = {
            "status": "healthy" if db_health["is_connected"] else "unhealthy",
            "details": db_health
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
        
    # LLM services health (only test current provider to reduce noise)
    try:
        llm_manager = get_llm_manager()
        current_provider = llm_manager.model_type.value
        
        # Only test the current provider to avoid spam in LangSmith
        llm_health = await llm_manager.test_connection(provider=current_provider)
        
        working_providers = [
            provider for provider, result in llm_health.items()
            if result.get("status") == "success"
        ]
        
        health_status["checks"]["llm"] = {
            "status": "healthy" if working_providers else "unhealthy",
            "working_providers": working_providers,
            "current_provider": current_provider,
            "details": llm_health
        }
        
        if not working_providers:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["checks"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
        
    # File system health
    try:
        data_dir = settings.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = data_dir / ".health_check"
        test_file.write_text("health_check")
        test_file.unlink()
        
        health_status["checks"]["filesystem"] = {
            "status": "healthy",
            "data_directory": str(data_dir),
            "writable": True
        }
    except Exception as e:
        health_status["checks"]["filesystem"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
        
    return health_status


@router.get("/status", response_model=Dict[str, Any])
async def get_system_status() -> Dict[str, Any]:
    """
    Get system status and information.
    
    Returns:
        System status information
    """
    return {
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture(),
            "processor": platform.processor()
        },
        "application": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value,
            "debug_mode": settings.debug
        },
        "configuration": {
            "default_language": settings.default_language.value,
            "supported_languages": [lang.value for lang in settings.supported_languages],
            "database_url": settings.database.url.split("://")[0] + "://***",  # Hide credentials
            "llm_providers": {
                "openai_configured": bool(settings.llm.openai_api_key),
                "anthropic_configured": bool(settings.llm.anthropic_api_key),
                "ollama_configured": bool(settings.llm.ollama_base_url)
            }
        },
        "runtime": {
            "uptime": "runtime_placeholder",  # Could track actual uptime
            "timestamp": datetime.utcnow().isoformat()
        }
    }


@router.get("/info", response_model=Dict[str, Any])
async def get_api_info() -> Dict[str, Any]:
    """
    Get API information and capabilities.
    
    Returns:
        API information
    """
    return {
        "api": {
            "name": "QCM Generator Pro API",
            "version": "1.0.0",
            "description": "Local multilingual QCM generation from PDF documents",
            "documentation_url": "/docs",
            "openapi_url": "/openapi.json"
        },
        "features": {
            "document_processing": {
                "supported_formats": ["PDF"],
                "max_file_size_mb": settings.processing.max_pdf_size_bytes // (1024 * 1024),
                "language_detection": True,
                "theme_extraction": True
            },
            "qcm_generation": {
                "progressive_workflow": True,
                "question_types": ["multiple-choice", "multiple-selection"],
                "difficulty_levels": ["easy", "medium", "hard"],
                "languages": ["fr", "en"],
                "validation": True
            },
            "export_formats": {
                "udemy_csv": "Direct upload to Udemy courses",
                "json": "Complete question data with metadata"
            },
            "llm_integration": {
                "providers": ["OpenAI", "Anthropic", "Ollama"],
                "local_models": True,
                "cloud_apis": True
            }
        },
        "endpoints": {
            "documents": {
                "upload": "POST /api/documents/upload",
                "list": "GET /api/documents/",
                "get": "GET /api/documents/{id}",
                "delete": "DELETE /api/documents/{id}"
            },
            "generation": {
                "start": "POST /api/generation/start",
                "status": "GET /api/generation/sessions/{session_id}/status",
                "questions": "GET /api/generation/sessions/{session_id}/questions",
                "validate": "POST /api/generation/questions/{question_id}/validate"
            },
            "export": {
                "export": "POST /api/export/{session_id}",
                "download": "GET /api/export/download/{filename}",
                "formats": "GET /api/export/formats"
            }
        },
        "limits": {
            "max_questions_per_session": settings.generation.max_questions_per_session,
            "max_file_size_mb": settings.processing.max_pdf_size_bytes // (1024 * 1024),
            "concurrent_sessions": settings.ui.concurrent_sessions
        }
    }


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get basic usage metrics.
    
    Args:
        db: Database session
        
    Returns:
        Usage metrics
    """
    try:
        from ...models.database import Document as DocumentModel
        from ...models.database import GenerationSession as GenerationSessionModel
        from ...models.database import Question as QuestionModel
        
        # Get basic counts
        total_documents = db.query(DocumentModel).count()
        total_sessions = db.query(GenerationSessionModel).count()
        total_questions = db.query(QuestionModel).count()
        
        # Get recent activity (last 7 days)
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        recent_documents = db.query(DocumentModel).filter(
            DocumentModel.upload_date >= week_ago
        ).count()
        
        recent_sessions = db.query(GenerationSessionModel).filter(
            GenerationSessionModel.created_at >= week_ago
        ).count()
        
        return {
            "totals": {
                "documents": total_documents,
                "generation_sessions": total_sessions,
                "questions_generated": total_questions
            },
            "recent_activity": {
                "period": "last_7_days",
                "documents_uploaded": recent_documents,
                "sessions_created": recent_sessions
            },
            "averages": {
                "questions_per_session": round(total_questions / max(total_sessions, 1), 2)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@router.get("/version", response_model=Dict[str, str])
async def get_version() -> Dict[str, str]:
    """
    Get version information.
    
    Returns:
        Version details
    """
    return {
        "version": "0.1.0",
        "api_version": "1.0.0",
        "build_date": "2024-01-01",  # Would be set during build
        "git_commit": "dev",  # Would be set during build
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }