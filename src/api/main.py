"""
QCM Generator Pro - Main FastAPI Application

This module creates and configures the main FastAPI application
with all routes, middleware, and dependencies.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from ..core.config import settings
from ..core.database import init_database
from .routes import documents, export, generation, health

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level.value.upper()),
    format=settings.logging.format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting QCM Generator Pro API")
    
    try:
        # Initialize database
        init_database()
        logger.info("Database initialized successfully")
        
        # Ensure data directories exist
        settings.ensure_directories()
        logger.info("Data directories created")
        
        # Log configuration
        logger.info(f"Environment: {settings.environment.value}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"Default language: {settings.default_language.value}")
        
        # Test LLM connectivity (non-blocking)
        try:
            from ..services.llm_manager import get_llm_manager
            llm_manager = get_llm_manager()
            health_results = await llm_manager.test_connection()
            working_providers = [
                provider for provider, result in health_results.items()
                if result.get("status") == "success"
            ]
            if working_providers:
                logger.info(f"LLM providers available: {working_providers}")
            else:
                logger.warning("No LLM providers available - some features may be limited")
        except Exception as e:
            logger.warning(f"LLM connectivity check failed: {e}")
            
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
        
    # Shutdown
    logger.info("Shutting down QCM Generator Pro API")


# Create FastAPI application
app = FastAPI(
    title="QCM Generator Pro API",
    description="Local multilingual QCM generation from PDF documents with LLM integration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None
)


# Security Middleware
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", settings.host]
    )


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=settings.security.cors_allow_credentials,
    allow_methods=settings.security.cors_allow_methods,
    allow_headers=settings.security.cors_allow_headers,
)


# Custom Exception Handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Any) -> JSONResponse:
    """Handle 404 errors with custom response."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Not Found",
            "message": f"The requested resource {request.url.path} was not found",
            "status_code": 404
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Any) -> JSONResponse:
    """Handle 500 errors with custom response."""
    logger.error(f"Internal server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500
        }
    )


# Request/Response Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log HTTP requests and responses."""
    start_time = __import__("time").time()
    
    # Log request
    logger.debug(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = __import__("time").time() - start_time
    logger.debug(
        f"Response: {response.status_code} "
        f"({process_time:.3f}s) {request.method} {request.url.path}"
    )
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-API-Version"] = "1.0.0"
    
    return response


# Rate Limiting Middleware (placeholder)
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Basic rate limiting middleware (placeholder)."""
    # In production, implement proper rate limiting
    # For now, just pass through
    return await call_next(request)


# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information.
    
    Returns:
        API welcome information
    """
    return {
        "message": "Welcome to QCM Generator Pro API",
        "version": "1.0.0",
        "description": "Local multilingual QCM generation from PDF documents",
        "documentation": "/docs" if settings.debug else "Documentation disabled in production",
        "health_check": "/health",
        "features": [
            "PDF document processing",
            "LLM-based theme extraction", 
            "Progressive QCM generation (1→5→all)",
            "Multi-provider LLM support",
            "Question validation",
            "Export to Udemy CSV format"
        ],
        "endpoints": {
            "documents": "/api/documents",
            "generation": "/api/generation", 
            "export": "/api/export",
            "health": "/health"
        }
    }


# Include routers
app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(generation.router, prefix="/api")
app.include_router(export.router, prefix="/api")


# Additional utility endpoints
@app.get("/api", response_model=Dict[str, Any])
async def api_info() -> Dict[str, Any]:
    """
    API information endpoint.
    
    Returns:
        API structure and endpoints
    """
    return {
        "api_version": "1.0.0",
        "service": "QCM Generator Pro",
        "endpoints": {
            "health": {
                "basic": "GET /api/health",
                "detailed": "GET /api/health/detailed",
                "status": "GET /api/status",
                "info": "GET /api/info",
                "metrics": "GET /api/metrics"
            },
            "documents": {
                "upload": "POST /api/documents/upload",
                "list": "GET /api/documents/",
                "get": "GET /api/documents/{id}",
                "delete": "DELETE /api/documents/{id}",
                "reprocess": "POST /api/documents/{id}/reprocess"
            },
            "generation": {
                "start": "POST /api/generation/start",
                "session_questions": "GET /api/generation/sessions/{session_id}/questions",
                "session_status": "GET /api/generation/sessions/{session_id}/status",
                "validate_question": "POST /api/generation/questions/{question_id}/validate",
                "batch_validate": "POST /api/generation/questions/batch-validate"
            },
            "export": {
                "export_session": "POST /api/export/{session_id}",
                "download": "GET /api/export/download/{filename}",
                "formats": "GET /api/export/formats",
                "cleanup": "DELETE /api/export/cleanup"
            }
        },
        "authentication": "Not implemented (development mode)",
        "rate_limiting": "Basic implementation",
        "cors": "Configured for development"
    }


# Error handling for startup
@app.get("/startup-check")
async def startup_check() -> Dict[str, Any]:
    """
    Check if application started successfully.
    
    Returns:
        Startup status
    """
    try:
        # Basic checks
        from ..core.database import check_database_health
        db_health = check_database_health()
        
        return {
            "status": "ok",
            "database": db_health["is_connected"],
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            "environment": settings.environment.value
        }
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": __import__("datetime").datetime.utcnow().isoformat()
            }
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.debug,
        log_level=settings.logging.level.value.lower(),
        workers=1  # Single worker for development
    )