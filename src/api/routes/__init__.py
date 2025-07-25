"""
QCM Generator Pro - API Routes Package

This package contains all FastAPI route modules.
"""

from . import documents, export, generation, health

__all__ = ["documents", "generation", "export", "health"]