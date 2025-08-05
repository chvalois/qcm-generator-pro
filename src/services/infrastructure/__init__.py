"""Infrastructure services.

This module contains infrastructure-level services including
vector stores, progress tracking, and other foundational components.
"""

from .rag_engine import (
    get_rag_engine,
    add_document_to_rag,
    get_question_context,
    switch_rag_engine,
)
from .progress_tracker import (
    get_progress_tracker,
    get_progress_state,
    start_progress_session,
    update_progress,
    increment_progress,
    complete_progress_session,
    fail_progress_session,
)

__all__ = [
    # RAG Engine
    "get_rag_engine",
    "add_document_to_rag",
    "get_question_context",
    "switch_rag_engine",
    # Progress Tracking
    "get_progress_tracker",
    "get_progress_state",
    "start_progress_session",
    "update_progress",
    "increment_progress", 
    "complete_progress_session",
    "fail_progress_session",
]