"""
Session State Management

Centralized management of Streamlit session state for the QCM Generator Pro application.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
import uuid


class SessionStateManager:
    """Manages Streamlit session state variables."""
    
    # Default session state keys and their initial values
    DEFAULT_STATE = {
        "current_session_id": None,
        "generated_questions": [],
        "processed_documents": {},
        "selected_themes": [],
        "current_config": None,
        "progress_session_id": None,
        "validation_step": 0,
        "export_ready": False,
    }
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize session state with default values if not already set."""
        for key, default_value in cls.DEFAULT_STATE.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a session state value.
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
            
        Returns:
            Session state value or default
        """
        return st.session_state.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Set a session state value.
        
        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
    
    @classmethod
    def clear(cls, key: str) -> None:
        """
        Clear a session state value.
        
        Args:
            key: Session state key to clear
        """
        if key in st.session_state:
            del st.session_state[key]
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all session state to defaults."""
        for key in list(st.session_state.keys()):
            if key in cls.DEFAULT_STATE:
                st.session_state[key] = cls.DEFAULT_STATE[key]
    
    # Convenience methods for common session state operations
    
    @classmethod
    def get_current_session_id(cls) -> Optional[str]:
        """Get current session ID."""
        return cls.get("current_session_id")
    
    @classmethod
    def create_new_session(cls) -> str:
        """Create a new session ID and set it as current."""
        session_id = str(uuid.uuid4())
        cls.set("current_session_id", session_id)
        return session_id
    
    @classmethod
    def get_or_create_session_id(cls) -> str:
        """Get current session ID or create a new one if none exists."""
        session_id = cls.get_current_session_id()
        if not session_id:
            session_id = cls.create_new_session()
        return session_id
    
    @classmethod
    def get_generated_questions(cls) -> List[Dict[str, Any]]:
        """Get generated questions list."""
        return cls.get("generated_questions", [])
    
    @classmethod
    def set_generated_questions(cls, questions: List[Dict[str, Any]]) -> None:
        """Set the entire generated questions list."""
        cls.set("generated_questions", questions)
    
    @classmethod
    def add_generated_question(cls, question: Dict[str, Any]) -> None:
        """Add a generated question to the list."""
        questions = cls.get_generated_questions()
        questions.append(question)
        cls.set("generated_questions", questions)
    
    @classmethod
    def clear_generated_questions(cls) -> None:
        """Clear all generated questions."""
        cls.set("generated_questions", [])
    
    @classmethod
    def get_processed_documents(cls) -> Dict[str, Any]:
        """Get processed documents dictionary."""
        return cls.get("processed_documents", {})
    
    @classmethod
    def add_processed_document(cls, doc_id: str, doc_info: Dict[str, Any]) -> None:
        """Add a processed document."""
        docs = cls.get_processed_documents()
        docs[doc_id] = doc_info
        cls.set("processed_documents", docs)
    
    @classmethod
    def get_selected_themes(cls) -> List[str]:
        """Get selected themes list."""
        return cls.get("selected_themes", [])
    
    @classmethod
    def set_selected_themes(cls, themes: List[str]) -> None:
        """Set selected themes."""
        cls.set("selected_themes", themes)
    
    @classmethod
    def is_export_ready(cls) -> bool:
        """Check if export is ready."""
        return cls.get("export_ready", False)
    
    @classmethod
    def set_export_ready(cls, ready: bool = True) -> None:
        """Set export ready state."""
        cls.set("export_ready", ready)