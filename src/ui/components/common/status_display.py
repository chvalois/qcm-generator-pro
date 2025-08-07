"""
Status Display Component

Provides consistent status messages (success, error, info, warning) for the application.
"""

import streamlit as st
from enum import Enum
from typing import Optional


class StatusType(Enum):
    """Status message types."""
    SUCCESS = "success"
    ERROR = "error" 
    INFO = "info"
    WARNING = "warning"


class StatusDisplay:
    """Status message display component."""
    
    @staticmethod
    def show_success(message: str, container: Optional[st.container] = None) -> None:
        """
        Display a success message.
        
        Args:
            message: Success message to display
            container: Optional container to display in
        """
        target = container if container else st
        target.markdown(f"""
        <div class="status-success">
            ✅ {message}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_error(message: str, container: Optional[st.container] = None) -> None:
        """
        Display an error message.
        
        Args:
            message: Error message to display
            container: Optional container to display in
        """
        target = container if container else st
        target.markdown(f"""
        <div class="status-error">
            ❌ {message}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_info(message: str, container: Optional[st.container] = None) -> None:
        """
        Display an info message.
        
        Args:
            message: Info message to display
            container: Optional container to display in
        """
        target = container if container else st
        target.markdown(f"""
        <div class="status-info">
            ℹ️ {message}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_warning(message: str, container: Optional[st.container] = None) -> None:
        """
        Display a warning message.
        
        Args:
            message: Warning message to display
            container: Optional container to display in
        """
        target = container if container else st
        target.warning(f"⚠️ {message}")
    
    @staticmethod
    def show_status(status_type: StatusType, message: str, container: Optional[st.container] = None) -> None:
        """
        Display a status message of the specified type.
        
        Args:
            status_type: Type of status message
            message: Message to display
            container: Optional container to display in
        """
        if status_type == StatusType.SUCCESS:
            StatusDisplay.show_success(message, container)
        elif status_type == StatusType.ERROR:
            StatusDisplay.show_error(message, container)
        elif status_type == StatusType.INFO:
            StatusDisplay.show_info(message, container)
        elif status_type == StatusType.WARNING:
            StatusDisplay.show_warning(message, container)