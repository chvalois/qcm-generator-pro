"""
Core UI Logic

This module contains core UI management and state handling logic.
"""

# Import only SessionStateManager to avoid circular imports
from .session_state import SessionStateManager

# InterfaceManager can be imported directly when needed
# from .interface_manager import InterfaceManager

__all__ = [
    "SessionStateManager",
]