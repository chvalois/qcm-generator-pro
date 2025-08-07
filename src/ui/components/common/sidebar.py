"""
Sidebar Navigation Component

Provides the main navigation sidebar for the QCM Generator Pro interface.
"""

import streamlit as st
from typing import List, Optional


class SidebarNavigation:
    """Main navigation sidebar component."""
    
    DEFAULT_TABS = [
        "📄 Upload de Documents", 
        "📚 Gestion Documents", 
        "🎯 Génération QCM", 
        "🏷️ Génération par Titre", 
        "📤 Export", 
        "⚙️ Système"
    ]
    
    def __init__(self, tabs: Optional[List[str]] = None, title: str = "🧭 Navigation"):
        """
        Initialize sidebar navigation.
        
        Args:
            tabs: List of tab names (uses default if not provided)
            title: Sidebar title
        """
        self.tabs = tabs if tabs is not None else self.DEFAULT_TABS
        self.title = title
    
    def render(self) -> str:
        """
        Render the sidebar navigation and return selected tab.
        
        Returns:
            Selected tab name
        """
        st.sidebar.title(self.title)
        
        tab_choice = st.sidebar.radio(
            "Choisir une section:",
            self.tabs
        )
        
        return tab_choice
    
    def add_sidebar_info(self, info_text: str) -> None:
        """
        Add additional information to the sidebar.
        
        Args:
            info_text: Information text to display
        """
        st.sidebar.info(info_text)
    
    def add_sidebar_section(self, title: str, content_func) -> None:
        """
        Add a custom section to the sidebar.
        
        Args:
            title: Section title
            content_func: Function that renders the section content
        """
        with st.sidebar.expander(title):
            content_func()