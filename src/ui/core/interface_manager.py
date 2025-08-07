"""
Interface Manager

Central manager for the Streamlit interface, coordinates between components and pages.
"""

import streamlit as st
from typing import Optional, Dict, Any

from src.ui.components.common.header import ApplicationHeader
from src.ui.components.common.sidebar import SidebarNavigation
from src.ui.core.session_state import SessionStateManager
from src.services.llm.simple_examples_loader import get_examples_loader

# Import page components
from src.ui.pages import (
    DocumentUploadPage,
    DocumentManagerPage,
    QCMGenerationPage, 
    TitleGenerationPage,
    ExportPage,
    SystemPage
)


class InterfaceManager:
    """Central interface manager for the QCM Generator Pro application."""
    
    def __init__(self, streamlit_interface=None):
        """Initialize the interface manager."""
        self.header = ApplicationHeader()
        self.sidebar = SidebarNavigation()
        self.examples_loader = get_examples_loader()
        
        # Store the StreamlitQCMInterface instance
        # For now, we'll use self as a placeholder until proper integration
        self.streamlit_interface = streamlit_interface or self
        
        # Initialize page components
        self.document_upload_page = DocumentUploadPage(self.streamlit_interface)
        self.document_manager_page = DocumentManagerPage(self.streamlit_interface)
        self.qcm_generation_page = QCMGenerationPage(self.streamlit_interface)
        self.title_generation_page = TitleGenerationPage(self.streamlit_interface)
        self.export_page = ExportPage(self.streamlit_interface)
        self.system_page = SystemPage(self.streamlit_interface)
        
        # Initialize session state
        SessionStateManager.initialize()
    
    def configure_page(self) -> None:
        """Configure the Streamlit page settings."""
        try:
            # Éviter les appels multiples à set_page_config
            if not hasattr(st, '_config_set'):
                st.set_page_config(
                    page_title="QCM Generator Pro",
                    page_icon="🎯",
                    layout="wide",
                    initial_sidebar_state="expanded"
                )
                st._config_set = True
        except Exception as e:
            # L'erreur est probablement due à un set_page_config déjà appelé
            st.write(f"⚠️ Configuration déjà définie: {e}")
    
    def render_header(self) -> None:
        """Render the application header with styling."""
        try:
            self.header.render()
        except Exception as e:
            st.error(f"Erreur lors du rendu de l'en-tête: {e}")
            st.exception(e)
    
    def render_navigation(self) -> str:
        """
        Render the navigation sidebar.
        
        Returns:
            Selected tab name
        """
        try:
            selected = self.sidebar.render()
            return selected
        except Exception as e:
            st.error(f"Erreur lors du rendu de la navigation: {e}")
            st.exception(e)
            return self.sidebar.DEFAULT_TABS[0]  # Fallback
    
    def get_available_example_files(self) -> list[str]:
        """
        Get list of available few-shot example files.
        
        Returns:
            List of example file names
        """
        try:
            return self.examples_loader.list_available_projects()
        except Exception as e:
            st.error(f"Failed to get example files: {e}")
            return []
    
    def add_sidebar_info(self) -> None:
        """Add additional information to the sidebar."""
        # Add generated questions count
        questions_count = len(SessionStateManager.get_generated_questions())
        if questions_count > 0:
            self.sidebar.add_sidebar_info(f"Questions générées: {questions_count}")
    
    def handle_page_routing(self, selected_tab: str) -> None:
        """
        Handle routing to different pages based on selected tab.
        
        Args:
            selected_tab: Selected tab/page name
        """
        try:
            if selected_tab == "📄 Upload de Documents":
                self.document_upload_page.render()
            elif selected_tab == "📚 Gestion Documents":
                self.document_manager_page.render()
            elif selected_tab == "🎯 Génération QCM":
                self.qcm_generation_page.render()
            elif selected_tab == "🏷️ Génération par Titre":
                self.title_generation_page.render()
            elif selected_tab == "📤 Export":
                self.export_page.render()
            elif selected_tab == "⚙️ Système":
                self.system_page.render()
            else:
                st.error(f"Page non trouvée: {selected_tab}")
        except Exception as e:
            st.error(f"Erreur lors du rendu de la page '{selected_tab}': {e}")
            st.exception(e)
    
    # Add stub methods that might be expected by page components
    def get_available_themes(self) -> list:
        """Get available themes (stub for page compatibility)."""
        # This will be properly implemented in Phase 2.3
        return []
    
    def test_llm_connection(self) -> str:
        """Test LLM connection (stub for page compatibility)."""
        # This will be properly implemented in Phase 2.3
        return "❌ Interface non connectée"
    
    def show_ollama_model_downloads(self) -> None:
        """Show Ollama model downloads (stub for page compatibility)."""
        # This will be properly implemented in Phase 2.3
        st.info("🚧 Gestion des modèles Ollama - À implémenter")
    
    def export_questions(self, export_format: str) -> tuple:
        """Export questions (stub for page compatibility)."""
        # This will be properly implemented in Phase 2.3
        return ("❌ Export non disponible", "Interface non connectée")
    
    def run(self) -> None:
        """Run the complete interface."""
        # Configure page
        self.configure_page()
        
        # Render header
        self.render_header()
        
        # Render navigation and get selected tab
        selected_tab = self.render_navigation()
        
        # Add sidebar information
        self.add_sidebar_info()
        
        # Route to appropriate page
        self.handle_page_routing(selected_tab)