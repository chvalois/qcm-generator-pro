"""
QCM Generator Pro - Streamlit Web Interface

This module provides a user-friendly web interface using Streamlit
for the QCM generation system with component-based architecture.
"""

# Suppress deprecation warnings before other imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Any

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.core.config import settings
from src.services.llm.simple_examples_loader import get_examples_loader

logger = logging.getLogger(__name__)


class StreamlitQCMInterface:
    """Main Streamlit interface for QCM Generator Pro."""

    def __init__(self):
        """Initialize the Streamlit interface."""
        # Initialize session state
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = None
        if "generated_questions" not in st.session_state:
            st.session_state.generated_questions = []
        if "processed_documents" not in st.session_state:
            st.session_state.processed_documents = {}

        # Initialize examples loader
        self.examples_loader = get_examples_loader()
        
        # Initialize services
        from src.services.ui import StreamlitHelpersService, DocumentUIService, OllamaUIService
        from src.services.export import ExportService
        from src.ui.components.documents import DocumentDisplayComponent
        
        self.helpers_service = StreamlitHelpersService()
        self.document_service = DocumentUIService()
        self.ollama_service = OllamaUIService()
        self.export_service = ExportService()
        self.document_display = DocumentDisplayComponent()
        
    def get_available_example_files(self) -> List[str]:
        """Get list of available few-shot example files."""
        return self.helpers_service.get_available_example_files()

    def get_available_themes(self) -> List[str]:
        """Get available themes from processed documents."""
        return self.helpers_service.get_available_themes()
    
    def test_llm_connection(self) -> str:
        """Test LLM connection status."""
        return self.helpers_service.test_llm_connection()
    
    def show_ollama_model_downloads(self) -> None:
        """Show Ollama model management interface."""
        self.ollama_service.show_model_downloads()
    
    def export_questions(self, export_format: str) -> Tuple[str, str]:
        """
        Export generated questions to specified format.
        
        Args:
            export_format: Export format ("CSV (Udemy)" or "JSON")
        
        Returns:
            Tuple of (status_message, download_info)
        """
        return self.export_service.export_questions(export_format)

    def upload_and_process_document(self, file, config: Optional[Any] = None) -> tuple[str, str, str, Optional[Any]]:
        """Upload and process a PDF document."""
        return self.document_service.upload_and_process_document(file, config)


def create_streamlit_interface():
    """Create the main Streamlit interface using the new component architecture."""
    
    # Import the new interface manager
    from src.ui.core.interface_manager import InterfaceManager
    
    # Create the StreamlitQCMInterface for compatibility
    streamlit_interface = StreamlitQCMInterface()
    
    # Initialize and run the interface manager
    interface_manager = InterfaceManager(streamlit_interface)
    interface_manager.run()


def launch_streamlit_app():
    """Launch the Streamlit application."""
    logger.info("Starting Streamlit interface...")

    try:
        create_streamlit_interface()

    except Exception as e:
        logger.error(f"Failed to start Streamlit interface: {e}")
        st.error(f"Erreur d'initialisation: {str(e)}")
        raise


if __name__ == "__main__":
    # Launch the Streamlit interface
    launch_streamlit_app()
