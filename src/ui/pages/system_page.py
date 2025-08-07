"""
System Page

Handles system configuration, LLM status, and application monitoring.
"""

import streamlit as st

from src.ui.components.common.status_display import StatusDisplay


class SystemPage:
    """System configuration and monitoring page component."""
    
    def __init__(self, interface):
        """
        Initialize the system page.
        
        Args:
            interface: StreamlitQCMInterface instance for system operations
        """
        self.interface = interface
    
    def render_llm_configuration(self) -> None:
        """Render LLM configuration section."""
        st.subheader("🤖 Configuration LLM")
        
        try:
            # Test LLM connections
            if st.button("🔍 Tester les connexions LLM"):
                with st.spinner("Test des connexions..."):
                    status = self.interface.test_llm_connection()
                    if "✅" in status:
                        StatusDisplay.show_success(status.replace("✅", "").strip())
                    else:
                        StatusDisplay.show_error(status.replace("❌", "").strip())
                        
        except Exception as e:
            StatusDisplay.show_error(f"Erreur configuration LLM: {e}")
            st.exception(e)
    
    def render_ollama_models(self) -> None:
        """Render Ollama model management section."""
        st.subheader("🦙 Gestion des modèles Ollama")
        
        try:
            self.interface.show_ollama_model_downloads()
        except Exception as e:
            StatusDisplay.show_error(f"Erreur gestion Ollama: {e}")
    
    def render_system_info(self) -> None:
        """Render system information section."""
        st.subheader("ℹ️ Informations système")
        
        # Basic system info
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Version Streamlit", st.__version__)
            
        with col2:
            st.metric("Session State Keys", len(st.session_state.keys()))
        
        # Session state debug info
        if st.checkbox("Afficher les détails de session"):
            st.json({
                key: str(value)[:100] + "..." if len(str(value)) > 100 else value 
                for key, value in st.session_state.items()
            })
    
    def render(self) -> None:
        """Render the complete system page."""
        st.header("🔧 État du système")
        
        # LLM Configuration
        self.render_llm_configuration()
        
        st.divider()
        
        # Ollama Models
        self.render_ollama_models()
        
        st.divider()
        
        # System Info
        self.render_system_info()