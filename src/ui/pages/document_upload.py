"""
Document Upload Page

Handles PDF document upload, configuration, and processing.
"""

import streamlit as st
from typing import Tuple, Optional, Any

from src.ui.components.common.status_display import StatusDisplay
from src.ui.components.common.forms import ConfigurationForms
from src.ui.core.session_state import SessionStateManager
from src.models.schemas import ProcessingConfig, TitleStructureConfig


class DocumentUploadPage:
    """Document upload and processing page component."""
    
    def __init__(self, interface):
        """
        Initialize the document upload page.
        
        Args:
            interface: StreamlitQCMInterface instance for processing
        """
        self.interface = interface
    
    def render_upload_form(self) -> Tuple[Optional[Any], ProcessingConfig]:
        """
        Render the document upload form with configuration.
        
        Returns:
            Tuple of (uploaded_file, processing_config)
        """
        uploaded_file = st.file_uploader(
            "Sélectionner un fichier PDF",
            type=['pdf'],
            help="Uploadez un document PDF pour l'analyser et générer des questions"
        )
        
        # Configuration section
        with st.expander("⚙️ Configuration du traitement", expanded=False):
            # Chunk configuration
            chunk_size, chunk_overlap = ConfigurationForms.render_chunk_configuration()
            
            # Title structure configuration
            st.subheader("📋 Structure des titres")
            st.markdown("**Définissez les patterns attendus pour chaque niveau de titre :**")
            
            # Add intelligent pattern explanation
            st.info("""
            🧠 **Détection Intelligente** : Donnez seulement UN exemple par type de pattern. 
            Le système généralisera automatiquement !
            
            **Exemples :**
            - `Parcours 1` → détectera `Parcours 1`, `Parcours 2`, `Parcours 15`, etc.
            - `I.` → détectera `I.`, `II.`, `XV.`, etc.
            - `1.` → détectera `1.`, `2.`, `25.`, etc.
            """)
            
            # H1 patterns
            h1_input = st.text_area(
                "H1 - Titres de niveau 1",
                placeholder="Parcours 1\nI.\nChapitre 1",
                help="⚡ UN exemple par type suffit ! Ex: 'Parcours 1' détectera tous les 'Parcours X'"
            )
            
            # H2 patterns  
            h2_input = st.text_area(
                "H2 - Titres de niveau 2",
                placeholder="Module 1\n1.\n1.1",
                help="⚡ UN exemple par type suffit ! Ex: 'Module 1' détectera tous les 'Module X'"
            )
            
            # H3 patterns
            h3_input = st.text_area(
                "H3 - Titres de niveau 3", 
                placeholder="Unité 1\ni.\na)",
                help="⚡ UN exemple par type suffit ! Ex: 'Unité 1' détectera tous les 'Unité X'"
            )
            
            # H4 patterns
            h4_input = st.text_area(
                "H4 - Titres de niveau 4",
                placeholder="a.\n1)",
                help="⚡ UN exemple par type suffit ! Ex: 'a.' détectera 'a.', 'b.', 'z.', etc."
            )
            
            use_auto_detection = st.checkbox(
                "Utiliser la détection automatique en complément",
                value=False,
                help="Active la détection automatique si les patterns définis ne suffisent pas"
            )
        
        # Parse patterns from text areas
        h1_patterns = [p.strip() for p in h1_input.split('\n') if p.strip()] if h1_input else []
        h2_patterns = [p.strip() for p in h2_input.split('\n') if p.strip()] if h2_input else []
        h3_patterns = [p.strip() for p in h3_input.split('\n') if p.strip()] if h3_input else []
        h4_patterns = [p.strip() for p in h4_input.split('\n') if p.strip()] if h4_input else []
        
        # Create processing config
        title_config = TitleStructureConfig(
            h1_patterns=h1_patterns,
            h2_patterns=h2_patterns, 
            h3_patterns=h3_patterns,
            h4_patterns=h4_patterns,
            use_auto_detection=use_auto_detection
        )
        processing_config = ProcessingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            title_structure=title_config
        )
        
        return uploaded_file, processing_config
    
    def render_processed_documents(self) -> None:
        """Render the list of processed documents."""
        processed_docs = SessionStateManager.get_processed_documents()
        
        if processed_docs:
            st.subheader("📋 Documents traités")
            for doc_id, doc_info in processed_docs.items():
                with st.expander(f"Document: {doc_info['filename']}"):
                    st.write(f"**Pages:** {doc_info['total_pages']}")
                    st.write(f"**Thèmes:** {len(doc_info.get('themes', []))}")
                    st.write(f"**Langue:** {doc_info['language']}")
    
    def process_document(self, uploaded_file: Any, processing_config: ProcessingConfig, result_container) -> None:
        """
        Process the uploaded document.
        
        Args:
            uploaded_file: Uploaded file object
            processing_config: Processing configuration
            result_container: Streamlit container for results
        """
        with st.spinner("Traitement en cours..."):
            try:
                status, doc_info, themes_info, metadata = self.interface.upload_and_process_document(
                    uploaded_file, processing_config
                )
                
                if "✅" in status:
                    StatusDisplay.show_success(status.replace("✅", "").strip())
                    with result_container:
                        st.markdown(doc_info)
                        st.markdown(themes_info)
                        
                        # Display title structure if available
                        if metadata:
                            from src.services.ui.document_ui_service import DocumentUIService
                            doc_service = DocumentUIService()
                            doc_service._display_title_structure(metadata)
                else:
                    StatusDisplay.show_error(status.replace("❌", "").strip())
                    
            except Exception as e:
                StatusDisplay.show_error(f"Erreur lors du traitement: {e}")
                st.exception(e)
    
    def render(self) -> None:
        """Render the complete document upload page."""
        st.header("📤 Téléchargement et traitement de documents")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Render upload form
            uploaded_file, processing_config = self.render_upload_form()
            
            # Process button
            if st.button("🚀 Traiter le document", type="primary", disabled=uploaded_file is None):
                self.process_document(uploaded_file, processing_config, col2)
        
        # Display processed documents
        self.render_processed_documents()