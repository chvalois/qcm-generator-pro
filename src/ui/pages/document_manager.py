"""
Document Manager Page

Handles document management, RAG engine configuration, and document operations.
"""

import streamlit as st
from typing import List, Dict, Any, Optional

from src.ui.components.common.status_display import StatusDisplay
from src.ui.core.session_state import SessionStateManager
from src.services.document.document_manager import get_document_manager, list_stored_documents
from src.services.infrastructure.rag_engine import get_rag_engine, switch_rag_engine
from src.ui.components.documents.document_display import DocumentDisplayComponent


class DocumentManagerPage:
    """Document management page component."""
    
    def __init__(self, interface):
        """
        Initialize the document manager page.
        
        Args:
            interface: StreamlitQCMInterface instance for document operations
        """
        self.interface = interface
        self.document_display = DocumentDisplayComponent()
        self.initialize_session_state()
    
    def initialize_session_state(self) -> None:
        """Initialize session state variables for document management."""
        if 'bulk_selection_mode' not in st.session_state:
            st.session_state.bulk_selection_mode = False
        if 'selected_docs_for_deletion' not in st.session_state:
            st.session_state.selected_docs_for_deletion = []
        if 'confirm_bulk_delete' not in st.session_state:
            st.session_state.confirm_bulk_delete = False
        if 'confirm_delete_doc' not in st.session_state:
            st.session_state.confirm_delete_doc = None
        if 'show_document_details' not in st.session_state:
            st.session_state.show_document_details = None
        if 'show_document_stats' not in st.session_state:
            st.session_state.show_document_stats = None
        if 'show_document_chunks' not in st.session_state:
            st.session_state.show_document_chunks = None
    
    def render_rag_configuration(self) -> None:
        """Render RAG engine configuration section."""
        st.subheader("⚙️ Configuration de persistance")
        
        # Show current RAG engine type
        current_engine = get_rag_engine()
        engine_type = type(current_engine).__name__
        st.info(f"**Moteur actuel:** {engine_type}")
        
        # Engine switching
        new_engine_type = st.selectbox(
            "Type de moteur RAG:",
            ["simple", "chromadb"],
            index=0 if "Simple" in engine_type else 1,
            help="SimpleRAGEngine: mémoire temporaire, ChromaDBRAGEngine: persistance"
        )
        
        if st.button("🔄 Changer de moteur"):
            with st.spinner("Changement en cours..."):
                success = switch_rag_engine(new_engine_type)
                if success:
                    StatusDisplay.show_success(f"Moteur changé vers {new_engine_type}")
                    st.rerun()
                else:
                    StatusDisplay.show_error("Échec du changement de moteur")
        
        # Migration button
        if new_engine_type == "chromadb":
            st.subheader("📊 Migration des données")
            if st.button("🚀 Migrer vers ChromaDB"):
                with st.spinner("Migration en cours..."):
                    st.info("💡 Exécuter: `python scripts/migrate_to_chromadb.py migrate`")
    
    def render_bulk_controls(self, stored_docs: List[Dict[str, Any]]) -> None:
        """
        Render bulk document management controls.
        
        Args:
            stored_docs: List of stored documents
        """
        col_control1, col_control2, col_control3 = st.columns([2, 1, 1])
        
        with col_control1:
            StatusDisplay.show_success(f"{len(stored_docs)} document(s) trouvé(s)")
        
        with col_control2:
            if st.button("☑️ Sélection multiple"):
                st.session_state.bulk_selection_mode = not st.session_state.bulk_selection_mode
                st.rerun()
        
        with col_control3:
            if st.session_state.bulk_selection_mode:
                selected_docs = st.session_state.selected_docs_for_deletion
                if selected_docs and st.button(f"🗑️ Supprimer ({len(selected_docs)})"):
                    st.session_state.confirm_bulk_delete = True
                    st.rerun()
    
    def render_bulk_delete_confirmation(self) -> None:
        """Render bulk delete confirmation dialog."""
        if st.session_state.confirm_bulk_delete:
            selected_docs = st.session_state.selected_docs_for_deletion
            st.error(f"⚠️ Confirmer la suppression de {len(selected_docs)} document(s) ?")
            
            col_confirm1, col_confirm2 = st.columns(2)
            
            with col_confirm1:
                if st.button("✅ Oui, supprimer", type="primary"):
                    self.execute_bulk_delete(selected_docs)
            
            with col_confirm2:
                if st.button("❌ Annuler"):
                    st.session_state.confirm_bulk_delete = False
                    st.rerun()
    
    def execute_bulk_delete(self, selected_docs: List[int]) -> None:
        """
        Execute bulk document deletion.
        
        Args:
            selected_docs: List of document IDs to delete
        """
        success_count = 0
        doc_manager = get_document_manager()
        
        for doc_id in selected_docs:
            if doc_manager.delete_document(doc_id):
                success_count += 1
                # Also remove from RAG engine
                try:
                    rag_engine = get_rag_engine()
                    if hasattr(rag_engine, 'document_chunks') and str(doc_id) in rag_engine.document_chunks:
                        del rag_engine.document_chunks[str(doc_id)]
                except:
                    pass
        
        StatusDisplay.show_success(f"{success_count}/{len(selected_docs)} document(s) supprimé(s)")
        
        # Reset states
        st.session_state.confirm_bulk_delete = False
        st.session_state.selected_docs_for_deletion = []
        st.session_state.bulk_selection_mode = False
        st.rerun()
    
    def render_document_item(self, doc: Dict[str, Any]) -> None:
        """
        Render a single document item with actions.
        
        Args:
            doc: Document information dictionary
        """
        doc_id = doc['id']
        
        # Create expander with selection checkbox if in bulk mode
        if st.session_state.bulk_selection_mode:
            col_check, col_expand = st.columns([0.1, 0.9])
            
            with col_check:
                is_selected = doc_id in st.session_state.selected_docs_for_deletion
                if st.checkbox("", value=is_selected, key=f"select_{doc_id}"):
                    if doc_id not in st.session_state.selected_docs_for_deletion:
                        st.session_state.selected_docs_for_deletion.append(doc_id)
                else:
                    if doc_id in st.session_state.selected_docs_for_deletion:
                        st.session_state.selected_docs_for_deletion.remove(doc_id)
            
            with col_expand:
                with st.expander(f"📄 {doc['filename']} ({doc['total_pages']} pages)"):
                    self.render_document_details(doc)
        else:
            with st.expander(f"📄 {doc['filename']} ({doc['total_pages']} pages)"):
                self.render_document_details(doc)
                self.render_document_actions(doc)
                self.render_individual_delete_confirmation(doc)
    
    def render_document_details(self, doc: Dict[str, Any]) -> None:
        """
        Render document details inside expander.
        
        Args:
            doc: Document information dictionary
        """
        self.document_display.display_document_details(doc)
    
    def render_document_actions(self, doc: Dict[str, Any]) -> None:
        """
        Render individual document action buttons.
        
        Args:
            doc: Document information dictionary
        """
        doc_id = doc['id']
        st.markdown("---")
        col_actions = st.columns(5)
        
        with col_actions[0]:
            if st.button("🎯 Utiliser", key=f"use_{doc_id}"):
                st.session_state.selected_document_for_generation = doc_id
                StatusDisplay.show_success(f"Document {doc['filename']} sélectionné pour génération")
        
        with col_actions[1]:
            if st.button("👁️ Détails", key=f"details_{doc_id}"):
                st.session_state.show_document_details = doc_id
                st.rerun()
        
        with col_actions[2]:
            if st.button("📊 Stats", key=f"stats_{doc_id}"):
                st.session_state.show_document_stats = doc_id
                st.rerun()
        
        with col_actions[3]:
            if st.button("📝 Chunks", key=f"chunks_{doc_id}"):
                st.session_state.show_document_chunks = doc_id
                st.rerun()
        
        with col_actions[4]:
            if st.button("🗑️ Supprimer", key=f"delete_{doc_id}"):
                st.session_state.confirm_delete_doc = doc_id
                st.rerun()
    
    def render_individual_delete_confirmation(self, doc: Dict[str, Any]) -> None:
        """
        Render individual document delete confirmation.
        
        Args:
            doc: Document information dictionary
        """
        doc_id = doc['id']
        
        if st.session_state.confirm_delete_doc == doc_id:
            st.error(f"⚠️ Confirmer la suppression de '{doc['filename']}' ?")
            col_confirm1, col_confirm2 = st.columns(2)
            
            with col_confirm1:
                if st.button("✅ Oui, supprimer", key=f"confirm_del_{doc_id}", type="primary"):
                    self.execute_individual_delete(doc_id, doc['filename'])
            
            with col_confirm2:
                if st.button("❌ Annuler", key=f"cancel_del_{doc_id}"):
                    st.session_state.confirm_delete_doc = None
                    st.rerun()
    
    def execute_individual_delete(self, doc_id: int, filename: str) -> None:
        """
        Execute individual document deletion.
        
        Args:
            doc_id: Document ID to delete
            filename: Document filename for display
        """
        doc_manager = get_document_manager()
        if doc_manager.delete_document(doc_id):
            # Also remove from RAG engine
            try:
                rag_engine = get_rag_engine()
                if hasattr(rag_engine, 'document_chunks') and str(doc_id) in rag_engine.document_chunks:
                    del rag_engine.document_chunks[str(doc_id)]
            except:
                pass
            
            StatusDisplay.show_success(f"Document '{filename}' supprimé")
            st.session_state.confirm_delete_doc = None
            st.rerun()
        else:
            StatusDisplay.show_error("Échec de la suppression")
    
    def render_detailed_document_views(self, stored_docs: List[Dict[str, Any]]) -> None:
        """
        Render detailed document information views.
        
        Args:
            stored_docs: List of stored documents
        """
        # Show detailed document information if requested
        if st.session_state.show_document_details:
            doc_id = st.session_state.show_document_details
            doc = next((d for d in stored_docs if d['id'] == doc_id), None)
            if doc:
                st.subheader(f"📋 Détails: {doc['filename']}")
                self.document_display.display_detailed_document_info(doc)
                if st.button("❌ Fermer détails"):
                    st.session_state.show_document_details = None
                    st.rerun()
        
        # Show document statistics if requested
        if st.session_state.show_document_stats:
            doc_id = st.session_state.show_document_stats
            doc = next((d for d in stored_docs if d['id'] == doc_id), None)
            if doc:
                st.subheader(f"📊 Statistiques: {doc['filename']}")
                self.document_display.display_document_statistics(doc)
                if st.button("❌ Fermer statistiques"):
                    st.session_state.show_document_stats = None
                    st.rerun()
        
        # Show document chunks if requested
        if st.session_state.show_document_chunks:
            doc_id = st.session_state.show_document_chunks
            doc = next((d for d in stored_docs if d['id'] == doc_id), None)
            if doc:
                st.subheader(f"📝 Chunks: {doc['filename']}")
                self.document_display.display_document_chunks(doc_id, doc)
                if st.button("❌ Fermer chunks"):
                    st.session_state.show_document_chunks = None
                    st.rerun()
    
    def render(self) -> None:
        """Render the complete document manager page."""
        st.header("📚 Gestion des documents et thèmes")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self.render_rag_configuration()
        
        with col2:
            st.subheader("📄 Documents stockés")
            
            try:
                stored_docs = list_stored_documents()
                
                if stored_docs:
                    # Document management controls
                    self.render_bulk_controls(stored_docs)
                    
                    # Bulk deletion confirmation
                    self.render_bulk_delete_confirmation()
                    
                    # Document list
                    for doc in stored_docs:
                        self.render_document_item(doc)
                    
                    # Detailed views
                    self.render_detailed_document_views(stored_docs)
                    
                else:
                    st.info("Aucun document trouvé. Uploadez des documents dans l'onglet 'Upload de Documents'.")
                    
            except Exception as e:
                StatusDisplay.show_error(f"Erreur lors du chargement des documents: {e}")
                st.exception(e)