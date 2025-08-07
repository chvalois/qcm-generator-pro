"""Document Display Components."""

import streamlit as st
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentDisplayComponent:
    """Components for displaying document information."""
    
    def __init__(self):
        """Initialize document display component."""
        pass
    
    def display_document_details(self, doc: Dict[str, Any]) -> None:
        """Display basic document details in expander."""
        st.write(f"**ID:** {doc['id']}")
        st.write(f"**Upload:** {doc['upload_date']}")
        st.write(f"**Langue:** {doc['language']}")
        st.write(f"**Chunks:** {doc['chunk_count']}")
        st.write(f"**Statut:** {doc['processing_status']}")

        if doc['themes']:
            st.write("**Thèmes:**")
            for theme in doc['themes']:
                confidence = theme.get('confidence', 0)
                keywords = theme.get('keywords', [])
                st.write(f"  • {theme['name']} (confiance: {confidence:.2f})")
                if keywords:
                    st.write(f"    Mots-clés: {', '.join(keywords[:5])}")
    
    def display_detailed_document_info(self, doc: Dict[str, Any]) -> None:
        """Display detailed document information."""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Informations générales")
            st.write(f"**Nom du fichier:** {doc['filename']}")
            st.write(f"**ID:** {doc['id']}")
            st.write(f"**Pages:** {doc['total_pages']}")
            st.write(f"**Langue:** {doc['language']}")
            st.write(f"**Statut:** {doc['processing_status']}")

        with col2:
            st.markdown("### 📊 Statistiques")
            st.metric("Chunks de texte", doc['chunk_count'])
            st.metric("Thèmes détectés", len(doc['themes']))

            # Parse upload date
            try:
                from datetime import datetime
                upload_dt = datetime.fromisoformat(doc['upload_date'].replace('Z', '+00:00'))
                st.write(f"**Upload:** {upload_dt.strftime('%d/%m/%Y à %H:%M')}")
            except:
                st.write(f"**Upload:** {doc['upload_date']}")

        if doc['themes']:
            st.markdown("### 🎯 Thèmes détectés")
            for i, theme in enumerate(doc['themes'], 1):
                with st.expander(f"Thème {i}: {theme['name']}"):
                    st.write(f"**Confiance:** {theme.get('confidence', 0):.2f}")
                    keywords = theme.get('keywords', [])
                    if keywords:
                        st.write(f"**Mots-clés:** {', '.join(keywords)}")
    
    def display_document_statistics(self, doc: Dict[str, Any]) -> None:
        """Display document statistics."""
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Pages", doc['total_pages'])
        with col2:
            st.metric("Chunks", doc['chunk_count'])
        with col3:
            st.metric("Thèmes", len(doc['themes']))
        with col4:
            if doc['themes']:
                avg_confidence = sum(theme.get('confidence', 0) for theme in doc['themes']) / len(doc['themes'])
                st.metric("Confiance moy.", f"{avg_confidence:.2f}")
            else:
                st.metric("Confiance moy.", "N/A")

        # Theme distribution chart (simplified version)
        if doc['themes']:
            st.markdown("### 📈 Distribution des thèmes")
            
            for theme in doc['themes']:
                confidence = theme.get('confidence', 0)
                st.write(f"**{theme['name']}** - Confiance: {confidence:.2f}")
                st.progress(confidence)
    
    def display_document_chunks(self, doc_id: int, doc: Dict[str, Any]) -> None:
        """Display document chunks with navigation."""
        try:
            from src.services.document.document_manager import get_document_chunks
            
            chunks = get_document_chunks(str(doc_id), include_titles=True)
            
            if not chunks:
                st.warning("Aucun chunk trouvé pour ce document")
                return
            
            st.markdown("### 📑 Navigation par chunks")
            
            # Chunk selector
            chunk_options = [
                f"Chunk {i+1} ({chunk.get('word_count', 'N/A')} mots)" 
                for i, chunk in enumerate(chunks)
            ]
            
            selected_chunk_str = st.selectbox(
                f"Sélectionner un chunk (Total: {len(chunks)})",
                chunk_options,
                key="chunk_selector"
            )
            
            # Extract chunk index
            selected_index = chunk_options.index(selected_chunk_str)
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅️ Précédent", disabled=selected_index == 0):
                    if selected_index > 0:
                        st.session_state.chunk_selector = chunk_options[selected_index - 1]
                        st.rerun()
            
            with col2:
                st.write(f"Chunk {selected_index + 1} sur {len(chunks)}")
            
            with col3:
                if st.button("➡️ Suivant", disabled=selected_index == len(chunks) - 1):
                    if selected_index < len(chunks) - 1:
                        st.session_state.chunk_selector = chunk_options[selected_index + 1]
                        st.rerun()
            
            # Display selected chunk
            chunk = chunks[selected_index]
            
            # Metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ordre", chunk.get('chunk_order', selected_index + 1))
            with col2:
                st.metric("Mots", chunk.get('word_count', len(chunk.get('chunk_text', '').split())))
            with col3:
                st.metric("Caractères", len(chunk.get('chunk_text', '')))
            with col4:
                word_count = chunk.get('word_count', len(chunk.get('chunk_text', '').split()))
                reading_time = max(1, word_count // 200)  # ~200 words/minute
                st.metric("Lecture", f"{reading_time}min")
            
            # Title hierarchy
            title_hierarchy = chunk.get('title_hierarchy', {})
            if title_hierarchy and any(title_hierarchy.values()):
                st.markdown("### 🏷️ Hiérarchie des titres")
                for level in ['h1_title', 'h2_title', 'h3_title', 'h4_title']:
                    title = title_hierarchy.get(level)
                    if title:
                        level_name = level.replace('_title', '').upper()
                        st.write(f"**{level_name}:** {title}")
            
            # Chunk content
            st.markdown("### 📝 Contenu du chunk")
            chunk_text = chunk.get('chunk_text', 'Aucun contenu disponible')
            
            # Display in scrollable text area
            st.text_area(
                "Contenu complet:",
                chunk_text,
                height=400,
                key=f"chunk_content_{selected_index}",
                disabled=True
            )
            
        except Exception as e:
            logger.error(f"Error displaying document chunks: {e}")
            st.error(f"Erreur lors de l'affichage des chunks: {e}")