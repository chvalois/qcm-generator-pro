"""Title Structure Analyzer Component for analyzing document title hierarchy."""

import streamlit as st
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TitleStructureAnalyzer:
    """Component for analyzing and displaying document title structure."""
    
    def __init__(self):
        """Initialize the title structure analyzer."""
        pass
    
    def render_structure_analysis(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze and render document title structure.
        
        Args:
            document_id: ID of the document to analyze
            
        Returns:
            Dictionary containing title structure or None if error
        """
        try:
            from src.services.generation.title_based_generator import get_title_based_generator
            title_generator = get_title_based_generator()
            
            # Show loading while analyzing
            with st.spinner("🔍 Analyse de la structure des titres..."):
                title_structure = title_generator.get_document_title_structure(document_id)
            
            if "error" in title_structure:
                st.error(f"❌ Erreur lors de l'analyse: {title_structure['error']}")
                return None
            
            # Display structure overview
            self._render_structure_overview(title_structure)
            
            return title_structure
            
        except Exception as e:
            logger.error(f"Error in title structure analysis: {e}")
            st.error(f"❌ Erreur lors de l'analyse de la structure: {str(e)}")
            return None
    
    def _render_structure_overview(self, title_structure: Dict[str, Any]) -> None:
        """
        Render the structure overview metrics.
        
        Args:
            title_structure: Title structure data
        """
        st.subheader("📊 Aperçu de la structure")
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Chunks totaux", title_structure.get('total_chunks', 0))
        
        with col_stats2:
            statistics = title_structure.get('statistics', {})
            chunks_with_titles = statistics.get('chunks_with_titles', 0)
            st.metric("Chunks avec titres", chunks_with_titles)
        
        with col_stats3:
            h1_titles = title_structure.get('h1_titles', {})
            st.metric("Titres H1", len(h1_titles))
    
    def get_title_suggestions(self, document_id: str, title_structure: Dict[str, Any]) -> list:
        """
        Get title suggestions based on content analysis.
        
        Args:
            document_id: ID of the document
            title_structure: Analyzed title structure
            
        Returns:
            List of title suggestions
        """
        try:
            from src.services.generation.title_based_generator import get_title_based_generator
            title_generator = get_title_based_generator()
            
            suggestions = title_generator.get_title_suggestions(document_id, min_chunks=1)
            return suggestions or []
            
        except Exception as e:
            logger.error(f"Error getting title suggestions: {e}")
            return []