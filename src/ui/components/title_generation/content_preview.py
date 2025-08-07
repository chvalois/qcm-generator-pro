"""Content Preview Component for previewing selected content chunks."""

import streamlit as st
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ContentPreviewComponent:
    """Component for previewing content chunks based on title selection."""
    
    def __init__(self):
        """Initialize the content preview component."""
        pass
    
    def render_selection_preview(
        self, 
        document_id: str,
        h1: Optional[str], 
        h2: Optional[str], 
        h3: Optional[str], 
        h4: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Render preview of selected content chunks.
        
        Args:
            document_id: ID of the document
            h1, h2, h3, h4: Selected title levels
            
        Returns:
            List of matching chunks or None if error
        """
        if not any([h1, h2, h3, h4]):
            return None
        
        try:
            from src.services.generation.title_based_generator import get_title_based_generator, TitleSelectionCriteria
            
            title_generator = get_title_based_generator()
            
            # Create selection criteria
            criteria = TitleSelectionCriteria(
                document_id=document_id,
                h1_title=h1,
                h2_title=h2, 
                h3_title=h3,
                h4_title=h4
            )
            
            # Get matching chunks preview
            matching_chunks = title_generator.get_chunks_for_title_selection(criteria)
            
            # Display selection preview
            st.subheader("📋 Aperçu de la sélection")
            st.info(f"📍 **Chemin sélectionné:** {criteria.get_title_path()}")
            st.info(f"📊 **Chunks correspondants:** {len(matching_chunks)}")
            
            # Content warnings
            if len(matching_chunks) < 2:
                st.warning("⚠️ Peu de contenu disponible pour cette sélection. Considérez élargir votre sélection.")
            elif len(matching_chunks) > 50:
                st.warning("⚠️ Beaucoup de contenu sélectionné. Considérez affiner votre sélection pour de meilleurs résultats.")
            else:
                st.success("✅ Quantité de contenu appropriée pour la génération.")
            
            return matching_chunks
            
        except Exception as e:
            logger.error(f"Error in content preview: {e}")
            st.error(f"❌ Erreur lors de l'aperçu du contenu: {str(e)}")
            return None
    
    def calculate_content_metrics(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate content metrics from chunks.
        
        Args:
            chunks: List of content chunks
            
        Returns:
            Dictionary with metrics (total_words, recommended_questions, etc.)
        """
        total_words = 0
        
        for chunk in chunks:
            # Try different ways to get word count
            if 'word_count' in chunk:
                total_words += chunk['word_count']
            elif 'chunk_text' in chunk:
                total_words += len(chunk['chunk_text'].split())
            elif 'content' in chunk:
                total_words += len(chunk['content'].split())
        
        # Calculate recommended questions (1 question per ~150 words)
        recommended_questions = max(1, total_words // 150)
        
        return {
            'total_words': total_words,
            'recommended_questions': recommended_questions,
            'chunk_count': len(chunks)
        }
    
    def get_title_level_and_limits(self, h1: str, h2: str, h3: str, h4: str) -> tuple[str, int]:
        """
        Determine the title level and corresponding question limits.
        
        Args:
            h1, h2, h3, h4: Selected title levels
            
        Returns:
            Tuple of (title_level, max_questions)
        """
        if h4:
            return "H4", 10
        elif h3:
            return "H3", 20
        elif h2:
            return "H2", 50
        elif h1:
            return "H1", 100
        else:
            return "Multiple", 20