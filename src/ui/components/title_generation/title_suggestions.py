"""Title Suggestions Component for automatic title suggestions."""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TitleSuggestionsComponent:
    """Component for displaying and handling automatic title suggestions."""
    
    def __init__(self):
        """Initialize the title suggestions component."""
        pass
    
    def render_suggestions(self, suggestions: List[Dict[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Render title suggestions interface.
        
        Args:
            suggestions: List of title suggestions
            
        Returns:
            Tuple of (is_suggestion_selected, selected_suggestion_data)
        """
        if not suggestions:
            st.warning("⚠️ Aucune suggestion disponible. Utilisez la sélection manuelle.")
            return False, None
        
        st.info(f"💡 {len(suggestions)} suggestions de titres avec suffisamment de contenu:")
        
        # Display suggestions as options
        suggestion_options = {}
        for i, suggestion in enumerate(suggestions):
            label = f"[{suggestion['level']}] {suggestion['title']} ({suggestion['chunk_count']} chunks)"
            suggestion_options[label] = suggestion
        
        selected_suggestion = st.selectbox(
            "Suggestions automatiques:",
            options=["Sélection manuelle"] + list(suggestion_options.keys()),
            help="Choisissez une suggestion ou faites une sélection manuelle",
            key="title_suggestions_select"
        )
        
        if selected_suggestion != "Sélection manuelle":
            suggestion = suggestion_options[selected_suggestion]
            st.success(f"✅ Suggestion sélectionnée: {suggestion.get('description', suggestion['title'])}")
            return True, suggestion
        
        return False, None
    
    def extract_criteria_from_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[str, str, str, str]:
        """
        Extract title criteria from selected suggestion.
        
        Args:
            suggestion: Selected suggestion data
            
        Returns:
            Tuple of (h1_title, h2_title, h3_title, h4_title)
        """
        try:
            criteria = suggestion.get('criteria', {})
            
            if hasattr(criteria, 'h1_title'):
                # It's a TitleSelectionCriteria object
                return (
                    criteria.h1_title,
                    criteria.h2_title, 
                    criteria.h3_title,
                    criteria.h4_title
                )
            else:
                # It's a dictionary
                return (
                    criteria.get('h1_title'),
                    criteria.get('h2_title'),
                    criteria.get('h3_title'), 
                    criteria.get('h4_title')
                )
        except Exception as e:
            logger.warning(f"Could not extract criteria from suggestion: {e}")
            return (None, None, None, None)