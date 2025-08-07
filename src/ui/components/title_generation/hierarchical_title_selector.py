"""Hierarchical Title Selector Component for cascading H1->H2->H3->H4 selection."""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HierarchicalTitleSelector:
    """Component for hierarchical title selection with cascading dropdowns."""
    
    def __init__(self):
        """Initialize the hierarchical title selector."""
        pass
    
    def render_manual_selection(
        self, 
        title_structure: Dict[str, Any],
        suggested_h1: Optional[str] = None,
        suggested_h2: Optional[str] = None, 
        suggested_h3: Optional[str] = None,
        suggested_h4: Optional[str] = None
    ) -> Tuple[str, str, str, str]:
        """
        Render manual hierarchical title selection interface.
        
        Args:
            title_structure: Document title structure
            suggested_h1: Pre-selected H1 from suggestion
            suggested_h2: Pre-selected H2 from suggestion
            suggested_h3: Pre-selected H3 from suggestion
            suggested_h4: Pre-selected H4 from suggestion
            
        Returns:
            Tuple of (final_h1, final_h2, final_h3, final_h4)
        """
        st.subheader("🔧 Sélection manuelle")
        
        # Build H1 options from structure
        h1_titles = title_structure.get('h1_titles', {})
        h1_options = ["Tous"] + list(h1_titles.keys())
        
        # Determine default index for H1
        h1_default_index = 0
        if suggested_h1 and suggested_h1 in h1_options:
            h1_default_index = h1_options.index(suggested_h1)
        
        selected_h1_manual = st.selectbox(
            "Titre H1:",
            options=h1_options,
            index=h1_default_index,
            key="manual_h1"
        )
        
        # H2 options depend on H1 selection
        h2_options = ["Tous"]
        if selected_h1_manual != "Tous" and selected_h1_manual in h1_titles:
            h2_titles = h1_titles[selected_h1_manual].get('h2_titles', {})
            h2_options.extend(h2_titles.keys())
        
        # Determine default index for H2
        h2_default_index = 0
        if suggested_h2 and suggested_h2 in h2_options:
            h2_default_index = h2_options.index(suggested_h2)
        
        selected_h2_manual = st.selectbox(
            "Titre H2:",
            options=h2_options,
            index=h2_default_index,
            key="manual_h2"
        )
        
        # H3 options depend on H1 and H2 selection
        h3_options = ["Tous"]
        if (selected_h1_manual != "Tous" and selected_h2_manual != "Tous" and 
            selected_h1_manual in h1_titles):
            h2_titles = h1_titles[selected_h1_manual].get('h2_titles', {})
            if selected_h2_manual in h2_titles:
                h3_titles = h2_titles[selected_h2_manual].get('h3_titles', {})
                h3_options.extend(h3_titles.keys())
        
        # Determine default index for H3
        h3_default_index = 0
        if suggested_h3 and suggested_h3 in h3_options:
            h3_default_index = h3_options.index(suggested_h3)
        
        selected_h3_manual = st.selectbox(
            "Titre H3:",
            options=h3_options,
            index=h3_default_index,
            key="manual_h3"
        )
        
        # H4 options depend on H1, H2, and H3 selection
        h4_options = ["Tous"]
        if (selected_h1_manual != "Tous" and selected_h2_manual != "Tous" and 
            selected_h3_manual != "Tous" and selected_h1_manual in h1_titles):
            h2_titles = h1_titles[selected_h1_manual].get('h2_titles', {})
            if selected_h2_manual in h2_titles:
                h3_titles = h2_titles[selected_h2_manual].get('h3_titles', {})
                if selected_h3_manual in h3_titles:
                    h4_titles = h3_titles[selected_h3_manual].get('h4_titles', {})
                    h4_options.extend(h4_titles.keys())
        
        # Determine default index for H4
        h4_default_index = 0
        if suggested_h4 and suggested_h4 in h4_options:
            h4_default_index = h4_options.index(suggested_h4)
        
        selected_h4_manual = st.selectbox(
            "Titre H4:",
            options=h4_options,
            index=h4_default_index,
            key="manual_h4"
        )
        
        # Create final selection criteria - return None for "Tous" values
        final_h1 = selected_h1_manual if selected_h1_manual != "Tous" else None
        final_h2 = selected_h2_manual if selected_h2_manual != "Tous" else None
        final_h3 = selected_h3_manual if selected_h3_manual != "Tous" else None
        final_h4 = selected_h4_manual if selected_h4_manual != "Tous" else None
        
        return final_h1, final_h2, final_h3, final_h4
    
    def has_valid_selection(self, h1: str, h2: str, h3: str, h4: str) -> bool:
        """
        Check if at least one title level is selected.
        
        Args:
            h1, h2, h3, h4: Title selections
            
        Returns:
            True if at least one title is selected
        """
        return any([h1, h2, h3, h4])