"""
Reusable Form Components

Provides common form elements and configurations used across the application.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from src.models.enums import Difficulty, Language, QuestionType


class ConfigurationForms:
    """Reusable configuration form components."""
    
    @staticmethod
    def render_chunk_configuration() -> Tuple[int, int]:
        """
        Render chunk configuration form.
        
        Returns:
            Tuple of (chunk_size, chunk_overlap)
        """
        st.subheader("📏 Paramètres des chunks")
        
        chunk_size = st.slider(
            "Taille des chunks (caractères)", 
            min_value=500, max_value=3000, value=1000, step=100,
            help="Taille de chaque segment de texte en caractères"
        )
        
        chunk_overlap = st.slider(
            "Chevauchement entre chunks", 
            min_value=0, max_value=500, value=200, step=50,
            help="Nombre de caractères partagés entre chunks consécutifs"
        )
        
        return chunk_size, chunk_overlap
    
    @staticmethod
    def render_generation_configuration() -> Dict[str, Any]:
        """
        Render QCM generation configuration form.
        
        Returns:
            Configuration dictionary
        """
        config = {}
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Configuration de base")
            config['num_questions'] = st.slider(
                "Nombre de questions", 
                min_value=1, max_value=50, value=10, step=1
            )
            
            config['language'] = st.selectbox(
                "Langue des questions",
                options=[Language.FR, Language.EN],  # Only French and English
                format_func=lambda x: {
                    "fr": "Français", 
                    "en": "English"
                }.get(x.value, x.value),
                index=0
            )
        
        with col2:
            st.subheader("🎯 Types de questions")
            config['question_types'] = {}
            
            # Question types with sliders
            multiple_choice_pct = st.slider(
                "Choix unique (%)", 
                min_value=0, max_value=100, value=70, step=5
            )
            
            multiple_selection_pct = 100 - multiple_choice_pct
            st.info(f"Choix multiples: {multiple_selection_pct}%")
            
            config['question_types'] = {
                QuestionType.MULTIPLE_CHOICE: multiple_choice_pct / 100,
                QuestionType.MULTIPLE_SELECTION: multiple_selection_pct / 100
            }
        
        # Difficulty distribution
        st.subheader("📈 Distribution des difficultés")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            easy_pct = st.slider("Facile (%)", 0, 100, 30, step=5)
        with col2:
            medium_pct = st.slider("Moyen (%)", 0, 100, 50, step=5)
        with col3:
            hard_pct = st.slider("Difficile (%)", 0, 100, 20, step=5)
        
        # Normalize percentages
        total = easy_pct + medium_pct + hard_pct
        if total > 0:
            config['difficulty_distribution'] = {
                Difficulty.EASY: easy_pct / total,
                Difficulty.MEDIUM: medium_pct / total, 
                Difficulty.HARD: hard_pct / total
            }
        else:
            config['difficulty_distribution'] = {
                Difficulty.EASY: 0.3,
                Difficulty.MEDIUM: 0.5,
                Difficulty.HARD: 0.2
            }
        
        return config
    
    @staticmethod
    def render_model_selection() -> str:
        """
        Render LLM model selection form.
        
        Returns:
            Selected model name
        """
        st.subheader("🤖 Sélection du modèle")
        
        model_options = [
            "gpt-4o-mini",
            "gpt-4o", 
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "mistral:7b",
            "llama3:8b",
            "llama3:70b"
        ]
        
        selected_model = st.selectbox(
            "Modèle LLM",
            options=model_options,
            index=0,
            help="Sélectionner le modèle LLM pour la génération"
        )
        
        return selected_model
    
    @staticmethod
    def render_title_pattern_configuration() -> List[str]:
        """
        Render title pattern configuration form.
        
        Returns:
            List of title patterns
        """
        st.subheader("📋 Structure des titres")
        st.markdown("**Définissez les patterns attendus pour chaque niveau de titre :**")
        
        # Add intelligent pattern explanation
        st.info("""
        🧠 **Détection Intelligente** : Donnez seulement UN exemple par type de pattern. 
        Le système généralisera automatiquement !
        
        **Exemples :**
        - Pattern numérique : "1.1 Introduction" → détecte "1.2", "2.1", etc.
        - Pattern alphabétique : "A. Concepts" → détecte "B.", "C.", etc.
        - Pattern mixte : "Chapitre 1 : Bases" → détecte autres chapitres
        """)
        
        patterns = []
        
        for level in range(1, 4):  # Support up to 3 levels
            pattern = st.text_input(
                f"Pattern niveau {level}",
                placeholder=f"Ex: {'1.' if level == 1 else '1.1' if level == 2 else '1.1.1'} Titre exemple",
                key=f"title_pattern_{level}"
            )
            if pattern.strip():
                patterns.append(pattern.strip())
        
        return patterns