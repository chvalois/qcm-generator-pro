"""Title Generation Configuration Component."""

import streamlit as st
from typing import Dict, Any, List, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TitleGenerationConfig:
    """Component for title generation configuration options."""
    
    def __init__(self):
        """Initialize the title generation config component."""
        pass
    
    def render_generation_config(
        self,
        matching_chunks: List[Dict[str, Any]],
        title_level: str,
        max_questions: int,
        content_metrics: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Render generation configuration interface.
        
        Args:
            matching_chunks: List of matching content chunks
            title_level: Detected title level (H1, H2, etc.)
            max_questions: Maximum allowed questions for this level
            content_metrics: Content metrics dictionary
            
        Returns:
            Dictionary with generation configuration
        """
        st.subheader("⚙️ Configuration de génération")
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            # Display content information
            total_words = content_metrics['total_words']
            recommended_questions = min(content_metrics['recommended_questions'], max_questions)
            
            st.info(f"📊 **Niveau sélectionné:** {title_level} | "
                   f"**Mots:** {total_words:,} | "
                   f"**Recommandé:** {recommended_questions} questions")
            
            num_questions_title = st.slider(
                "Nombre de questions:",
                min_value=1,
                max_value=max_questions,
                value=recommended_questions,
                help=f"Max {max_questions} questions pour niveau {title_level} • "
                     f"Recommandation: 1 question pour 150 mots (~{total_words//150 or 1} questions)"
            )
            
            language_title = st.selectbox(
                "Langue:",
                options=["fr", "en"],
                index=0,
                format_func=lambda x: {"fr": "Français", "en": "English"}[x]
            )
        
        with col_config2:
            difficulty_title = st.selectbox(
                "Difficulté:",
                options=["mixed", "easy", "medium", "hard"],
                index=0,
                help="Mixed: mélange automatique des difficultés"
            )
            
            question_type_title = st.selectbox(
                "Type de questions:",
                options=["mixed", "multiple-choice", "multiple-selection"],
                index=0,
                help="Mixed: mélange de choix unique et multiple"
            )
        
        return {
            'num_questions': num_questions_title,
            'language': language_title,
            'difficulty': difficulty_title,
            'question_type': question_type_title
        }
    
    def render_few_shot_examples_config(self) -> Dict[str, Any]:
        """
        Render few-shot examples configuration.
        
        Returns:
            Dictionary with few-shot configuration
        """
        st.subheader("🎯 Few-Shot Examples")
        
        # Get available example files directly
        available_examples = self._get_available_example_files()
        
        if available_examples:
            use_examples_title = st.checkbox(
                "Utiliser des exemples guidés pour génération par titre",
                value=False,
                key="use_examples_title",
                help="Active l'utilisation d'exemples pour améliorer la qualité des questions par titre"
            )
            
            if use_examples_title:
                selected_examples_file_title = st.selectbox(
                    "Fichier d'exemples:",
                    options=available_examples,
                    key="examples_file_title",
                    help="Choisissez le fichier d'exemples correspondant à votre projet"
                )
                
                max_examples_title = st.slider(
                    "Nombre d'exemples max:",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key="max_examples_title",
                    help="Nombre maximum d'exemples à utiliser pour guider la génération par titre"
                )
                
                # Preview examples
                if st.checkbox("Aperçu des exemples", value=False, key="preview_examples_title"):
                    self._render_examples_preview(selected_examples_file_title, max_examples_title)
                
                return {
                    'use_examples': True,
                    'examples_file': selected_examples_file_title,
                    'max_examples': max_examples_title
                }
            else:
                return {'use_examples': False, 'examples_file': None, 'max_examples': 3}
        else:
            st.info("💡 Aucun fichier d'exemples disponible. Créez des fichiers JSON dans `data/few_shot_examples/`")
            return {'use_examples': False, 'examples_file': None, 'max_examples': 3}
    
    def _get_available_example_files(self) -> List[str]:
        """Get available few-shot example files directly from filesystem."""
        try:
            examples_dir = Path("data/few_shot_examples")
            if examples_dir.exists():
                files = [f.stem for f in examples_dir.glob("*.json")]
                logger.info(f"Found {len(files)} example files: {files}")
                return files
            else:
                logger.warning(f"Examples directory does not exist: {examples_dir}")
                return []
        except Exception as e:
            logger.warning(f"Could not get available example files: {e}")
            return []
    
    def _render_examples_preview(self, examples_file: str, max_examples: int) -> None:
        """
        Render preview of few-shot examples directly from JSON file.
        
        Args:
            examples_file: Selected examples file
            max_examples: Maximum number of examples to show
        """
        try:
            examples_path = Path(f"data/few_shot_examples/{examples_file}.json")
            
            if examples_path.exists():
                with open(examples_path, 'r', encoding='utf-8') as f:
                    examples_data = json.load(f)
                
                # Get examples from the loaded data
                examples_list = examples_data.get('examples', [])
                limited_examples = examples_list[:max_examples]
                
                if limited_examples:
                    st.write(f"**📋 {len(limited_examples)} exemple(s) trouvé(s) dans {examples_file}:**")
                    for i, ex in enumerate(limited_examples, 1):
                        with st.expander(f"Exemple {i}: {ex.get('theme', 'N/A')}", expanded=False):
                            st.write(f"**Question:** {ex.get('question', '')}")
                            st.write(f"**Type:** {ex.get('type', 'N/A')} | **Difficulté:** {ex.get('difficulty', 'N/A')}")
                            options = ex.get('options', [])
                            if options:
                                st.write("**Options:**")
                                for opt in options[:2]:  # Show first 2 options
                                    st.write(f"  - {opt}")
                                if len(options) > 2:
                                    st.write(f"  ... et {len(options)-2} autres")
                else:
                    st.warning("Aucun exemple trouvé dans ce fichier")
            else:
                st.warning(f"Fichier d'exemples '{examples_file}.json' non trouvé")
                
        except Exception as e:
            logger.warning(f"Could not preview examples: {e}")
            st.warning("Aperçu des exemples non disponible")