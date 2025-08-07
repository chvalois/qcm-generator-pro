"""
Export Page

Handles export of generated questions to different formats.
"""

import streamlit as st
from typing import Tuple

from src.ui.components.common.status_display import StatusDisplay
from src.ui.core.session_state import SessionStateManager


class ExportPage:
    """Export page component."""
    
    def __init__(self, interface):
        """
        Initialize the export page.
        
        Args:
            interface: StreamlitQCMInterface instance for export functionality
        """
        self.interface = interface
    
    def render_export_controls(self) -> str:
        """
        Render export format selection and button.
        
        Returns:
            Selected export format
        """
        export_format = st.selectbox(
            "Format d'export",
            options=["CSV (Udemy)", "JSON"],
            help="Choisissez le format d'export souhaité"
        )
        
        return export_format
    
    def execute_export(self, export_format: str, result_container) -> None:
        """
        Execute the export process.
        
        Args:
            export_format: Selected export format
            result_container: Container for displaying results
        """
        with result_container:
            with st.spinner("Export en cours..."):
                try:
                    status, download_info = self.interface.export_questions(export_format)
                    
                    if "✅" in status:
                        StatusDisplay.show_success(status.replace("✅", "").strip())
                        st.markdown(download_info)
                    else:
                        StatusDisplay.show_error(status.replace("❌", "").strip())
                        
                except Exception as e:
                    StatusDisplay.show_error(f"Erreur lors de l'export: {e}")
                    st.exception(e)
    
    def render_export_preview(self) -> None:
        """Render preview of questions to be exported."""
        st.subheader("👀 Aperçu des données à exporter")
        
        questions = SessionStateManager.get_generated_questions()
        st.info(f"Questions prêtes à l'export: {len(questions)}")
        
        if st.checkbox("Afficher le détail"):
            # Show first 3 questions as preview
            for i, question in enumerate(questions[:3]):
                # Count correct answers for export preview
                correct_count = sum(1 for opt in question.options if hasattr(opt, 'is_correct') and opt.is_correct)
                
                # Determine question type display info
                if hasattr(question, 'question_type'):
                    q_type = question.question_type.value if hasattr(question.question_type, 'value') else question.question_type
                    if q_type == "multiple-choice":
                        type_info = " (1 bonne réponse)"
                    elif q_type == "multiple-selection":
                        type_info = f" ({correct_count} bonnes réponses)"
                    else:
                        type_info = f" ({correct_count} bonne(s) réponse(s))"
                else:
                    type_info = f" ({correct_count} bonne(s) réponse(s))"
                
                st.write(f"**{i+1}.** {question.question_text}{type_info}")
            
            if len(questions) > 3:
                st.write(f"... et {len(questions) - 3} autres questions")
    
    def render(self) -> None:
        """Render the complete export page."""
        st.header("💾 Export des questions générées")
        
        questions = SessionStateManager.get_generated_questions()
        
        if not questions:
            st.warning("⚠️ Aucune question à exporter. Générez d'abord des questions.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Render export controls
            export_format = self.render_export_controls()
            
            # Export button
            if st.button("📁 Exporter", type="primary"):
                self.execute_export(export_format, col2)
        
        # Display export preview
        self.render_export_preview()