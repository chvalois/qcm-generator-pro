"""
QCM Generation Page

Handles standard QCM generation with progressive workflow.
"""

import streamlit as st

from src.ui.components.common.status_display import StatusDisplay
from src.ui.components.common.forms import ConfigurationForms


class QCMGenerationPage:
    """QCM generation page component."""
    
    def __init__(self, interface):
        """
        Initialize the QCM generation page.
        
        Args:
            interface: StreamlitQCMInterface instance for generation
        """
        self.interface = interface
    
    def render_generation_form(self) -> dict:
        """
        Render the QCM generation configuration form.
        
        Returns:
            Configuration dictionary
        """
        st.subheader("⚙️ Configuration de génération")
        
        # Use the reusable form components
        config = ConfigurationForms.render_generation_configuration()
        
        # Document selection
        st.subheader("📄 Sélection du document")
        
        # Get available documents
        documents = []  # Initialize empty list
        try:
            from src.services.document.document_manager import get_document_manager
            doc_manager = get_document_manager()
            documents = doc_manager.list_documents()
            
            if documents:
                # Create document options for selectbox
                doc_options = ["Tous les documents"] + [
                    f"{doc['filename']} (ID: {doc['id']}, {doc['total_pages']} pages)"
                    for doc in documents
                ]
                
                selected_doc_option = st.selectbox(
                    "Document source",
                    options=doc_options,
                    help="Choisissez un document spécifique ou tous les documents"
                )
                
                if selected_doc_option == "Tous les documents":
                    config['document_id'] = None
                    config['selected_document_info'] = "Tous les documents"
                    selected_doc = None
                else:
                    # Extract document ID from the selected option
                    doc_index = doc_options.index(selected_doc_option) - 1  # -1 because of "Tous les documents"
                    selected_doc = documents[doc_index]
                    config['document_id'] = selected_doc['id']
                    config['selected_document_info'] = f"{selected_doc['filename']} ({selected_doc['total_pages']} pages)"
                    
                    # Show document themes if available
                    if selected_doc.get('themes'):
                        st.info(f"🎯 **Thèmes dans ce document:** {', '.join([theme['name'] for theme in selected_doc['themes']])}")
                        
            else:
                st.warning("⚠️ Aucun document trouvé. Uploadez d'abord des documents.")
                config['document_id'] = None
                config['selected_document_info'] = "Aucun document disponible"
                selected_doc = None
                
        except Exception as e:
            st.error(f"Erreur lors du chargement des documents: {e}")
            config['document_id'] = None
            config['selected_document_info'] = "Erreur de chargement"
            selected_doc = None
        
        # Additional QCM-specific options
        with st.expander("🎯 Options avancées", expanded=False):
            # Get themes based on selected document
            if selected_doc and selected_doc.get('themes'):
                # Use themes from the specific selected document
                available_themes = [theme['name'] for theme in selected_doc['themes']]
                help_text = f"Thèmes disponibles dans {selected_doc['filename']}"
            else:
                # Use all themes if no specific document selected or no themes in document
                available_themes = self.interface.get_available_themes()
                if config.get('document_id'):
                    help_text = "Aucun thème détecté dans ce document, utilisation de tous les thèmes"
                else:
                    help_text = "Thèmes disponibles dans tous les documents"
            
            config['themes_filter'] = st.multiselect(
                "Filtrer par thèmes (optionnel)",
                options=available_themes,
                help=help_text
            )
        
        return config
    
    def execute_generation(self, config: dict) -> None:
        """
        Execute QCM generation with the given configuration.
        
        Args:
            config: Generation configuration
        """
        with st.spinner("Génération en cours..."):
            try:
                from src.services.generation.qcm_generator import generate_progressive_qcm_sync
                from src.models.schemas import GenerationConfig
                from src.ui.core.session_state import SessionStateManager
                
                # Convert UI config to GenerationConfig
                generation_config = GenerationConfig(
                    num_questions=config['num_questions'],
                    language=config['language'],
                    question_types=config.get('question_types', {}),
                    difficulty_distribution=config.get('difficulty_distribution', {}),
                    themes_filter=config.get('themes_filter', []),
                    model="gpt-4o-mini"  # Default model, configured in System page
                )
                
                # Generate topics based on available themes
                available_themes = self.interface.get_available_themes()
                
                if config.get('themes_filter'):
                    topics = config['themes_filter']
                elif available_themes:
                    # Extract theme names from dictionaries
                    if available_themes and isinstance(available_themes[0], dict):
                        topics = [theme.get('name', str(theme)) for theme in available_themes[:5]]
                    else:
                        topics = available_themes[:5]  # Use first 5 themes
                else:
                    topics = ["General Knowledge"]  # Fallback
                
                # Execute progressive QCM generation
                session_id = SessionStateManager.get_or_create_session_id()
                
                # Prepare document IDs based on selection
                document_ids = None
                if config.get('document_id'):
                    document_ids = [str(config['document_id'])]  # Convert to string list
                
                results = generate_progressive_qcm_sync(
                    topics=topics,
                    config=generation_config,
                    document_ids=document_ids,
                    themes_filter=config.get('themes_filter'),
                    session_id=session_id
                )
                
                # Store generated questions in session state
                if results and 'final_questions' in results:
                    questions = results['final_questions']
                    SessionStateManager.set_generated_questions(questions)
                    
                    StatusDisplay.show_success(f"✅ {len(questions)} questions générées avec succès !")
                    
                    # Display generation summary
                    st.subheader("📊 Résumé de la génération")
                    
                    # Show selected document info
                    doc_info = config.get('selected_document_info', 'Information non disponible')
                    st.write(f"📄 **Document source:** {doc_info}")
                    
                    if 'validation_steps' in results:
                        st.write("**Étapes de validation:**")
                        for step in results['validation_steps']:
                            step_name = step.get('step', 'Étape inconnue')
                            step_count = step.get('questions_count', 0)
                            st.write(f"• {step_name}: {step_count} questions")
                    
                    st.rerun()  # Refresh to show generated questions
                else:
                    StatusDisplay.show_warning("⚠️ Aucune question générée. Vérifiez la configuration.")
                
            except Exception as e:
                StatusDisplay.show_error(f"Erreur lors de la génération: {e}")
                st.exception(e)
    
    def render(self) -> None:
        """Render the complete QCM generation page."""
        st.header("🎯 Génération de QCM")
        
        # Check if documents are available
        if not self.interface.get_available_themes():
            st.warning("⚠️ Aucun document traité trouvé. Uploadez d'abord des documents.")
            return
        
        # Generation form
        config = self.render_generation_form()
        
        # Generation button
        if st.button("🚀 Générer des questions", type="primary"):
            self.execute_generation(config)
        
        # Display existing questions if any
        from src.ui.core.session_state import SessionStateManager
        questions = SessionStateManager.get_generated_questions()
        
        if questions:
            st.divider()
            st.subheader(f"📝 Questions générées ({len(questions)})")
            
            for i, question in enumerate(questions[:3]):  # Show first 3
                # Handle both dict and object structures
                if isinstance(question, dict):
                    question_text = question.get('question_text', question.get('question', 'Question non trouvée'))
                    options = question.get('options', [])
                    correct_answers = question.get('correct_answers', [])
                    explanation = question.get('explanation', '')
                else:
                    question_text = getattr(question, 'question_text', 'Question non trouvée')
                    options = getattr(question, 'options', [])
                    correct_answers = getattr(question, 'correct_answers', [])
                    explanation = getattr(question, 'explanation', '')
                
                with st.expander(f"Question {i+1}: {question_text[:50]}..."):
                    st.write(f"**Question:** {question_text}")
                    
                    # Display options with correct answers marked
                    for j, option in enumerate(options):
                        if isinstance(option, dict):
                            option_text = option.get('text', str(option))
                            is_correct = option.get('is_correct', False)
                        else:
                            # Handle object attributes
                            option_text = getattr(option, 'text', str(option))
                            is_correct = getattr(option, 'is_correct', False)
                        
                        # Use is_correct from option or check index in correct_answers
                        if is_correct or j in correct_answers:
                            prefix = "✅"
                            style = "**"
                        else:
                            prefix = "◯"
                            style = ""
                        
                        st.write(f"{prefix} {style}{option_text}{style}")
                    
                    # Display explanation if available
                    if explanation:
                        st.markdown("**💡 Explication:**")
                        st.write(explanation)
                    else:
                        st.info("ℹ️ Aucune explication disponible")
            
            if len(questions) > 3:
                st.info(f"... et {len(questions) - 3} autres questions disponibles pour export")