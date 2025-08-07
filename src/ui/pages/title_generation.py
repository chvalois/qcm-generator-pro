"""Title Generation Page for QCM Generator Pro."""

import streamlit as st
from typing import Dict, Any, Optional, List
import logging
import uuid
import asyncio
import threading
import time
from src.ui.components.common import StatusDisplay
from src.ui.components.title_generation import (
    TitleStructureAnalyzer,
    TitleSuggestionsComponent,
    HierarchicalTitleSelector,
    ContentPreviewComponent,
    TitleGenerationConfig
)

logger = logging.getLogger(__name__)


class TitleGenerationPage:
    """Complete title-based QCM generation page with all original features."""
    
    def __init__(self, interface):
        """Initialize the title generation page."""
        self.interface = interface
        self.status_display = StatusDisplay()
        self.structure_analyzer = TitleStructureAnalyzer()
        self.suggestions_component = TitleSuggestionsComponent()
        self.title_selector = HierarchicalTitleSelector()
        self.content_preview = ContentPreviewComponent()
        self.generation_config = TitleGenerationConfig()
    
    def render(self) -> None:
        """Render the complete title generation page."""
        st.header("🎯 Génération par titre")
        st.write("Générez des questions basées sur la structure des titres de votre document.")
        
        # Document selection
        selected_doc = self._render_document_selection()
        if not selected_doc:
            st.info("📄 Veuillez d'abord uploader des documents dans la section 'Upload de documents'.")
            return
        
        # Title structure analysis and selection
        self._render_complete_title_workflow(selected_doc)
    
    def _render_document_selection(self) -> Optional[Dict[str, Any]]:
        """Render document selection interface."""
        try:
            from src.services.document.document_manager import get_document_manager
            doc_manager = get_document_manager()
            documents = doc_manager.list_documents()
            
            if not documents:
                return None
            
            doc_options = [
                f"{doc['filename']} (ID: {doc['id']}, {doc['total_pages']} pages)"
                for doc in documents
            ]
            
            selected_doc_str = st.selectbox(
                "📄 Sélectionnez le document:",
                options=doc_options,
                help="Choisissez le document pour la génération par titre"
            )
            
            # Extract document ID
            selected_doc_id = int(selected_doc_str.split("ID: ")[1].split(",")[0])
            selected_doc = next(doc for doc in documents if doc['id'] == selected_doc_id)
            
            return selected_doc
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des documents: {str(e)}")
            return None
    
    def _render_complete_title_workflow(self, selected_doc: Dict[str, Any]) -> None:
        """Render the complete title generation workflow."""
        document_id = str(selected_doc['id'])
        
        # Step 1: Analyze title structure
        title_structure = self.structure_analyzer.render_structure_analysis(document_id)
        if not title_structure:
            return
        
        # Step 2: Get and display suggestions
        st.subheader("🎯 Sélection des titres")
        suggestions = self.structure_analyzer.get_title_suggestions(document_id, title_structure)
        
        is_suggestion_selected, selected_suggestion = self.suggestions_component.render_suggestions(suggestions)
        
        # Extract suggested titles if a suggestion was selected
        suggested_h1, suggested_h2, suggested_h3, suggested_h4 = (None, None, None, None)
        if is_suggestion_selected and selected_suggestion:
            suggested_h1, suggested_h2, suggested_h3, suggested_h4 = (
                self.suggestions_component.extract_criteria_from_suggestion(selected_suggestion)
            )
        
        # Step 3: Manual title selection with hierarchical dropdowns
        final_h1, final_h2, final_h3, final_h4 = self.title_selector.render_manual_selection(
            title_structure, suggested_h1, suggested_h2, suggested_h3, suggested_h4
        )
        
        # Step 4: Preview selection and get content metrics
        if self.title_selector.has_valid_selection(final_h1, final_h2, final_h3, final_h4):
            matching_chunks = self.content_preview.render_selection_preview(
                document_id, final_h1, final_h2, final_h3, final_h4
            )
            
            if matching_chunks:
                # Calculate content metrics
                content_metrics = self.content_preview.calculate_content_metrics(matching_chunks)
                title_level, max_questions = self.content_preview.get_title_level_and_limits(
                    final_h1, final_h2, final_h3, final_h4
                )
                
                # Step 5: Generation configuration
                config = self.generation_config.render_generation_config(
                    matching_chunks, title_level, max_questions, content_metrics
                )
                
                # Step 6: Few-shot examples configuration
                examples_config = self.generation_config.render_few_shot_examples_config()
                
                # Step 7: Generation execution
                if st.button("🚀 Générer questions depuis titre", type="primary"):
                    self._execute_complete_title_generation(
                        document_id, final_h1, final_h2, final_h3, final_h4,
                        config, examples_config
                    )
    
    def _execute_complete_title_generation(
        self,
        document_id: str,
        h1: Optional[str], 
        h2: Optional[str], 
        h3: Optional[str], 
        h4: Optional[str],
        config: Dict[str, Any],
        examples_config: Dict[str, Any]
    ) -> None:
        """Execute complete title-based generation with progress tracking."""
        try:
            # Create progress container
            progress_container = st.empty()
            
            # Show initial progress
            with progress_container.container():
                st.info("🔄 Génération depuis titre en cours...")
                progress_bar = st.progress(0.0, text="Initialisation...")
                status_text = st.empty()
            
            try:
                from src.services.generation.title_based_generator import get_title_based_generator, TitleSelectionCriteria
                from src.models.schemas import GenerationConfig
                from src.models.enums import Language, Difficulty, QuestionType
                from src.services.infrastructure.progress_tracker import start_progress_session, get_progress_state
                
                # Initialize title generator
                title_generator = get_title_based_generator()
                
                # Create selection criteria
                criteria = TitleSelectionCriteria(
                    document_id=document_id,
                    h1_title=h1,
                    h2_title=h2,
                    h3_title=h3,
                    h4_title=h4
                )
                
                # Start progress tracking
                progress_session_id = f"title_{uuid.uuid4().hex[:8]}"
                start_progress_session(
                    session_id=progress_session_id,
                    total_questions=config['num_questions'],
                    initial_step=f"Génération par titre: {criteria.get_title_path()}"
                )
                
                # Create difficulty distribution
                if config['difficulty'] == "mixed":
                    difficulty_dist = {
                        Difficulty.EASY: 0.3,
                        Difficulty.MEDIUM: 0.5,
                        Difficulty.HARD: 0.2
                    }
                else:
                    difficulty_dist = {Difficulty(config['difficulty']): 1.0}
                
                # Create question type distribution
                if config['question_type'] == "mixed":
                    type_dist = {
                        QuestionType.MULTIPLE_CHOICE: 0.7,
                        QuestionType.MULTIPLE_SELECTION: 0.3
                    }
                else:
                    type_dist = {QuestionType(config['question_type'].replace('-', '_').upper()): 1.0}
                
                # Create generation configuration
                generation_config = GenerationConfig(
                    num_questions=config['num_questions'],
                    language=Language(config['language']),
                    difficulty_distribution=difficulty_dist,
                    question_types=type_dist
                )
                
                # Variables for thread communication
                questions_result = [None]
                generation_complete = [False]
                generation_error = [None]
                
                # Function to run generation in background
                def run_title_generation():
                    try:
                        logger.info(f"Starting title generation for: {criteria.get_title_path()}")
                        logger.info(f"Examples config: {examples_config}")
                        
                        session_id_param = f"title_session_{abs(hash(criteria.get_title_path())) % 10000}"
                        
                        # Generate questions with progress tracking
                        try:
                            
                            result = asyncio.run(title_generator.generate_questions_from_title(
                                criteria, generation_config, session_id_param,
                                progress_session_id=progress_session_id,
                                examples_file=examples_config.get('examples_file') if examples_config.get('use_examples') else None,
                                max_examples=examples_config.get('max_examples', 3)
                            ))
                            questions_result[0] = result
                            logger.info(f"Title generation completed: {len(result) if result else 0} questions")
                        except Exception as e:
                            logger.error(f"Title generation failed: {e}")
                            raise e
                    except Exception as e:
                        generation_error[0] = str(e)
                        logger.error(f"Title generation failed: {e}")
                    finally:
                        generation_complete[0] = True
                
                # Start generation in background thread
                generation_thread = threading.Thread(target=run_title_generation)
                generation_thread.start()
                
                # Update progress while generation runs
                while not generation_complete[0]:
                    try:
                        progress_state = get_progress_state(progress_session_id)
                        if progress_state:
                            progress_value = progress_state.get('progress_percentage', 0.0) / 100.0
                            current_step = progress_state.get('current_step', 'En cours...')
                            progress_bar.progress(progress_value, text=current_step)
                            status_text.text(f"⏳ {current_step}")
                        time.sleep(0.5)
                    except Exception:
                        pass
                
                # Wait for thread to complete
                generation_thread.join(timeout=120)  # 2 minute timeout
                
                # Process results
                progress_container.empty()
                
                if generation_error[0]:
                    st.error(f"❌ Erreur lors de la génération par titre: {generation_error[0]}")
                elif questions_result[0]:
                    questions = questions_result[0]
                    st.success(f"✅ {len(questions)} questions générées avec succès depuis les titres!")
                    
                    # Store questions in session state
                    from src.ui.core.session_state import SessionStateManager
                    SessionStateManager.set_generated_questions(questions)
                    
                    # Display questions preview
                    self._display_generated_questions_preview(questions)
                else:
                    st.warning("⚠️ Aucune question générée depuis les titres sélectionnés.")
                    
            except Exception as e:
                progress_container.empty()
                logger.error(f"Complete title generation error: {e}")
                st.error(f"❌ Erreur lors de la génération complète par titre: {str(e)}")
                
        except Exception as fatal_e:
            logger.error(f"Fatal error in title generation: {fatal_e}")
            try:
                st.error(f"❌ Erreur fatale: {fatal_e}")
            except:
                logger.error("Could not display error message")
    
    def _display_generated_questions_preview(self, questions: List[Dict[str, Any]]) -> None:
        """Display a preview of generated questions with robust error handling."""
        if not questions:
            st.warning("Aucune question à afficher.")
            return
        
        try:
            st.subheader("📋 Aperçu des questions générées")
            
            # Show summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions générées", len(questions))
            with col2:
                # Safely extract question types
                question_types = []
                for q in questions:
                    if isinstance(q, dict):
                        q_type = q.get('question_type', q.get('type', 'unknown'))
                    else:
                        q_type = getattr(q, 'question_type', getattr(q, 'type', 'unknown'))
                    question_types.append(str(q_type))
                unique_types = len(set(question_types))
                st.metric("Types de questions", unique_types)
            with col3:
                # Safely extract difficulties
                difficulties = []
                for q in questions:
                    if isinstance(q, dict):
                        difficulty = q.get('difficulty', 'unknown')
                    else:
                        difficulty = getattr(q, 'difficulty', 'unknown')
                    difficulties.append(str(difficulty))
                unique_difficulties = len(set(difficulties))
                st.metric("Niveaux de difficulté", unique_difficulties)
            
            # Display first few questions
            num_preview = min(3, len(questions))
            for i in range(num_preview):
                try:
                    question = questions[i]
                    
                    # Safely extract question text
                    if isinstance(question, dict):
                        question_text = question.get('question_text', question.get('question', 'Question non disponible'))
                        question_type = question.get('question_type', question.get('type', 'N/A'))
                        difficulty = question.get('difficulty', 'N/A')
                        options = question.get('options', [])
                        correct_answers = question.get('correct_answers', [])
                        explanation = question.get('explanation', '')
                    else:
                        # Handle QuestionCreate objects
                        question_text = getattr(question, 'question_text', getattr(question, 'question', 'Question non disponible'))
                        question_type = getattr(question, 'question_type', getattr(question, 'type', 'N/A'))
                        difficulty = getattr(question, 'difficulty', 'N/A')
                        options = getattr(question, 'options', [])
                        correct_answers = getattr(question, 'correct_answers', [])
                        explanation = getattr(question, 'explanation', '')
                    
                    # Create safe preview text
                    preview_text = str(question_text)[:100] if question_text else "Question sans texte"
                    
                    with st.expander(f"Question {i+1}: {preview_text}...", expanded=i==0):
                        st.write(f"**Question:** {question_text}")
                        st.write(f"**Type:** {question_type} | **Difficulté:** {difficulty}")
                        
                        if options:
                            st.write("**Options:**")
                            for j, option in enumerate(options):
                                try:
                                    # Handle different option formats
                                    if isinstance(option, dict):
                                        option_text = option.get('text', str(option))
                                        is_correct = option.get('is_correct', j in correct_answers)
                                    else:
                                        option_text = getattr(option, 'text', str(option))
                                        is_correct = getattr(option, 'is_correct', j in correct_answers)
                                    
                                    marker = "✅" if is_correct else "○"
                                    st.write(f"  {marker} {j+1}. {option_text}")
                                except Exception as opt_e:
                                    st.write(f"  ? {j+1}. [Erreur affichage option: {opt_e}]")
                        
                        if explanation:
                            st.write(f"**Explication:** {explanation}")
                        
                except Exception as q_e:
                    logger.error(f"Error displaying question {i+1}: {q_e}")
                    st.error(f"Erreur affichage question {i+1}: {q_e}")
            
            if len(questions) > num_preview:
                st.info(f"... et {len(questions) - num_preview} autres questions. Consultez la section 'Génération QCM' pour voir toutes les questions.")
        
        except Exception as e:
            logger.error(f"Error in _display_generated_questions_preview: {e}")
            st.error(f"Erreur lors de l'affichage des questions: {e}")
            # Fallback: just show raw count
            st.write(f"✅ {len(questions)} questions générées (aperçu non disponible)")
