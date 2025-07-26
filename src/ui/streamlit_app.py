"""
QCM Generator Pro - Streamlit Web Interface

This module provides a user-friendly web interface using Streamlit
for the QCM generation system with all features integrated.
"""

import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
from dotenv import load_dotenv
from stqdm import stqdm

# Load environment variables
load_dotenv()

from src.core.config import settings
from src.models.enums import Difficulty, ExportFormat, Language, QuestionType
from src.models.schemas import GenerationConfig, GenerationSessionCreate
from src.services.document_manager import (
    get_available_themes,
    get_document_chunks,
    get_document_manager,
    get_stored_document,
    list_stored_documents,
)
from src.services.llm_manager import generate_llm_response_sync, test_llm_connection
from src.services.pdf_processor import process_pdf, validate_pdf_file
from src.services.qcm_generator import generate_progressive_qcm
from src.services.rag_engine import (
    add_document_to_rag,
    get_question_context,
    get_rag_engine,
    switch_rag_engine,
)
from src.services.theme_extractor import extract_document_themes_sync
from src.services.validator import validate_questions_batch

logger = logging.getLogger(__name__)


class StreamlitQCMInterface:
    """Main Streamlit interface for QCM Generator Pro."""

    def __init__(self):
        """Initialize the Streamlit interface."""
        # Initialize session state
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = None
        if "generated_questions" not in st.session_state:
            st.session_state.generated_questions = []
        if "processed_documents" not in st.session_state:
            st.session_state.processed_documents = {}

    def _get_unique_filename(self, directory: Path, original_filename: str) -> Path:
        """
        Generate a unique filename to avoid conflicts while preserving original name.
        
        Args:
            directory: Target directory path
            original_filename: Original filename from upload
            
        Returns:
            Path to unique filename
        """
        # Clean filename to be filesystem-safe
        safe_filename = "".join(c for c in original_filename if c.isalnum() or c in "._- ").strip()
        if not safe_filename.lower().endswith('.pdf'):
            safe_filename += '.pdf'

        base_path = directory / safe_filename

        # If file doesn't exist, use original name
        if not base_path.exists():
            return base_path

        # If file exists, add numeric suffix
        filename_stem = base_path.stem
        filename_suffix = base_path.suffix
        counter = 1

        while True:
            new_filename = f"{filename_stem}_{counter}{filename_suffix}"
            new_path = directory / new_filename
            if not new_path.exists():
                return new_path
            counter += 1

    def upload_and_process_document(self, file) -> tuple[str, str, str]:
        """Upload and process a PDF document."""
        try:
            if file is None:
                return "❌ Aucun fichier sélectionné", "", ""

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Validation du fichier...")
            progress_bar.progress(0.1)

            # Create permanent file with original name
            upload_dir = settings.data_dir / "pdfs"
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename preserving original name
            original_filename = file.name
            file_path = self._get_unique_filename(upload_dir, original_filename)

            # Write file content
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())

            # Validate PDF content
            try:
                from src.core.validator import validate_pdf_file
                if not validate_pdf_file(file_path):
                    return "❌ Fichier PDF invalide ou corrompu", "", ""
            except ImportError:
                # Basic validation fallback
                if file_path.stat().st_size == 0:
                    return "❌ Fichier vide", "", ""

            status_text.text("Traitement du document avec DocumentManager...")
            progress_bar.progress(0.3)

            # Use DocumentManager for processing and storage
            try:
                import asyncio
                doc_manager = get_document_manager()
                document = asyncio.run(doc_manager.process_and_store_document(
                    file_path,
                    store_in_rag=True
                ))

                # Store in session state as well (for backward compatibility)
                st.session_state.processed_documents[str(document.id)] = {
                    "filename": document.filename,
                    "total_pages": document.total_pages,
                    "language": document.language,
                    "processing_status": document.processing_status,
                    "themes": []  # Will be populated from database
                }

                progress_bar.progress(0.8)
                status_text.text("Finalisation...")

                # Get themes from database
                themes = doc_manager.get_document_themes(str(document.id))
                theme_names = [theme.theme_name for theme in themes]

                progress_bar.progress(1.0)
                status_text.text("✅ Traitement terminé!")

                # File is now permanently stored in data/pdfs/ directory

                return (
                    f"✅ Document traité avec succès! ID: {document.id}",
                    f"Fichier: {document.filename} ({document.total_pages} pages)",
                    f"Thèmes détectés: {', '.join(theme_names) if theme_names else 'Aucun'}"
                )

            except Exception as e:
                logger.error(f"DocumentManager processing failed: {e}")
                return f"❌ Erreur de traitement avec DocumentManager: {str(e)}", "", ""

        except Exception as e:
            logger.error(f"Upload processing failed: {e}")
            return f"❌ Erreur lors du traitement: {str(e)}", "", ""

    def get_available_themes(self) -> list[str]:
        """Get list of available themes from stored documents."""
        try:
            doc_manager = get_document_manager()
            all_themes = doc_manager.get_all_themes()
            return [theme['name'] for theme in all_themes]
        except Exception as e:
            logger.error(f"Failed to get available themes: {e}")
            return []

    def generate_questions(
        self,
        num_questions: int,
        language: str,
        difficulty_easy: float,
        difficulty_medium: float,
        difficulty_hard: float,
        mc_ratio: float,
        ms_ratio: float,
        selected_themes: list[str]
    ) -> tuple[str, str, str]:
        """Generate QCM questions using progressive workflow."""
        try:
            if not st.session_state.processed_documents:
                return "❌ Aucun document traité", "", ""

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Configuration de la génération...")
            progress_bar.progress(0.1)

            # Create generation config
            config = GenerationConfig(
                num_questions=num_questions,
                language=Language(language),
                difficulty_distribution={
                    Difficulty.EASY: difficulty_easy / 100,
                    Difficulty.MEDIUM: difficulty_medium / 100,
                    Difficulty.HARD: difficulty_hard / 100
                },
                question_types={
                    QuestionType.MULTIPLE_CHOICE: mc_ratio / 100,
                    QuestionType.MULTIPLE_SELECTION: ms_ratio / 100
                },
                themes_filter=selected_themes if selected_themes else None
            )

            status_text.text("Préparation des thèmes...")
            progress_bar.progress(0.2)

            # Get all themes if none selected
            topics = selected_themes if selected_themes else self.get_available_themes()
            if not topics:
                topics = ["Contenu général"]

            status_text.text("Phase 1: Génération test (1 question)...")
            progress_bar.progress(0.3)

            # Create session
            session_id = str(uuid.uuid4())
            st.session_state.current_session_id = session_id

            # Handle single question vs progressive generation
            if num_questions == 1:
                status_text.text("Génération d'une question...")
                progress_bar.progress(0.7)

                # Direct generation for single question
                import asyncio

                from src.services.qcm_generator import generate_qcm_question

                question = asyncio.run(generate_qcm_question(
                    topic=topics[0] if topics else "Contenu général",
                    config=config,
                    document_ids=list(st.session_state.processed_documents.keys()),
                    session_id=session_id
                ))

                st.session_state.generated_questions = [question]

            else:
                # Progressive generation workflow
                status_text.text("Démarrage du workflow progressif...")
                progress_bar.progress(0.5)

                # Initialize session state for progressive workflow
                if "generation_session" not in st.session_state:
                    st.session_state.generation_session = {
                        "session_id": session_id,
                        "current_phase": 1,
                        "total_requested": num_questions,
                        "questions_so_far": [],
                        "status": "phase_1_ready"
                    }

                # Start progressive generation - this will be handled by a separate function
                st.session_state.generated_questions = []

                # Show progressive workflow UI
                progress_bar.empty()
                status_text.empty()
                return self._handle_progressive_generation(num_questions, config, topics, session_id)

            status_text.text("Génération terminée!")
            progress_bar.progress(1.0)

            # Prepare display info
            generation_info = f"""🎯 **Génération terminée!**
            
📊 **Statistiques**:
• Questions générées: {len(st.session_state.generated_questions)}
• Questions demandées: {num_questions}
• Taux de succès: {len(st.session_state.generated_questions)/num_questions*100:.1f}%
"""

            # Questions preview
            questions_preview = "📝 **Aperçu des questions générées:**\n\n"
            for i, question in enumerate(st.session_state.generated_questions[:3]):
                # Count correct answers
                correct_count = sum(1 for opt in question.options if hasattr(opt, 'is_correct') and opt.is_correct)

                questions_preview += f"**Question {i+1}** ({question.difficulty.value if hasattr(question.difficulty, 'value') else question.difficulty}):\n"
                questions_preview += f"{question.question_text}\n\n"
                for j, option in enumerate(question.options):
                    # Extract correct answers from QuestionOption is_correct flags
                    is_correct = option.is_correct if hasattr(option, 'is_correct') else False
                    marker = "✅" if is_correct else "▫️"
                    option_text = option.text if hasattr(option, 'text') else str(option)
                    questions_preview += f"{marker} {option_text}\n"
                questions_preview += f"\n💡 {question.explanation}\n\n---\n\n"

            if len(st.session_state.generated_questions) > 3:
                questions_preview += f"... et {len(st.session_state.generated_questions) - 3} autres questions"

            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()

            return "✅ Questions générées avec succès!", generation_info, questions_preview

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return f"❌ Erreur: {str(e)}", "", ""

    def _handle_progressive_generation(
        self,
        total_questions: int,
        config: GenerationConfig,
        topics: list[str],
        session_id: str
    ) -> tuple[str, str, str]:
        """Handle the progressive generation workflow with user validation."""
        try:
            session = st.session_state.generation_session

            # Phase 1: Generate 1 test question
            if session["status"] == "phase_1_ready":
                st.info("🔍 **Phase 1 : Question test**")
                st.write("Génération d'une question test pour valider la qualité...")

                if st.button("🚀 Générer la question test"):
                    with st.spinner("Génération en cours..."):
                        try:
                            import asyncio

                            from src.services.qcm_generator import generate_qcm_question

                            question = asyncio.run(generate_qcm_question(
                                topic=topics[0] if topics else "Contenu général",
                                config=config,
                                document_ids=list(st.session_state.processed_documents.keys()),
                                session_id=session_id
                            ))

                            session["questions_so_far"] = [question]
                            session["status"] = "phase_1_review"
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération: {str(e)}")
                            logger.error(f"Question generation failed: {e}")

                # Return status for phase 1 ready (waiting for user action)
                return "🔍 Phase 1 - En attente", "", ""

            # Phase 1: Review test question
            elif session["status"] == "phase_1_review":
                st.success("✅ Question test générée!")

                # Display the test question
                question = session["questions_so_far"][0]

                st.write("**Question test:**")
                st.write(f"**{question.question_text}**")

                for i, option in enumerate(question.options):
                    option_text = option.text if hasattr(option, 'text') else str(option)
                    is_correct = option.is_correct if hasattr(option, 'is_correct') else False
                    marker = "✅" if is_correct else "▫️"
                    st.write(f"{marker} {option_text}")

                st.write(f"💡 **Explication:** {question.explanation}")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Continuer", type="primary"):
                        batch_size = min(5, total_questions - 1)
                        session["status"] = "phase_2_ready"
                        session["next_batch_size"] = batch_size
                        st.rerun()

                with col2:
                    if st.button("❌ Recommencer"):
                        session["questions_so_far"] = []
                        session["status"] = "phase_1_ready"
                        st.rerun()

                # Return status for phase 1 review (waiting for user action)
                return "✅ Phase 1 terminée - En attente", "", ""

            # Phase 2: Generate small batch
            elif session["status"] == "phase_2_ready":
                remaining = total_questions - len(session["questions_so_far"])
                batch_size = min(5, remaining)

                st.info(f"🔍 **Phase 2 : Petit lot de {batch_size} question(s)**")
                st.write(f"Génération de {batch_size} questions supplémentaires...")

                if st.button(f"🚀 Générer {batch_size} questions"):
                    with st.spinner("Génération en cours..."):
                        # Generate batch of questions
                        import asyncio

                        from src.services.qcm_generator import QCMGenerator

                        generator = QCMGenerator()
                        questions = asyncio.run(generator.generate_questions_batch(
                            topics=topics,
                            config=config,
                            document_ids=list(st.session_state.processed_documents.keys()),
                            batch_size=batch_size,
                            session_id=session_id
                        ))

                        session["questions_so_far"].extend(questions)
                        session["status"] = "phase_2_review"
                        st.rerun()

                # Return status for phase 2 ready (waiting for user action)
                return "🔍 Phase 2 - En attente", "", ""

            # Phase 2: Review batch
            elif session["status"] == "phase_2_review":
                st.success(f"✅ {session['next_batch_size']} questions générées!")

                # Display questions from this phase
                phase_2_questions = session["questions_so_far"][1:]  # Skip first question

                with st.expander("📋 Voir les questions générées", expanded=True):
                    for i, question in enumerate(phase_2_questions):
                        st.write(f"**Question {i+2}:** {question.question_text}")

                        for j, option in enumerate(question.options):
                            option_text = option.text if hasattr(option, 'text') else str(option)
                            is_correct = option.is_correct if hasattr(option, 'is_correct') else False
                            marker = "✅" if is_correct else "▫️"
                            st.write(f"  {marker} {option_text}")

                        st.write(f"  💡 {question.explanation}")
                        st.write("---")

                remaining = total_questions - len(session["questions_so_far"])

                col1, col2 = st.columns(2)

                with col1:
                    if remaining > 0:
                        if st.button(f"✅ Continuer ({remaining} restantes)", type="primary"):
                            session["status"] = "phase_3_ready"
                            st.rerun()
                    else:
                        if st.button("✅ Terminer", type="primary"):
                            session["status"] = "completed"
                            st.rerun()

                with col2:
                    if st.button("❌ Recommencer"):
                        session["questions_so_far"] = []
                        session["status"] = "phase_1_ready"
                        st.rerun()

                # Return status for phase 2 review (waiting for user action)
                return "✅ Phase 2 terminée - En attente", "", ""

            # Phase 3: Generate remaining questions
            elif session["status"] == "phase_3_ready":
                remaining = total_questions - len(session["questions_so_far"])

                st.info(f"🔍 **Phase 3 : Questions restantes ({remaining})**")
                st.write(f"Génération des {remaining} questions restantes...")

                if st.button(f"🚀 Générer les {remaining} questions restantes"):
                    with st.spinner("Génération finale..."):
                        # Generate remaining questions
                        import asyncio

                        from src.services.qcm_generator import QCMGenerator

                        generator = QCMGenerator()
                        questions = asyncio.run(generator.generate_questions_batch(
                            topics=topics,
                            config=config,
                            document_ids=list(st.session_state.processed_documents.keys()),
                            batch_size=remaining,
                            session_id=session_id
                        ))

                        session["questions_so_far"].extend(questions)
                        session["status"] = "completed"
                        st.rerun()

                # Return status for phase 3 ready (waiting for user action)
                return "🔍 Phase 3 - En attente", "", ""

            # Completed
            elif session["status"] == "completed":
                st.success("🎉 Génération terminée!")

                # Store final questions
                st.session_state.generated_questions = session["questions_so_far"]

                # Prepare final display
                generation_info = f"""🎯 **Génération terminée!**

📊 **Statistiques**:
- Questions générées: {len(session["questions_so_far"])}/{total_questions}
- Méthode: Workflow progressif avec validation
- Langue: {config.language.value}
"""

                questions_preview = "📝 **Aperçu des premières questions:**\n\n"
                for i, question in enumerate(session["questions_so_far"][:3]):
                    questions_preview += f"**Question {i+1}**: {question.question_text}\n\n"

                if len(session["questions_so_far"]) > 3:
                    questions_preview += f"... et {len(session['questions_so_far']) - 3} autres questions"

                # Clear the session for next generation
                del st.session_state.generation_session

                return "✅ Questions générées avec succès!", generation_info, questions_preview

            # Unknown status - should not happen (silently handle)
            logger.warning(f"Unexpected session status: {session['status']}")
            return f"❌ État inconnu: {session['status']}", "", ""

        except Exception as e:
            logger.error(f"Progressive generation failed: {e}")
            return f"❌ Erreur: {str(e)}", "", ""

    def export_questions(self, export_format: str) -> tuple[str, str]:
        """Export generated questions."""
        try:
            if not st.session_state.generated_questions:
                return "❌ Aucune question à exporter", ""

            # Create export directory
            export_dir = settings.data_dir / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)

            timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")

            if export_format == "CSV (Udemy)":
                # Export as Udemy CSV v2 format
                filename = f"qcm_export_{timestamp}.csv"
                export_path = export_dir / filename

                import csv
                with open(export_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Question", "Question Type",
                        "Answer Option 1", "Explanation 1",
                        "Answer Option 2", "Explanation 2",
                        "Answer Option 3", "Explanation 3",
                        "Answer Option 4", "Explanation 4",
                        "Answer Option 5", "Explanation 5",
                        "Answer Option 6", "Explanation 6",
                        "Correct Answers", "Overall Explanation", "Domain"
                    ])

                    for question in st.session_state.generated_questions:
                        # Extract option texts and find correct answers
                        option_texts = [opt.text if hasattr(opt, 'text') else str(opt) for opt in question.options]

                        # Find all correct answer indices (1-based for CSV)
                        correct_answers = []
                        for i, opt in enumerate(question.options):
                            if hasattr(opt, 'is_correct') and opt.is_correct:
                                correct_answers.append(str(i + 1))

                        if not correct_answers:
                            correct_answers = ["1"]  # Default to first option

                        # Determine question type
                        question_type = "multiple-choice"
                        if len(correct_answers) > 1:
                            question_type = "multi-select"

                        # Prepare row data (pad options to 6 items for Udemy v2 format)
                        row_data = [
                            question.question_text,
                            question_type
                        ]

                        # Add up to 6 options with empty explanations
                        for i in range(6):
                            if i < len(option_texts):
                                row_data.extend([option_texts[i], ""])  # Option text, explanation (empty)
                            else:
                                row_data.extend(["", ""])  # Empty option and explanation

                        # Add correct answers, overall explanation, and domain
                        row_data.extend([
                            ",".join(correct_answers),
                            question.explanation or "",
                            getattr(question, 'theme', '') or "General"
                        ])

                        writer.writerow(row_data)

            else:  # JSON
                filename = f"qcm_export_{timestamp}.json"
                export_path = export_dir / filename

                export_data = {
                    "export_info": {
                        "timestamp": timestamp,
                        "questions_count": len(st.session_state.generated_questions),
                        "session_id": st.session_state.current_session_id
                    },
                    "questions": []
                }

                for question in st.session_state.generated_questions:
                    # Extract correct answers from QuestionOption is_correct flags
                    correct_answers = [j for j, opt in enumerate(question.options) if hasattr(opt, 'is_correct') and opt.is_correct]

                    # Extract option texts
                    option_texts = [opt.text if hasattr(opt, 'text') else str(opt) for opt in question.options]

                    export_data["questions"].append({
                        "question_text": question.question_text,
                        "question_type": question.question_type.value if hasattr(question.question_type, 'value') else question.question_type,
                        "difficulty": question.difficulty.value if hasattr(question.difficulty, 'value') else question.difficulty,
                        "options": option_texts,
                        "correct_answers": correct_answers,
                        "explanation": question.explanation,
                        "language": question.language.value if hasattr(question.language, 'value') else question.language
                    })

                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

            download_info = f"""📁 **Fichier exporté:**
            
📄 Nom: {filename}
📍 Chemin: {export_path}
📊 Questions: {len(st.session_state.generated_questions)}
📅 Date: {timestamp}

⬇️ Le fichier est disponible dans le dossier exports/
"""

            return "✅ Export réussi!", download_info

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return f"❌ Erreur d'export: {str(e)}", ""

    def test_llm_connection(self) -> str:
        """Test LLM connectivity."""
        try:
            from src.services.llm_manager import test_llm_connection_sync

            results = test_llm_connection_sync()

            status_info = "🔗 **État des connexions LLM:**\n\n"
            for provider, result in results.items():
                if result.get("status") == "success":
                    status_info += f"✅ **{provider.upper()}**: Connecté\n"
                    status_info += f"   Modèle: {result.get('model', 'N/A')}\n\n"
                else:
                    status_info += f"❌ **{provider.upper()}**: {result.get('error', 'Erreur')}\n\n"

            return status_info

        except Exception as e:
            return f"❌ Erreur de test: {str(e)}"

    def initialize_progressive_generation(
        self,
        num_questions: int,
        language: str,
        difficulty_easy: float,
        difficulty_medium: float,
        difficulty_hard: float,
        mc_ratio: float,
        ms_ratio: float,
        selected_themes: list[str]
    ) -> None:
        """Initialize progressive generation session."""
        from src.models.enums import Difficulty, Language, QuestionType
        from src.models.schemas import GenerationConfig

        # Create configuration
        config = GenerationConfig(
            num_questions=num_questions,
            language=Language(language),
            difficulty_distribution={
                Difficulty.EASY: difficulty_easy / 100,
                Difficulty.MEDIUM: difficulty_medium / 100,
                Difficulty.HARD: difficulty_hard / 100
            },
            question_types={
                QuestionType.MULTIPLE_CHOICE: mc_ratio / 100,
                QuestionType.MULTIPLE_SELECTION: ms_ratio / 100
            },
            themes_filter=selected_themes if selected_themes else None
        )

        # Initialize session
        session_id = f"session_{int(__import__('time').time())}"
        topics = selected_themes if selected_themes else self.get_available_themes()

        st.session_state.generation_session = {
            "status": "phase_1_ready",
            "questions_so_far": [],
            "config": config,
            "topics": topics,
            "session_id": session_id,
            "total_questions": num_questions
        }

        st.session_state.current_session_id = session_id

    def handle_progressive_generation(self) -> tuple[str, str, str]:
        """Handle progressive generation workflow."""
        if not hasattr(st.session_state, 'generation_session') or not st.session_state.generation_session:
            return "❌ Aucune session de génération active", "", ""

        session = st.session_state.generation_session
        return self._handle_progressive_generation(
            total_questions=session["total_questions"],
            config=session["config"],
            topics=session["topics"],
            session_id=session["session_id"]
        )


def _display_document_details(doc: dict[str, Any]):
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


def _display_detailed_document_info(doc: dict[str, Any]):
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


def _display_document_statistics(doc: dict[str, Any]):
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

    # Theme distribution chart
    if doc['themes']:
        st.markdown("### 📈 Distribution des thèmes")

        try:
            import pandas as pd
            import plotly.express as px

            theme_data = []
            for theme in doc['themes']:
                theme_data.append({
                    'Thème': theme['name'],
                    'Confiance': theme.get('confidence', 0)
                })

            df = pd.DataFrame(theme_data)

            # Bar chart of theme confidence
            fig = px.bar(df, x='Thème', y='Confiance',
                        title='Confiance par thème',
                        color='Confiance',
                        color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            # Fallback without plotly
            st.write("📊 Statistiques des thèmes:")
            for theme in doc['themes']:
                confidence = theme.get('confidence', 0)
                st.progress(confidence, text=f"{theme['name']}: {confidence:.2f}")

def _display_document_chunks(doc_id: int, doc: dict[str, Any]):
    """Display document chunks with navigation and search functionality."""

    try:
        # Get chunks from the database
        chunks = get_document_chunks(str(doc_id))

        if not chunks:
            st.warning("❌ Aucun chunk trouvé pour ce document")
            return

        # Basic info
        st.info(f"📊 **Total :** {len(chunks)} chunks | **Document :** {doc['filename']}")

        # Title filter functionality
        st.markdown("### 🏷️ Filtres")

        # Collect all unique titles for filtering
        all_h1_titles = set()
        all_h2_titles = set()
        all_h3_titles = set()
        all_h4_titles = set()

        for chunk in chunks:
            hierarchy = chunk.get('title_hierarchy', {})
            if hierarchy:
                if hierarchy.get('h1_title'):
                    all_h1_titles.add(hierarchy['h1_title'])
                if hierarchy.get('h2_title'):
                    all_h2_titles.add(hierarchy['h2_title'])
                if hierarchy.get('h3_title'):
                    all_h3_titles.add(hierarchy['h3_title'])
                if hierarchy.get('h4_title'):
                    all_h4_titles.add(hierarchy['h4_title'])

        # Title filters
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            selected_h1 = st.selectbox(
                "Filtrer par H1:",
                options=["Tous"] + sorted(list(all_h1_titles)),
                key=f"h1_filter_{doc_id}"
            )

            selected_h3 = st.selectbox(
                "Filtrer par H3:",
                options=["Tous"] + sorted(list(all_h3_titles)),
                key=f"h3_filter_{doc_id}"
            )

        with col_filter2:
            selected_h2 = st.selectbox(
                "Filtrer par H2:",
                options=["Tous"] + sorted(list(all_h2_titles)),
                key=f"h2_filter_{doc_id}"
            )

            selected_h4 = st.selectbox(
                "Filtrer par H4:",
                options=["Tous"] + sorted(list(all_h4_titles)),
                key=f"h4_filter_{doc_id}"
            )

        # Search functionality
        st.markdown("### 🔍 Recherche dans les chunks")
        search_query = st.text_input("Rechercher dans le contenu des chunks:", key=f"search_chunks_{doc_id}")

        # Filter chunks based on title filters and search
        filtered_chunks = chunks

        # Apply title filters
        if selected_h1 != "Tous":
            filtered_chunks = [
                chunk for chunk in filtered_chunks
                if chunk.get('title_hierarchy', {}).get('h1_title') == selected_h1
            ]

        if selected_h2 != "Tous":
            filtered_chunks = [
                chunk for chunk in filtered_chunks
                if chunk.get('title_hierarchy', {}).get('h2_title') == selected_h2
            ]

        if selected_h3 != "Tous":
            filtered_chunks = [
                chunk for chunk in filtered_chunks
                if chunk.get('title_hierarchy', {}).get('h3_title') == selected_h3
            ]

        if selected_h4 != "Tous":
            filtered_chunks = [
                chunk for chunk in filtered_chunks
                if chunk.get('title_hierarchy', {}).get('h4_title') == selected_h4
            ]

        # Apply text search filter
        if search_query:
            filtered_chunks = [
                chunk for chunk in filtered_chunks
                if search_query.lower() in chunk['chunk_text'].lower()
            ]

        # Show filter results
        total_filters_applied = sum([
            1 for filter_val in [selected_h1, selected_h2, selected_h3, selected_h4, search_query]
            if filter_val and filter_val != "Tous"
        ])

        if total_filters_applied > 0:
            st.info(f"🔍 {len(filtered_chunks)} chunk(s) trouvé(s) avec les filtres appliqués")

        if not filtered_chunks:
            st.warning("❌ Aucun chunk ne correspond à votre recherche")
            return

        # Navigation
        st.markdown("### 📝 Navigation des chunks")

        # Chunk selector with title info
        chunk_options = []
        for chunk in filtered_chunks:
            hierarchy = chunk.get('title_hierarchy', {})
            title_info = ""

            # Get the most specific title available
            if hierarchy:
                if hierarchy.get('h4_title'):
                    title_info = f" - {hierarchy['h4_title']}"
                elif hierarchy.get('h3_title'):
                    title_info = f" - {hierarchy['h3_title']}"
                elif hierarchy.get('h2_title'):
                    title_info = f" - {hierarchy['h2_title']}"
                elif hierarchy.get('h1_title'):
                    title_info = f" - {hierarchy['h1_title']}"

            chunk_label = f"Chunk {chunk['chunk_order'] + 1} ({chunk['word_count']} mots){title_info}"
            chunk_options.append(chunk_label)
        selected_index = st.selectbox(
            "Sélectionner un chunk:",
            options=range(len(filtered_chunks)),
            format_func=lambda x: chunk_options[x],
            key=f"chunk_selector_{doc_id}"
        )

        # Navigation buttons
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])

        with col_nav1:
            if st.button("⬅️ Précédent", disabled=selected_index == 0, key=f"prev_chunk_{doc_id}"):
                st.session_state[f"chunk_selector_{doc_id}"] = selected_index - 1
                st.rerun()

        with col_nav2:
            st.write(f"**Chunk {selected_index + 1} / {len(filtered_chunks)}**")

        with col_nav3:
            if st.button("➡️ Suivant", disabled=selected_index == len(filtered_chunks) - 1, key=f"next_chunk_{doc_id}"):
                st.session_state[f"chunk_selector_{doc_id}"] = selected_index + 1
                st.rerun()

        # Display selected chunk
        if 0 <= selected_index < len(filtered_chunks):
            selected_chunk = filtered_chunks[selected_index]

            # Chunk metadata
            st.markdown("### 📋 Métadonnées du chunk")
            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)

            with col_meta1:
                st.metric("Ordre", selected_chunk['chunk_order'] + 1)
            with col_meta2:
                st.metric("Mots", selected_chunk['word_count'])
            with col_meta3:
                st.metric("Caractères", selected_chunk['char_count'])
            with col_meta4:
                # Calculate estimated reading time (200 words per minute)
                reading_time = max(1, selected_chunk['word_count'] // 200)
                st.metric("Lecture (min)", reading_time)

            # Additional metadata if available
            metadata = selected_chunk.get('metadata', {})
            if any(metadata.values()):
                st.markdown("**Métadonnées supplémentaires :**")
                meta_info = []
                if metadata.get('start_char'):
                    meta_info.append(f"Position début: {metadata['start_char']}")
                if metadata.get('end_char'):
                    meta_info.append(f"Position fin: {metadata['end_char']}")
                if metadata.get('page_number'):
                    meta_info.append(f"Page: {metadata['page_number']}")
                if meta_info:
                    st.write(" | ".join(meta_info))

            # Title hierarchy information
            title_hierarchy = selected_chunk.get('title_hierarchy', {})
            if title_hierarchy and any(title_hierarchy.values()):
                st.markdown("### 📑 Hiérarchie des titres")

                # Display title breadcrumb
                title_path = title_hierarchy.get('full_path', '')
                if title_path:
                    st.info(f"📍 **Chemin:** {title_path}")

                # Display individual title levels
                col_h1, col_h2 = st.columns(2)
                col_h3, col_h4 = st.columns(2)

                with col_h1:
                    h1_title = title_hierarchy.get('h1_title')
                    if h1_title:
                        st.markdown(f"**H1:** {h1_title}")
                    else:
                        st.markdown("**H1:** *Non défini*")

                with col_h2:
                    h2_title = title_hierarchy.get('h2_title')
                    if h2_title:
                        st.markdown(f"**H2:** {h2_title}")
                    else:
                        st.markdown("**H2:** *Non défini*")

                with col_h3:
                    h3_title = title_hierarchy.get('h3_title')
                    if h3_title:
                        st.markdown(f"**H3:** {h3_title}")
                    else:
                        st.markdown("**H3:** *Non défini*")

                with col_h4:
                    h4_title = title_hierarchy.get('h4_title')
                    if h4_title:
                        st.markdown(f"**H4:** {h4_title}")
                    else:
                        st.markdown("**H4:** *Non défini*")

            # Additional metadata if available
            metadata = selected_chunk.get('metadata', {})
            if any(metadata.values()):
                st.markdown("**Métadonnées supplémentaires :**")
                meta_info = []
                if metadata.get('start_char'):
                    meta_info.append(f"Position début: {metadata['start_char']}")
                if metadata.get('end_char'):
                    meta_info.append(f"Position fin: {metadata['end_char']}")
                if metadata.get('page_number'):
                    meta_info.append(f"Page: {metadata['page_number']}")

                if meta_info:
                    st.write(" | ".join(meta_info))

            # Chunk content
            st.markdown("### 📄 Contenu du chunk")

            # Highlight search terms if search is active
            chunk_content = selected_chunk['chunk_text']
            if search_query:
                # Simple highlighting (case-insensitive)
                import re
                pattern = re.compile(re.escape(search_query), re.IGNORECASE)
                chunk_content = pattern.sub(f"**{search_query.upper()}**", chunk_content)

            # Display in a text area for better readability
            st.text_area(
                "Contenu:",
                value=selected_chunk['chunk_text'],
                height=300,
                key=f"chunk_content_{doc_id}_{selected_index}",
                disabled=True
            )

            # Export functionality
            st.markdown("### 📤 Actions")
            col_export1, col_export2 = st.columns(2)

            with col_export1:
                if st.button("📋 Copier le chunk", key=f"copy_chunk_{doc_id}_{selected_index}"):
                    # This would copy to clipboard in a real browser environment
                    st.success("✅ Contenu copié ! (Utilisez Ctrl+A puis Ctrl+C dans la zone de texte)")

            with col_export2:
                # Download chunk as text file
                chunk_filename = f"chunk_{selected_chunk['chunk_order'] + 1}_{doc['filename'].replace('.pdf', '.txt')}"
                st.download_button(
                    "💾 Télécharger chunk",
                    data=selected_chunk['chunk_text'],
                    file_name=chunk_filename,
                    mime="text/plain",
                    key=f"download_chunk_{doc_id}_{selected_index}"
                )

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des chunks: {str(e)}")
        st.exception(e)


def create_streamlit_interface():
    """Create the main Streamlit interface."""

    # Page configuration
    st.set_page_config(
        page_title="QCM Generator Pro",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .status-success {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-error {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-info {
        padding: 1rem;
        background-color: #cce5ff;
        border-left: 4px solid #0066cc;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize interface
    interface = StreamlitQCMInterface()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 QCM Generator Pro</h1>
        <p>Génération automatique de QCM multilingues à partir de documents PDF</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    tab_choice = st.sidebar.radio(
        "Choisir une section:",
        ["📄 Upload de Documents", "📚 Gestion Documents", "🎯 Génération QCM", "🏷️ Génération par Titre", "📤 Export", "⚙️ Système"]
    )

    # Tab 1: Document Upload
    if tab_choice == "📄 Upload de Documents":
        st.header("📤 Téléchargement et traitement de documents")

        col1, col2 = st.columns([2, 3])

        with col1:
            uploaded_file = st.file_uploader(
                "Sélectionner un fichier PDF",
                type=['pdf'],
                help="Uploadez un document PDF pour l'analyser et générer des questions"
            )

            if st.button("🚀 Traiter le document", type="primary", disabled=uploaded_file is None):
                with st.spinner("Traitement en cours..."):
                    status, doc_info, themes_info = interface.upload_and_process_document(uploaded_file)

                    if "✅" in status:
                        st.success(status)
                        with col2:
                            st.markdown(doc_info)
                            st.markdown(themes_info)
                    else:
                        st.error(status)

        # Display current document status
        if st.session_state.processed_documents:
            st.subheader("📋 Documents traités")
            for doc_id, doc_info in st.session_state.processed_documents.items():
                with st.expander(f"Document: {doc_info['filename']}"):
                    st.write(f"**Pages:** {doc_info['total_pages']}")
                    st.write(f"**Thèmes:** {len(doc_info.get('themes', []))}")
                    st.write(f"**Langue:** {doc_info['language']}")

    # Tab 2: Document Management
    elif tab_choice == "📚 Gestion Documents":
        st.header("📚 Gestion des documents et thèmes")

        # Configuration de la persistance
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("⚙️ Configuration de persistance")

            # Show current RAG engine type
            current_engine = get_rag_engine()
            engine_type = type(current_engine).__name__
            st.info(f"**Moteur actuel:** {engine_type}")

            # Engine switching
            new_engine_type = st.selectbox(
                "Type de moteur RAG:",
                ["simple", "chromadb"],
                index=0 if "Simple" in engine_type else 1,
                help="SimpleRAGEngine: mémoire temporaire, ChromaDBRAGEngine: persistance"
            )

            if st.button("🔄 Changer de moteur"):
                with st.spinner("Changement en cours..."):
                    success = switch_rag_engine(new_engine_type)
                    if success:
                        st.success(f"✅ Moteur changé vers {new_engine_type}")
                        st.rerun()
                    else:
                        st.error("❌ Échec du changement de moteur")

            # Migration button
            if new_engine_type == "chromadb":
                st.subheader("📊 Migration des données")
                if st.button("🚀 Migrer vers ChromaDB"):
                    with st.spinner("Migration en cours..."):
                        # This would run the migration script
                        st.info("💡 Exécuter: `python scripts/migrate_to_chromadb.py migrate`")

        with col2:
            st.subheader("📄 Documents stockés")

            # Get stored documents
            try:
                stored_docs = list_stored_documents()

                if stored_docs:
                    # Document management controls
                    col_control1, col_control2, col_control3 = st.columns([2, 1, 1])

                    with col_control1:
                        st.success(f"✅ {len(stored_docs)} document(s) trouvé(s)")

                    with col_control2:
                        # Bulk selection
                        if st.button("☑️ Sélection multiple"):
                            st.session_state.bulk_selection_mode = not st.session_state.get('bulk_selection_mode', False)
                            st.rerun()

                    with col_control3:
                        # Bulk delete
                        if st.session_state.get('bulk_selection_mode', False):
                            selected_docs = st.session_state.get('selected_docs_for_deletion', [])
                            if selected_docs and st.button(f"🗑️ Supprimer ({len(selected_docs)})"):
                                st.session_state.confirm_bulk_delete = True
                                st.rerun()

                    # Bulk deletion confirmation
                    if st.session_state.get('confirm_bulk_delete', False):
                        selected_docs = st.session_state.get('selected_docs_for_deletion', [])
                        st.error(f"⚠️ Confirmer la suppression de {len(selected_docs)} document(s) ?")
                        col_confirm1, col_confirm2 = st.columns(2)

                        with col_confirm1:
                            if st.button("✅ Oui, supprimer", type="primary"):
                                success_count = 0
                                doc_manager = get_document_manager()

                                for doc_id in selected_docs:
                                    if doc_manager.delete_document(doc_id):
                                        success_count += 1
                                        # Also remove from RAG engine
                                        try:
                                            rag_engine = get_rag_engine()
                                            if hasattr(rag_engine, 'document_chunks') and str(doc_id) in rag_engine.document_chunks:
                                                del rag_engine.document_chunks[str(doc_id)]
                                        except:
                                            pass

                                st.success(f"✅ {success_count}/{len(selected_docs)} document(s) supprimé(s)")

                                # Reset states
                                st.session_state.confirm_bulk_delete = False
                                st.session_state.selected_docs_for_deletion = []
                                st.session_state.bulk_selection_mode = False
                                st.rerun()

                        with col_confirm2:
                            if st.button("❌ Annuler"):
                                st.session_state.confirm_bulk_delete = False
                                st.rerun()

                    # Initialize session state for bulk selection
                    if 'selected_docs_for_deletion' not in st.session_state:
                        st.session_state.selected_docs_for_deletion = []

                    # Document list
                    for doc in stored_docs:
                        doc_id = doc['id']

                        # Create expander with selection checkbox if in bulk mode
                        if st.session_state.get('bulk_selection_mode', False):
                            col_check, col_expand = st.columns([0.1, 0.9])

                            with col_check:
                                is_selected = doc_id in st.session_state.selected_docs_for_deletion
                                if st.checkbox("", value=is_selected, key=f"select_{doc_id}"):
                                    if doc_id not in st.session_state.selected_docs_for_deletion:
                                        st.session_state.selected_docs_for_deletion.append(doc_id)
                                else:
                                    if doc_id in st.session_state.selected_docs_for_deletion:
                                        st.session_state.selected_docs_for_deletion.remove(doc_id)

                            with col_expand:
                                with st.expander(f"📄 {doc['filename']} ({doc['total_pages']} pages)"):
                                    _display_document_details(doc)
                        else:
                            with st.expander(f"📄 {doc['filename']} ({doc['total_pages']} pages)"):
                                _display_document_details(doc)

                                # Individual document actions
                                st.markdown("---")
                                col_actions = st.columns(5)

                                with col_actions[0]:
                                    if st.button("🎯 Utiliser", key=f"use_{doc_id}"):
                                        # Store selected document for generation
                                        st.session_state.selected_document_for_generation = doc_id
                                        st.success(f"Document {doc['filename']} sélectionné pour génération")

                                with col_actions[1]:
                                    if st.button("👁️ Détails", key=f"details_{doc_id}"):
                                        st.session_state.show_document_details = doc_id
                                        st.rerun()

                                with col_actions[2]:
                                    if st.button("📊 Stats", key=f"stats_{doc_id}"):
                                        st.session_state.show_document_stats = doc_id
                                        st.rerun()

                                with col_actions[3]:
                                    if st.button("📝 Chunks", key=f"chunks_{doc_id}"):
                                        st.session_state.show_document_chunks = doc_id
                                        st.rerun()

                                with col_actions[4]:
                                    if st.button("🗑️ Supprimer", key=f"delete_{doc_id}"):
                                        st.session_state.confirm_delete_doc = doc_id
                                        st.rerun()

                                # Individual deletion confirmation
                                if st.session_state.get('confirm_delete_doc') == doc_id:
                                    st.error(f"⚠️ Confirmer la suppression de '{doc['filename']}' ?")
                                    col_confirm1, col_confirm2 = st.columns(2)

                                    with col_confirm1:
                                        if st.button("✅ Oui, supprimer", key=f"confirm_del_{doc_id}", type="primary"):
                                            doc_manager = get_document_manager()
                                            if doc_manager.delete_document(doc_id):
                                                # Also remove from RAG engine
                                                try:
                                                    rag_engine = get_rag_engine()
                                                    if hasattr(rag_engine, 'document_chunks') and str(doc_id) in rag_engine.document_chunks:
                                                        del rag_engine.document_chunks[str(doc_id)]
                                                except:
                                                    pass

                                                st.success(f"✅ Document '{doc['filename']}' supprimé")
                                                st.session_state.confirm_delete_doc = None
                                                st.rerun()
                                            else:
                                                st.error("❌ Échec de la suppression")

                                    with col_confirm2:
                                        if st.button("❌ Annuler", key=f"cancel_del_{doc_id}"):
                                            st.session_state.confirm_delete_doc = None
                                            st.rerun()

                    # Show detailed document information if requested
                    if st.session_state.get('show_document_details'):
                        doc_id = st.session_state.show_document_details
                        doc = next((d for d in stored_docs if d['id'] == doc_id), None)
                        if doc:
                            st.subheader(f"📋 Détails: {doc['filename']}")
                            _display_detailed_document_info(doc)
                            if st.button("❌ Fermer détails"):
                                st.session_state.show_document_details = None
                                st.rerun()

                    # Show document statistics if requested
                    if st.session_state.get('show_document_stats'):
                        doc_id = st.session_state.show_document_stats
                        doc = next((d for d in stored_docs if d['id'] == doc_id), None)
                        if doc:
                            st.subheader(f"📊 Statistiques: {doc['filename']}")
                            _display_document_statistics(doc)
                            if st.button("❌ Fermer stats"):
                                st.session_state.show_document_stats = None
                                st.rerun()

                    # Show document chunks if requested
                    if st.session_state.get('show_document_chunks'):
                        doc_id = st.session_state.show_document_chunks
                        doc = next((d for d in stored_docs if d['id'] == doc_id), None)
                        if doc:
                            st.subheader(f"📝 Chunks: {doc['filename']}")
                            _display_document_chunks(doc_id, doc)
                            if st.button("❌ Fermer chunks"):
                                st.session_state.show_document_chunks = None
                                st.rerun()
                else:
                    st.warning("❌ Aucun document stocké trouvé")
                    st.info("💡 Uploadez des documents dans la section 'Upload de Documents'")

            except Exception as e:
                st.error(f"❌ Erreur lors du chargement des documents: {e}")

        # Themes management section
        st.subheader("🎯 Gestion des thèmes")

        try:
            all_themes = get_available_themes()

            if all_themes:
                st.success(f"✅ {len(all_themes)} thème(s) unique(s) trouvé(s)")

                # Theme statistics
                col_stats = st.columns(4)
                with col_stats[0]:
                    st.metric("Thèmes totaux", len(all_themes))
                with col_stats[1]:
                    avg_confidence = sum(theme['avg_confidence'] for theme in all_themes) / len(all_themes)
                    st.metric("Confiance moyenne", f"{avg_confidence:.2f}")
                with col_stats[2]:
                    total_docs = sum(theme['document_count'] for theme in all_themes)
                    st.metric("Documents avec thèmes", total_docs)
                with col_stats[3]:
                    max_usage = max(theme['document_count'] for theme in all_themes)
                    st.metric("Usage max", max_usage)

                # Theme selection interface
                st.subheader("🔍 Sélection de thèmes")

                selected_themes = st.multiselect(
                    "Thèmes disponibles pour génération:",
                    options=[theme['name'] for theme in all_themes],
                    help="Sélectionnez les thèmes à utiliser pour la génération de questions"
                )

                if selected_themes:
                    st.success(f"✅ {len(selected_themes)} thème(s) sélectionné(s)")

                    # Store selected themes in session state
                    st.session_state.selected_themes_for_generation = selected_themes

                    # Show details of selected themes
                    with st.expander("📋 Détails des thèmes sélectionnés"):
                        for theme_name in selected_themes:
                            theme_info = next(t for t in all_themes if t['name'] == theme_name)
                            st.write(f"**{theme_name}**")
                            st.write(f"  - Utilisé dans {theme_info['document_count']} document(s)")
                            st.write(f"  - Confiance moyenne: {theme_info['avg_confidence']:.2f}")
                            if theme_info['keywords']:
                                st.write(f"  - Mots-clés: {', '.join(theme_info['keywords'][:8])}")

                # Theme details table
                if st.checkbox("📊 Afficher tous les détails des thèmes"):
                    import pandas as pd

                    theme_data = []
                    for theme in all_themes:
                        theme_data.append({
                            "Thème": theme['name'],
                            "Documents": theme['document_count'],
                            "Confiance": f"{theme['avg_confidence']:.2f}",
                            "Mots-clés": ', '.join(theme['keywords'][:5]) if theme['keywords'] else "Aucun"
                        })

                    df = pd.DataFrame(theme_data)
                    st.dataframe(df, use_container_width=True)
            else:
                st.warning("❌ Aucun thème trouvé")
                st.info("💡 Traitez des documents d'abord pour extraire des thèmes")

        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des thèmes: {e}")

        # RAG Engine information
        st.subheader("🔧 Informations du moteur RAG")

        try:
            rag_engine = get_rag_engine()

            if hasattr(rag_engine, 'get_collection_stats'):
                # ChromaDB engine
                stats = rag_engine.get_collection_stats()
                st.json(stats)
            elif hasattr(rag_engine, 'document_chunks'):
                # Simple engine
                chunk_count = sum(len(chunks) for chunks in rag_engine.document_chunks.values())
                st.info(f"SimpleRAGEngine: {len(rag_engine.document_chunks)} documents, {chunk_count} chunks en mémoire")
            else:
                st.warning("Impossible d'obtenir les statistiques du moteur RAG")

        except Exception as e:
            st.error(f"❌ Erreur lors de l'obtention des statistiques RAG: {e}")

    # Tab 3: QCM Generation
    elif tab_choice == "🎯 Génération QCM":
        st.header("⚡ Génération progressive de questions")

        # Check if we have either processed documents from upload or a selected document from management
        has_processed_docs = bool(st.session_state.processed_documents)
        has_selected_doc = bool(st.session_state.get('selected_document_for_generation'))

        if not has_processed_docs and not has_selected_doc:
            st.warning("⚠️ Veuillez d'abord traiter un document dans la section 'Upload de Documents' ou sélectionner un document dans 'Gestion Documents'")
            return

        # If we have a selected document from management but no processed documents, load it
        if has_selected_doc and not has_processed_docs:
            selected_doc_id = st.session_state.selected_document_for_generation
            try:

                selected_doc = get_stored_document(str(selected_doc_id))
                if selected_doc:
                    # Create a processed_documents entry for compatibility
                    st.session_state.processed_documents = {
                        str(selected_doc_id): {
                            'filename': selected_doc.filename,
                            'file_path': selected_doc.file_path,
                            'total_pages': selected_doc.total_pages,
                            'language': selected_doc.language,
                            'processing_status': selected_doc.processing_status,
                            'doc_metadata': selected_doc.doc_metadata or {},
                            'themes': []  # Will be loaded if needed
                        }
                    }
                    st.success(f"✅ Document sélectionné : {selected_doc.filename}")
                else:
                    st.error("❌ Document sélectionné introuvable")
                    return
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du document : {e}")
                return

        # Show currently selected document(s)
        st.subheader("📄 Document(s) disponible(s) pour génération")
        if st.session_state.processed_documents:
            for doc_id, doc_info in st.session_state.processed_documents.items():
                doc_source = "📤 Upload direct" if not has_selected_doc else "📚 Gestion documents"
                st.info(f"**{doc_info['filename']}** ({doc_info['total_pages']} pages) - {doc_source}")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📊 Configuration")

            num_questions = st.slider(
                "Nombre de questions",
                min_value=1,
                max_value=50,
                value=10,
                help="Nombre total de questions à générer"
            )

            language = st.selectbox(
                "Langue",
                options=["fr", "en"],
                index=0,
                help="Langue de génération des questions"
            )

            st.subheader("🎚️ Répartition difficulté (%)")
            difficulty_easy = st.slider("Facile", 0, 100, 30)
            difficulty_medium = st.slider("Moyen", 0, 100, 50)
            difficulty_hard = st.slider("Difficile", 0, 100, 20)

            # Normalize difficulty percentages
            total_difficulty = difficulty_easy + difficulty_medium + difficulty_hard
            if total_difficulty != 100:
                st.warning(f"⚠️ Total: {total_difficulty}% (devrait être 100%)")

            st.subheader("📝 Types de questions (%)")
            mc_ratio = st.slider("Choix multiple", 0, 100, 70)
            ms_ratio = st.slider("Sélection multiple", 0, 100, 30)

            # Normalize question type percentages
            total_types = mc_ratio + ms_ratio
            if total_types != 100:
                st.warning(f"⚠️ Total: {total_types}% (devrait être 100%)")

            # Themes selection
            available_themes = interface.get_available_themes()
            if available_themes:
                selected_themes = st.multiselect(
                    "Thèmes à inclure (laisser vide = tous)",
                    options=available_themes,
                    help="Sélectionnez les thèmes spécifiques à inclure"
                )
            else:
                selected_themes = []
                st.info("Aucun thème disponible")

            if st.button("🎯 Démarrer génération progressive", type="primary"):
                # Initialize progressive generation session
                interface.initialize_progressive_generation(
                    num_questions, language,
                    difficulty_easy, difficulty_medium, difficulty_hard,
                    mc_ratio, ms_ratio,
                    selected_themes
                )
                st.rerun()

        # Progressive generation workflow
        with col2:
            if hasattr(st.session_state, 'generation_session') and st.session_state.generation_session:
                status, generation_info, questions_preview = interface.handle_progressive_generation()

                # Only show non-internal status messages
                if status.startswith("✅"):
                    st.success(status)
                elif status.startswith("❌") and not ("État de session inattendu" in status or "État inconnu" in status):
                    st.error(status)
                elif not status.startswith("❌"):  # Don't show error states, only positive/info states
                    st.info(status)

                if generation_info:
                    st.markdown(generation_info)
                if questions_preview:
                    st.markdown(questions_preview)

        with col2:
            if st.session_state.generated_questions:
                st.subheader("📝 Questions générées")
                st.info(f"Nombre de questions: {len(st.session_state.generated_questions)}")

                # Display detailed questions
                for i, question in enumerate(st.session_state.generated_questions):
                    # Count correct answers for expander title
                    correct_count = sum(1 for opt in question.options if hasattr(opt, 'is_correct') and opt.is_correct)

                    # Determine question type display info for expander title
                    question_type_display = ""
                    if hasattr(question, 'question_type'):
                        q_type = question.question_type.value if hasattr(question.question_type, 'value') else question.question_type
                        if q_type == "multiple-choice":
                            question_type_display = " (1 bonne réponse)"
                        elif q_type == "multiple-selection":
                            question_type_display = f" ({correct_count} bonnes réponses)"
                        else:
                            question_type_display = f" ({correct_count} bonne(s) réponse(s))"
                    else:
                        question_type_display = f" ({correct_count} bonne(s) réponse(s))"

                    with st.expander(f"Question {i+1} - {question.difficulty.value if hasattr(question.difficulty, 'value') else question.difficulty}{question_type_display}"):
                        # Extract theme from generation_params or use fallback
                        theme = "Thème non spécifié"
                        if hasattr(question, 'generation_params') and question.generation_params:
                            theme = question.generation_params.get('topic', theme)
                        elif hasattr(question, 'theme'):
                            theme = question.theme

                        # Display question type info prominently
                        if hasattr(question, 'question_type'):
                            q_type = question.question_type.value if hasattr(question.question_type, 'value') else question.question_type
                            if q_type == "multiple-choice":
                                st.info("🔘 **Question à choix unique** - Sélectionnez UNE seule réponse")
                            elif q_type == "multiple-selection":
                                st.info(f"☑️ **Question à choix multiples** - Sélectionnez {correct_count} réponses")
                            else:
                                st.info(f"❓ **Type de question:** {q_type} - {correct_count} bonne(s) réponse(s)")

                        st.write(f"**Thème:** {theme}")
                        st.write(f"**Question:** {question.question_text}")
                        st.write("**Options:**")

                        # Extract correct answers from QuestionOption is_correct flags
                        correct_indices = [j for j, opt in enumerate(question.options) if hasattr(opt, 'is_correct') and opt.is_correct]

                        for j, option in enumerate(question.options):
                            option_text = option.text if hasattr(option, 'text') else str(option)
                            marker = "✅" if j in correct_indices else "❌"
                            st.write(f"{marker} {option_text}")
                        st.write(f"**Explication:** {question.explanation}")

    # Tab 4: Title-based Generation
    elif tab_choice == "🏷️ Génération par Titre":
        st.header("🏷️ Génération de QCM par titre")
        
        # Check if documents are available
        stored_docs = list_stored_documents()
        
        if not stored_docs:
            st.warning("⚠️ Aucun document disponible. Veuillez d'abord traiter un document.")
            return
        
        # Document selection
        st.subheader("📄 Sélection du document")
        
        doc_options = {f"{doc['filename']} (ID: {doc['id']})": doc['id'] for doc in stored_docs}
        selected_doc_display = st.selectbox(
            "Choisir un document:",
            options=list(doc_options.keys()),
            help="Sélectionnez le document pour lequel vous souhaitez générer des questions par titre"
        )
        
        if not selected_doc_display:
            return
        
        selected_doc_id = doc_options[selected_doc_display]
        selected_doc = next(doc for doc in stored_docs if doc['id'] == selected_doc_id)
        
        st.info(f"📊 Document sélectionné: **{selected_doc['filename']}** ({selected_doc['total_pages']} pages)")
        
        # Get title structure
        try:
            from src.services.title_based_generator import get_title_based_generator
            title_generator = get_title_based_generator()
            
            # Show loading while analyzing
            with st.spinner("🔍 Analyse de la structure des titres..."):
                title_structure = title_generator.get_document_title_structure(str(selected_doc_id))
            
            if "error" in title_structure:
                st.error(f"❌ Erreur lors de l'analyse: {title_structure['error']}")
                return
            
            # Display structure overview
            st.subheader("📊 Aperçu de la structure")
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Chunks totaux", title_structure['total_chunks'])
            with col_stats2:
                st.metric("Chunks avec titres", title_structure['statistics']['chunks_with_titles'])
            with col_stats3:
                st.metric("Titres H1", len(title_structure['h1_titles']))
            
            # Title selection interface
            st.subheader("🎯 Sélection des titres")
            
            # Get suggestions
            suggestions = title_generator.get_title_suggestions(str(selected_doc_id), min_chunks=2)
            
            if suggestions:
                st.info(f"💡 {len(suggestions)} suggestions de titres avec suffisamment de contenu:")
                
                # Display suggestions as options
                suggestion_options = {}
                for i, suggestion in enumerate(suggestions):
                    label = f"[{suggestion['level']}] {suggestion['title']} ({suggestion['chunk_count']} chunks)"
                    suggestion_options[label] = suggestion
                
                selected_suggestion = st.selectbox(
                    "Suggestions automatiques:",
                    options=["Sélection manuelle"] + list(suggestion_options.keys()),
                    help="Choisissez une suggestion ou faites une sélection manuelle"
                )
                
                if selected_suggestion != "Sélection manuelle":
                    suggestion = suggestion_options[selected_suggestion]
                    st.success(f"✅ Suggestion sélectionnée: {suggestion['description']}")
                    
                    # Pre-fill manual selection based on suggestion
                    criteria = suggestion['criteria']
                    selected_h1 = criteria.h1_title
                    selected_h2 = criteria.h2_title
                    selected_h3 = criteria.h3_title
                    selected_h4 = criteria.h4_title
                else:
                    selected_h1 = None
                    selected_h2 = None
                    selected_h3 = None
                    selected_h4 = None
            else:
                st.warning("⚠️ Aucune suggestion disponible. Utilisez la sélection manuelle.")
                selected_h1 = None
                selected_h2 = None
                selected_h3 = None
                selected_h4 = None
            
            # Manual title selection
            st.subheader("🔧 Sélection manuelle")
            
            # Build title options from structure
            h1_options = ["Tous"] + list(title_structure['h1_titles'].keys())
            
            selected_h1_manual = st.selectbox(
                "Titre H1:",
                options=h1_options,
                index=h1_options.index(selected_h1) if selected_h1 and selected_h1 in h1_options else 0,
                key="manual_h1"
            )
            
            # H2 options depend on H1 selection
            h2_options = ["Tous"]
            if selected_h1_manual != "Tous" and selected_h1_manual in title_structure['h1_titles']:
                h2_options.extend(title_structure['h1_titles'][selected_h1_manual]['h2_titles'].keys())
            
            selected_h2_manual = st.selectbox(
                "Titre H2:",
                options=h2_options,
                index=h2_options.index(selected_h2) if selected_h2 and selected_h2 in h2_options else 0,
                key="manual_h2"
            )
            
            # H3 options depend on H1 and H2 selection
            h3_options = ["Tous"]
            if (selected_h1_manual != "Tous" and selected_h2_manual != "Tous" and 
                selected_h1_manual in title_structure['h1_titles'] and
                selected_h2_manual in title_structure['h1_titles'][selected_h1_manual]['h2_titles']):
                h3_options.extend(title_structure['h1_titles'][selected_h1_manual]['h2_titles'][selected_h2_manual]['h3_titles'].keys())
            
            selected_h3_manual = st.selectbox(
                "Titre H3:",
                options=h3_options,
                index=h3_options.index(selected_h3) if selected_h3 and selected_h3 in h3_options else 0,
                key="manual_h3"
            )
            
            # H4 options depend on H1, H2, and H3 selection
            h4_options = ["Tous"]
            if (selected_h1_manual != "Tous" and selected_h2_manual != "Tous" and selected_h3_manual != "Tous" and
                selected_h1_manual in title_structure['h1_titles'] and
                selected_h2_manual in title_structure['h1_titles'][selected_h1_manual]['h2_titles'] and
                selected_h3_manual in title_structure['h1_titles'][selected_h1_manual]['h2_titles'][selected_h2_manual]['h3_titles']):
                h4_options.extend(title_structure['h1_titles'][selected_h1_manual]['h2_titles'][selected_h2_manual]['h3_titles'][selected_h3_manual]['h4_titles'].keys())
            
            selected_h4_manual = st.selectbox(
                "Titre H4:",
                options=h4_options,
                index=h4_options.index(selected_h4) if selected_h4 and selected_h4 in h4_options else 0,
                key="manual_h4"
            )
            
            # Create final selection criteria
            final_h1 = selected_h1_manual if selected_h1_manual != "Tous" else None
            final_h2 = selected_h2_manual if selected_h2_manual != "Tous" else None
            final_h3 = selected_h3_manual if selected_h3_manual != "Tous" else None
            final_h4 = selected_h4_manual if selected_h4_manual != "Tous" else None
            
            # Preview selection
            if any([final_h1, final_h2, final_h3, final_h4]):
                from src.services.title_based_generator import TitleSelectionCriteria
                
                criteria = TitleSelectionCriteria(
                    document_id=str(selected_doc_id),
                    h1_title=final_h1,
                    h2_title=final_h2,
                    h3_title=final_h3,
                    h4_title=final_h4
                )
                
                # Get matching chunks preview
                matching_chunks = title_generator.get_chunks_for_title_selection(criteria)
                
                st.subheader("📋 Aperçu de la sélection")
                st.info(f"📍 **Chemin sélectionné:** {criteria.get_title_path()}")
                st.info(f"📊 **Chunks correspondants:** {len(matching_chunks)}")
                
                if len(matching_chunks) < 2:
                    st.warning("⚠️ Peu de contenu disponible pour cette sélection. Considérez élargir votre sélection.")
                elif len(matching_chunks) > 50:
                    st.warning("⚠️ Beaucoup de contenu sélectionné. Considérez affiner votre sélection pour de meilleurs résultats.")
                
                # Generation configuration
                st.subheader("⚙️ Configuration de génération")
                
                col_config1, col_config2 = st.columns(2)
                
                with col_config1:
                    num_questions_title = st.slider(
                        "Nombre de questions:",
                        min_value=1,
                        max_value=min(20, max(5, len(matching_chunks) // 3)),
                        value=min(5, max(1, len(matching_chunks) // 5)),
                        help="Recommandé: 1 question pour 3-5 chunks"
                    )
                    
                    language_title = st.selectbox(
                        "Langue:",
                        options=["fr", "en"],
                        index=0
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
                
                # Generate button
                if st.button("🚀 Générer questions depuis titre", type="primary"):
                    with st.spinner("🔄 Génération en cours..."):
                        try:
                            from src.models.schemas import GenerationConfig
                            from src.models.enums import Language, Difficulty, QuestionType
                            
                            # Create configuration
                            if difficulty_title == "mixed":
                                difficulty_dist = {
                                    Difficulty.EASY: 0.3,
                                    Difficulty.MEDIUM: 0.5,
                                    Difficulty.HARD: 0.2
                                }
                            else:
                                difficulty_dist = {Difficulty(difficulty_title): 1.0}
                            
                            if question_type_title == "mixed":
                                type_dist = {
                                    QuestionType.MULTIPLE_CHOICE: 0.7,
                                    QuestionType.MULTIPLE_SELECTION: 0.3
                                }
                            else:
                                type_dist = {QuestionType(question_type_title.replace('-', '_').upper()): 1.0}
                            
                            config = GenerationConfig(
                                num_questions=num_questions_title,
                                language=Language(language_title),
                                difficulty_distribution=difficulty_dist,
                                question_types=type_dist
                            )
                            
                            # Generate questions
                            import asyncio
                            questions = asyncio.run(title_generator.generate_questions_from_title(
                                criteria, config, f"title_session_{abs(hash(criteria.get_title_path())) % 10000}"
                            ))
                            
                            if questions:
                                # Store in session state
                                st.session_state.generated_questions = questions
                                st.session_state.current_session_id = f"title_{abs(hash(criteria.get_title_path())) % 10000}"
                                
                                st.success(f"✅ {len(questions)} questions générées avec succès!")
                                
                                # Display generated questions
                                st.subheader("📝 Questions générées")
                                
                                for i, question in enumerate(questions):
                                    with st.expander(f"Question {i+1} - {question.difficulty.value if hasattr(question.difficulty, 'value') else question.difficulty}"):
                                        st.write(f"**Titre source:** {criteria.get_title_path()}")
                                        st.write(f"**Question:** {question.question_text}")
                                        st.write("**Options:**")
                                        
                                        for j, option in enumerate(question.options):
                                            option_text = option.text if hasattr(option, 'text') else str(option)
                                            is_correct = option.is_correct if hasattr(option, 'is_correct') else False
                                            marker = "✅" if is_correct else "❌"
                                            st.write(f"{marker} {option_text}")
                                        
                                        st.write(f"**Explication:** {question.explanation}")
                                
                                st.info("💡 Les questions ont été ajoutées à la session. Vous pouvez les exporter dans la section 'Export'.")
                            else:
                                st.error("❌ Aucune question générée. Vérifiez la sélection et réessayez.")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération: {str(e)}")
                            logger.error(f"Title-based generation failed: {e}")
            else:
                st.info("👆 Sélectionnez au moins un niveau de titre pour continuer")
        
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du service de génération par titre: {str(e)}")
            logger.error(f"Failed to load title-based generator: {e}")

    # Tab 5: Export
    elif tab_choice == "📤 Export":
        st.header("💾 Export des questions générées")

        if not st.session_state.generated_questions:
            st.warning("⚠️ Aucune question à exporter. Générez d'abord des questions.")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            export_format = st.selectbox(
                "Format d'export",
                options=["CSV (Udemy)", "JSON"],
                help="Choisissez le format d'export souhaité"
            )

            if st.button("📁 Exporter", type="primary"):
                with col2:
                    with st.spinner("Export en cours..."):
                        status, download_info = interface.export_questions(export_format)

                        if "✅" in status:
                            st.success(status)
                            st.markdown(download_info)
                        else:
                            st.error(status)

        # Display export preview
        st.subheader("👀 Aperçu des données à exporter")
        st.info(f"Questions prêtes à l'export: {len(st.session_state.generated_questions)}")

        if st.checkbox("Afficher le détail"):
            for i, question in enumerate(st.session_state.generated_questions[:3]):
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
            if len(st.session_state.generated_questions) > 3:
                st.write(f"... et {len(st.session_state.generated_questions) - 3} autres questions")

    # Tab 5: System Status
    elif tab_choice == "⚙️ Système":
        st.header("🔧 État du système")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔗 Tester connexions LLM"):
                with st.spinner("Test des connexions..."):
                    llm_status = interface.test_llm_connection()
                    st.markdown(llm_status)

        with col2:
            st.subheader("📊 Statistiques de session")
            st.metric("Documents traités", len(st.session_state.processed_documents))
            st.metric("Questions générées", len(st.session_state.generated_questions))

        # System information
        st.subheader("📋 Informations système")
        st.markdown("""
        <div class="status-info">
        <ul>
        <li><strong>Version:</strong> QCM Generator Pro v0.1.0</li>
        <li><strong>Interface:</strong> Streamlit</li>
        <li><strong>Fonctionnalités:</strong> Upload PDF, Génération LLM, Export CSV/JSON</li>
        <li><strong>Support:</strong> Français, Anglais</li>
        <li><strong>Mode:</strong> Interface Web Moderne</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Debug information (if needed)
        if st.checkbox("Mode debug"):
            st.subheader("🐛 Informations de débogage")
            st.write("**Session State:**")
            st.json({
                "current_session_id": st.session_state.current_session_id,
                "processed_documents_count": len(st.session_state.processed_documents),
                "generated_questions_count": len(st.session_state.generated_questions)
            })


def launch_streamlit_app():
    """Launch the Streamlit application."""
    logger.info("Starting Streamlit interface...")

    try:
        create_streamlit_interface()

    except Exception as e:
        logger.error(f"Failed to start Streamlit interface: {e}")
        st.error(f"Erreur d'initialisation: {str(e)}")
        raise


if __name__ == "__main__":
    # Launch the Streamlit interface
    launch_streamlit_app()
