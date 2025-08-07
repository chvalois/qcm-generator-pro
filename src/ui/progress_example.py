"""
Example of how to integrate progress tracking in Streamlit UI.

This file shows how to use the progress tracking system in a Streamlit interface
for real-time monitoring of question generation.
"""

import streamlit as st
import asyncio
import time
from typing import Optional

from src.services.generation.enhanced_qcm_generator import EnhancedQCMGenerator, GenerationMode
from src.models.schemas import GenerationConfig
from src.models.enums import Language, Difficulty, QuestionType
from src.ui.progress_components import ProgressDisplay, ProgressSidebar, show_progress_metrics
from src.services.infrastructure.progress_tracker import get_progress_tracker, ProgressStatus


def example_generation_with_progress():
    """Example function showing how to integrate progress tracking."""
    
    st.title("🎯 Génération de Questions avec Suivi en Temps Réel")
    
    # Sidebar for monitoring all sessions
    progress_sidebar = ProgressSidebar()
    progress_sidebar.render()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Configuration de la génération")
        
        # Generation parameters
        num_questions = st.slider("Nombre de questions", 1, 50, 10)
        language = st.selectbox("Langue", ["fr", "en"], index=0)
        difficulty = st.selectbox("Difficulté", ["easy", "medium", "hard", "mixed"], index=1)
        
        # Document selection (mockup)
        document_id = st.selectbox("Document", [1, 2, 3], index=0)
        
        # Generate button
        if st.button("🚀 Démarrer la génération", type="primary"):
            # Create session ID
            session_id = f"demo_{int(time.time())}"
            
            # Store session ID in session state for progress tracking
            st.session_state.current_session_id = session_id
            st.session_state.generation_running = True
            
            # Start generation (this would be async in real implementation)
            start_async_generation(session_id, num_questions, document_id)
    
    with col2:
        st.header("📊 Progression")
        
        # Show progress if generation is running
        if hasattr(st.session_state, 'current_session_id') and st.session_state.get('generation_running', False):
            display_live_progress(st.session_state.current_session_id)


def start_async_generation(session_id: str, num_questions: int, document_id: int):
    """
    Start asynchronous generation with progress tracking.
    
    Note: In a real Streamlit app, you would use st.session_state to manage
    the async execution and possibly use threading or a task queue.
    """
    
    # Create configuration
    config = GenerationConfig(
        num_questions=num_questions,
        language=Language.FR,
        difficulty=Difficulty.MEDIUM,
        question_types={QuestionType.UNIQUE_CHOICE: 0.7, QuestionType.MULTIPLE_SELECTION: 0.3},
        temperature=0.7,
        max_tokens=1000
    )
    
    # Show starting message
    st.info(f"✨ Génération démarrée - Session: {session_id[:8]}...")
    
    # In a real app, you would run this asynchronously
    # For demo purposes, we'll simulate the process
    simulate_generation_process(session_id, num_questions)


def simulate_generation_process(session_id: str, num_questions: int):
    """
    Simulate the generation process for demonstration.
    
    In a real implementation, this would be handled by the actual generator.
    """
    from src.services.progress_tracker import start_progress_session, update_progress, complete_progress_session
    
    # Start progress session
    start_progress_session(session_id, num_questions, "Simulation démarrée")
    
    # Simulate progress updates
    placeholder = st.empty()
    
    for i in range(num_questions):
        time.sleep(0.5)  # Simulate processing time
        
        update_progress(
            session_id=session_id,
            processed_questions=i + 1,
            current_step=f"Génération question {i + 1}/{num_questions}",
            metadata={"simulated": True, "step": i + 1}
        )
        
        # Update display
        with placeholder.container():
            display_current_progress(session_id)
    
    # Complete session
    complete_progress_session(session_id, "Génération terminée")
    st.session_state.generation_running = False
    st.success("🎉 Génération terminée avec succès!")


def display_live_progress(session_id: str):
    """Display live progress for a session."""
    
    progress_display = ProgressDisplay(session_id)
    
    # Create container for live updates
    progress_container = st.container()
    
    with progress_container:
        # Get current state
        tracker = get_progress_tracker()
        state = tracker.get_progress(session_id)
        
        if state:
            # Show detailed progress
            progress_display.render_detailed_progress()
            
            # Auto-refresh if still running
            if state.status in [ProgressStatus.PENDING, ProgressStatus.RUNNING]:
                # Add refresh button or auto-refresh
                if st.button("🔄 Actualiser", key=f"refresh_{session_id}"):
                    st.rerun()
                
                # Auto-refresh every 2 seconds
                time.sleep(2)
                st.rerun()
            
            # Show metrics when completed
            if state.status == ProgressStatus.COMPLETED:
                st.balloons()
                show_progress_metrics(session_id)
                
        else:
            st.warning("Session de progression non trouvée")


def display_current_progress(session_id: str):
    """Display current progress state."""
    
    tracker = get_progress_tracker()
    state = tracker.get_progress(session_id)
    
    if not state:
        st.warning("Session non trouvée")
        return
    
    # Progress bar
    st.progress(
        state.progress_percentage / 100.0,
        text=f"Questions: {state.processed_questions}/{state.total_questions}"
    )
    
    # Current step
    st.text(state.current_step)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Progression", f"{state.progress_percentage:.1f}%")
    
    with col2:
        if state.elapsed_time:
            st.metric("Temps écoulé", f"{state.elapsed_time:.1f}s")
    
    with col3:
        if state.estimated_time_remaining:
            st.metric("Temps restant", f"{state.estimated_time_remaining:.1f}s")


def demo_multiple_sessions():
    """Demo showing multiple concurrent sessions."""
    
    st.header("🔄 Sessions multiples")
    
    if st.button("Démarrer 3 sessions simultanées"):
        for i in range(3):
            session_id = f"multi_{i}_{int(time.time())}"
            st.write(f"Démarrage session {i+1}: {session_id}")
            
            # Start sessions with different configurations
            simulate_generation_process(session_id, 5 + i * 2)


def demo_progress_callbacks():
    """Demo showing how to use progress callbacks."""
    
    st.header("📞 Callbacks de progression")
    
    def progress_callback(state):
        """Example callback function."""
        st.sidebar.success(f"Callback: {state.processed_questions} questions générées")
    
    tracker = get_progress_tracker()
    
    if st.button("Démarrer avec callback"):
        session_id = f"callback_{int(time.time())}"
        
        # Register callback
        tracker.register_callback(session_id, progress_callback)
        
        # Start generation
        simulate_generation_process(session_id, 8)


# Main app structure for demo
def main():
    """Main demo application."""
    
    st.set_page_config(
        page_title="Progress Tracking Demo",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Démonstration du Suivi de Progression")
    
    tab1, tab2, tab3 = st.tabs(["Génération Simple", "Sessions Multiples", "Callbacks"])
    
    with tab1:
        example_generation_with_progress()
    
    with tab2:
        demo_multiple_sessions()
    
    with tab3:
        demo_progress_callbacks()


if __name__ == "__main__":
    main()