"""
Streamlit Progress Components

UI components for displaying real-time progress of question generation.
"""

import streamlit as st
import time
from typing import Optional, Dict, Any
from datetime import datetime

from src.services.infrastructure.progress_tracker import get_progress_tracker, ProgressState, ProgressStatus


class ProgressDisplay:
    """Streamlit component for displaying generation progress."""
    
    def __init__(self, session_id: str):
        """
        Initialize progress display.
        
        Args:
            session_id: Session ID to track
        """
        self.session_id = session_id
        self.tracker = get_progress_tracker()
        
    def render_progress_bar(self, container=None) -> None:
        """
        Render a progress bar with current status.
        
        Args:
            container: Streamlit container to render in (optional)
        """
        progress_state = self.tracker.get_progress(self.session_id)
        
        if not progress_state:
            if container:
                container.warning("Session de progression non trouvée")
            else:
                st.warning("Session de progression non trouvée")
            return
            
        target_container = container if container else st
        
        # Progress bar
        progress_bar = target_container.progress(
            progress_state.progress_percentage / 100.0,
            text=f"Questions générées: {progress_state.processed_questions}/{progress_state.total_questions}"
        )
        
        # Status indicator
        status_color = self._get_status_color(progress_state.status)
        target_container.markdown(
            f"**Statut:** <span style='color: {status_color}'>{self._get_status_text(progress_state.status)}</span>",
            unsafe_allow_html=True
        )
        
        # Current step
        if progress_state.current_step:
            target_container.text(f"Étape actuelle: {progress_state.current_step}")
            
        # Time information
        if progress_state.elapsed_time:
            elapsed_str = self._format_duration(progress_state.elapsed_time)
            target_container.text(f"Temps écoulé: {elapsed_str}")
            
            if progress_state.estimated_time_remaining:
                remaining_str = self._format_duration(progress_state.estimated_time_remaining)
                target_container.text(f"Temps estimé restant: {remaining_str}")
        
        # Error message if any
        if progress_state.error_message:
            target_container.error(f"Erreur: {progress_state.error_message}")
    
    def render_detailed_progress(self, container=None) -> None:
        """
        Render detailed progress information.
        
        Args:
            container: Streamlit container to render in (optional)
        """
        progress_state = self.tracker.get_progress(self.session_id)
        
        if not progress_state:
            if container:
                container.warning("Session de progression non trouvée")
            else:
                st.warning("Session de progression non trouvée")
            return
            
        target_container = container if container else st
        
        # Create columns for metrics
        col1, col2, col3, col4 = target_container.columns(4)
        
        with col1:
            st.metric(
                "Questions générées",
                f"{progress_state.processed_questions}",
                f"sur {progress_state.total_questions}"
            )
            
        with col2:
            st.metric(
                "Progression",
                f"{progress_state.progress_percentage:.1f}%"
            )
            
        with col3:
            if progress_state.elapsed_time:
                questions_per_minute = (progress_state.processed_questions / progress_state.elapsed_time) * 60 if progress_state.elapsed_time > 0 else 0
                st.metric(
                    "Vitesse",
                    f"{questions_per_minute:.1f} q/min"
                )
            else:
                st.metric("Vitesse", "N/A")
                
        with col4:
            if progress_state.estimated_time_remaining:
                remaining_str = self._format_duration(progress_state.estimated_time_remaining)
                st.metric("Temps restant", remaining_str)
            else:
                st.metric("Temps restant", "N/A")
                
        # Progress bar
        target_container.progress(
            progress_state.progress_percentage / 100.0,
            text=progress_state.current_step or "En cours..."
        )
        
        # Additional details in expandable section
        with target_container.expander("Détails de la session"):
            st.text(f"ID de session: {progress_state.session_id}")
            st.text(f"Statut: {self._get_status_text(progress_state.status)}")
            
            if progress_state.start_time:
                st.text(f"Démarré à: {progress_state.start_time.strftime('%H:%M:%S')}")
                
            if progress_state.end_time:
                st.text(f"Terminé à: {progress_state.end_time.strftime('%H:%M:%S')}")
                
            if progress_state.metadata:
                st.json(progress_state.metadata)
    
    def render_live_progress(
        self, 
        auto_refresh_seconds: int = 2,
        container=None
    ) -> Optional[ProgressState]:
        """
        Render live updating progress.
        
        Args:
            auto_refresh_seconds: Auto refresh interval in seconds
            container: Streamlit container to render in (optional)
            
        Returns:
            Current progress state
        """
        target_container = container if container else st
        
        # Create placeholder for content
        placeholder = target_container.empty()
        
        # Get current state
        progress_state = self.tracker.get_progress(self.session_id)
        
        if not progress_state:
            placeholder.warning("Session de progression non trouvée")
            return None
            
        # Render current progress in placeholder
        with placeholder.container():
            self.render_detailed_progress()
            
        # Auto-refresh if still running
        if progress_state.status in [ProgressStatus.PENDING, ProgressStatus.RUNNING]:
            time.sleep(auto_refresh_seconds)
            st.rerun()
            
        return progress_state
    
    def _get_status_color(self, status: ProgressStatus) -> str:
        """Get color for status display."""
        color_map = {
            ProgressStatus.PENDING: "#FFA500",      # Orange
            ProgressStatus.RUNNING: "#1E90FF",      # Blue
            ProgressStatus.COMPLETED: "#32CD32",    # Green
            ProgressStatus.FAILED: "#DC143C",       # Red
            ProgressStatus.CANCELLED: "#808080"     # Gray
        }
        return color_map.get(status, "#000000")
    
    def _get_status_text(self, status: ProgressStatus) -> str:
        """Get text for status display."""
        text_map = {
            ProgressStatus.PENDING: "En attente",
            ProgressStatus.RUNNING: "En cours",
            ProgressStatus.COMPLETED: "Terminé",
            ProgressStatus.FAILED: "Erreur",
            ProgressStatus.CANCELLED: "Annulé"
        }
        return text_map.get(status, status.value)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes:.0f}m {remaining_seconds:.0f}s"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {remaining_minutes:.0f}m"


class ProgressSidebar:
    """Sidebar component for monitoring all active progress sessions."""
    
    def __init__(self):
        """Initialize progress sidebar."""
        self.tracker = get_progress_tracker()
    
    def render(self) -> None:
        """Render progress sidebar."""
        st.sidebar.header("📊 Progression des générations")
        
        active_sessions = self.tracker.get_all_active_sessions()
        
        if not active_sessions:
            st.sidebar.info("Aucune génération en cours")
            return
            
        for session_id, state in active_sessions.items():
            with st.sidebar.expander(f"Session: {session_id[:8]}..."):
                # Progress bar
                st.progress(
                    state.progress_percentage / 100.0,
                    text=f"{state.processed_questions}/{state.total_questions}"
                )
                
                # Status
                status_text = self._get_status_text(state.status)
                st.text(f"Statut: {status_text}")
                
                # Current step
                if state.current_step:
                    st.text(f"Étape: {state.current_step}")
                
                # Time info
                if state.elapsed_time:
                    elapsed_str = self._format_duration(state.elapsed_time)
                    st.text(f"Temps: {elapsed_str}")
                
                # Cancel button for running sessions
                if state.status == ProgressStatus.RUNNING:
                    if st.button(f"Annuler", key=f"cancel_{session_id}"):
                        self.tracker.cancel_session(session_id)
                        st.rerun()
    
    def _get_status_text(self, status: ProgressStatus) -> str:
        """Get text for status display."""
        text_map = {
            ProgressStatus.PENDING: "En attente",
            ProgressStatus.RUNNING: "En cours",
            ProgressStatus.COMPLETED: "Terminé",
            ProgressStatus.FAILED: "Erreur",
            ProgressStatus.CANCELLED: "Annulé"
        }
        return text_map.get(status, status.value)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes:.0f}m"
        else:
            hours = seconds // 3600
            return f"{hours:.0f}h"


def show_progress_metrics(session_id: str) -> None:
    """
    Display progress metrics for a session.
    
    Args:
        session_id: Session ID to display metrics for
    """
    tracker = get_progress_tracker()
    stats = tracker.get_session_stats(session_id)
    
    if not stats:
        st.warning("Statistiques de session non disponibles")
        return
    
    st.subheader("📈 Métriques de génération")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Questions générées",
            stats["processed_questions"],
            f"sur {stats['total_questions']}"
        )
    
    with col2:
        st.metric(
            "Progression",
            f"{stats['progress_percentage']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Vitesse moyenne",
            f"{stats['questions_per_minute']:.1f} q/min"
        )
    
    # Time metrics
    if stats["elapsed_time"]:
        col4, col5 = st.columns(2)
        
        with col4:
            elapsed_display = ProgressDisplay("")._format_duration(stats["elapsed_time"])
            st.metric("Temps écoulé", elapsed_display)
        
        with col5:
            if stats["estimated_time_remaining"]:
                remaining_display = ProgressDisplay("")._format_duration(stats["estimated_time_remaining"])
                st.metric("Temps restant estimé", remaining_display)
            else:
                st.metric("Temps restant estimé", "N/A")


def create_progress_placeholder() -> Dict[str, Any]:
    """
    Create placeholders for progress display.
    
    Returns:
        Dictionary with placeholder objects
    """
    return {
        "progress_bar": st.empty(),
        "status_text": st.empty(),
        "metrics": st.empty(),
        "details": st.empty()
    }


def update_progress_placeholder(
    placeholders: Dict[str, Any], 
    session_id: str
) -> Optional[ProgressState]:
    """
    Update progress placeholders with current state.
    
    Args:
        placeholders: Dictionary of placeholder objects
        session_id: Session ID to track
        
    Returns:
        Current progress state
    """
    tracker = get_progress_tracker()
    state = tracker.get_progress(session_id)
    
    if not state:
        placeholders["status_text"].warning("Session de progression non trouvée")
        return None
    
    # Update progress bar
    placeholders["progress_bar"].progress(
        state.progress_percentage / 100.0,
        text=f"Questions générées: {state.processed_questions}/{state.total_questions}"
    )
    
    # Update status
    status_color = ProgressDisplay("")._get_status_color(state.status)
    status_text = ProgressDisplay("")._get_status_text(state.status)
    placeholders["status_text"].markdown(
        f"**Statut:** <span style='color: {status_color}'>{status_text}</span> - {state.current_step}",
        unsafe_allow_html=True
    )
    
    # Update metrics
    with placeholders["metrics"].container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Questions", f"{state.processed_questions}/{state.total_questions}")
        
        with col2:
            st.metric("Progression", f"{state.progress_percentage:.1f}%")
        
        with col3:
            if state.elapsed_time:
                qpm = (state.processed_questions / state.elapsed_time) * 60 if state.elapsed_time > 0 else 0
                st.metric("Vitesse", f"{qpm:.1f} q/min")
    
    return state