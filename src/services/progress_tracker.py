"""
Progress Tracker Service

Handles real-time progress tracking for question generation.
Provides thread-safe progress updates that can be consumed by the UI.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ProgressStatus(str, Enum):
    """Status of a progress tracker."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressState:
    """State of a progress tracker."""
    session_id: str
    total_questions: int
    processed_questions: int = 0
    current_step: str = ""
    status: ProgressStatus = ProgressStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_questions == 0:
            return 0.0
        return min(100.0, (self.processed_questions / self.total_questions) * 100)
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Calculate elapsed time in seconds."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if not self.start_time or self.processed_questions == 0:
            return None
        
        elapsed = self.elapsed_time
        if not elapsed:
            return None
            
        questions_per_second = self.processed_questions / elapsed
        remaining_questions = self.total_questions - self.processed_questions
        
        if questions_per_second > 0:
            return remaining_questions / questions_per_second
        return None


class ProgressTracker:
    """
    Thread-safe progress tracker for question generation.
    
    Supports real-time updates and can be consumed by multiple UI components.
    """
    
    def __init__(self):
        """Initialize progress tracker."""
        self._sessions: Dict[str, ProgressState] = {}
        self._lock = threading.Lock()
        self._callbacks: Dict[str, list[Callable[[ProgressState], None]]] = {}
        
    def start_session(
        self, 
        session_id: str, 
        total_questions: int,
        initial_step: str = "Initialisation"
    ) -> ProgressState:
        """
        Start a new progress tracking session.
        
        Args:
            session_id: Unique session identifier
            total_questions: Total number of questions to generate
            initial_step: Initial step description
            
        Returns:
            The created progress state
        """
        with self._lock:
            state = ProgressState(
                session_id=session_id,
                total_questions=total_questions,
                current_step=initial_step,
                status=ProgressStatus.RUNNING,
                start_time=datetime.now()
            )
            self._sessions[session_id] = state
            self._callbacks[session_id] = []
            
        logger.info(f"Started progress session {session_id}: {total_questions} questions")
        self._notify_callbacks(session_id, state)
        return state
    
    def update_progress(
        self, 
        session_id: str, 
        processed_questions: Optional[int] = None,
        current_step: Optional[str] = None,
        increment: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProgressState]:
        """
        Update progress for a session.
        
        Args:
            session_id: Session identifier
            processed_questions: Total processed questions (absolute)
            current_step: Current step description
            increment: Number of questions to increment (relative)
            metadata: Additional metadata to update
            
        Returns:
            Updated progress state or None if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Progress session {session_id} not found")
                return None
                
            state = self._sessions[session_id]
            
            # Update processed questions
            if processed_questions is not None:
                state.processed_questions = processed_questions
            elif increment > 0:
                state.processed_questions += increment
                
            # Update current step
            if current_step is not None:
                state.current_step = current_step
                
            # Update metadata
            if metadata:
                state.metadata.update(metadata)
                
            # Check if completed
            if state.processed_questions >= state.total_questions:
                state.status = ProgressStatus.COMPLETED
                state.end_time = datetime.now()
                
        logger.debug(f"Updated progress {session_id}: {state.processed_questions}/{state.total_questions}")
        self._notify_callbacks(session_id, state)
        return state
    
    def increment_progress(
        self, 
        session_id: str, 
        current_step: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProgressState]:
        """
        Increment progress by 1 question.
        
        Args:
            session_id: Session identifier
            current_step: Current step description
            metadata: Additional metadata
            
        Returns:
            Updated progress state or None if session not found
        """
        return self.update_progress(
            session_id=session_id,
            current_step=current_step,
            increment=1,
            metadata=metadata
        )
    
    def complete_session(
        self, 
        session_id: str, 
        final_step: str = "Terminé"
    ) -> Optional[ProgressState]:
        """
        Mark a session as completed.
        
        Args:
            session_id: Session identifier
            final_step: Final step description
            
        Returns:
            Final progress state or None if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Progress session {session_id} not found")
                return None
                
            state = self._sessions[session_id]
            state.status = ProgressStatus.COMPLETED
            state.current_step = final_step
            state.end_time = datetime.now()
            
        logger.info(f"Completed progress session {session_id}: {state.processed_questions}/{state.total_questions}")
        self._notify_callbacks(session_id, state)
        return state
    
    def fail_session(
        self, 
        session_id: str, 
        error_message: str,
        error_step: str = "Erreur"
    ) -> Optional[ProgressState]:
        """
        Mark a session as failed.
        
        Args:
            session_id: Session identifier
            error_message: Error description
            error_step: Error step description
            
        Returns:
            Final progress state or None if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Progress session {session_id} not found")
                return None
                
            state = self._sessions[session_id]
            state.status = ProgressStatus.FAILED
            state.current_step = error_step
            state.error_message = error_message
            state.end_time = datetime.now()
            
        logger.error(f"Failed progress session {session_id}: {error_message}")
        self._notify_callbacks(session_id, state)
        return state
    
    def get_progress(self, session_id: str) -> Optional[ProgressState]:
        """
        Get current progress state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Progress state or None if session not found
        """
        with self._lock:
            return self._sessions.get(session_id)
    
    def cancel_session(self, session_id: str) -> Optional[ProgressState]:
        """
        Cancel a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Final progress state or None if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Progress session {session_id} not found")
                return None
                
            state = self._sessions[session_id]
            state.status = ProgressStatus.CANCELLED
            state.current_step = "Annulé"
            state.end_time = datetime.now()
            
        logger.info(f"Cancelled progress session {session_id}")
        self._notify_callbacks(session_id, state)
        return state
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Remove a session from tracking.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was removed, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                if session_id in self._callbacks:
                    del self._callbacks[session_id]
                logger.debug(f"Cleaned up progress session {session_id}")
                return True
            return False
    
    def register_callback(
        self, 
        session_id: str, 
        callback: Callable[[ProgressState], None]
    ) -> bool:
        """
        Register a callback for progress updates.
        
        Args:
            session_id: Session identifier
            callback: Function to call on progress updates
            
        Returns:
            True if registered, False if session not found
        """
        with self._lock:
            if session_id not in self._callbacks:
                return False
            self._callbacks[session_id].append(callback)
            return True
    
    def _notify_callbacks(self, session_id: str, state: ProgressState) -> None:
        """Notify all registered callbacks for a session."""
        callbacks = self._callbacks.get(session_id, [])
        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_all_active_sessions(self) -> Dict[str, ProgressState]:
        """Get all currently active sessions."""
        with self._lock:
            return {
                session_id: state 
                for session_id, state in self._sessions.items()
                if state.status in [ProgressStatus.PENDING, ProgressStatus.RUNNING]
            }
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Statistics dictionary or None if session not found
        """
        state = self.get_progress(session_id)
        if not state:
            return None
            
        stats = {
            "session_id": state.session_id,
            "total_questions": state.total_questions,
            "processed_questions": state.processed_questions,
            "progress_percentage": state.progress_percentage,
            "current_step": state.current_step,
            "status": state.status.value,
            "elapsed_time": state.elapsed_time,
            "estimated_time_remaining": state.estimated_time_remaining,
            "questions_per_minute": 0.0
        }
        
        # Calculate questions per minute
        if state.elapsed_time and state.elapsed_time > 0:
            stats["questions_per_minute"] = (state.processed_questions / state.elapsed_time) * 60
            
        return stats


# Global progress tracker instance
_progress_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker


def start_progress_session(
    session_id: str, 
    total_questions: int,
    initial_step: str = "Initialisation"
) -> ProgressState:
    """Convenience function to start a progress session."""
    tracker = get_progress_tracker()
    return tracker.start_session(session_id, total_questions, initial_step)


def update_progress(
    session_id: str, 
    processed_questions: Optional[int] = None,
    current_step: Optional[str] = None,
    increment: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[ProgressState]:
    """Convenience function to update progress."""
    tracker = get_progress_tracker()
    return tracker.update_progress(session_id, processed_questions, current_step, increment, metadata)


def increment_progress(
    session_id: str, 
    current_step: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[ProgressState]:
    """Convenience function to increment progress by 1."""
    tracker = get_progress_tracker()
    return tracker.increment_progress(session_id, current_step, metadata)


def get_progress_state(session_id: str) -> Optional[ProgressState]:
    """Convenience function to get progress state."""
    tracker = get_progress_tracker()
    return tracker.get_progress(session_id)


def complete_progress_session(session_id: str, final_step: str = "Terminé") -> Optional[ProgressState]:
    """Convenience function to complete a session."""
    tracker = get_progress_tracker()
    return tracker.complete_session(session_id, final_step)


def fail_progress_session(
    session_id: str, 
    error_message: str, 
    error_step: str = "Erreur"
) -> Optional[ProgressState]:
    """Convenience function to fail a session."""
    tracker = get_progress_tracker()
    return tracker.fail_session(session_id, error_message, error_step)