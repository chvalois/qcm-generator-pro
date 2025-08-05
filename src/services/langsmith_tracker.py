"""
LangSmith Tracker Service

Handles LangSmith tracing and monitoring for LLM calls in QCM Generator Pro.
Provides comprehensive tracking of prompts, responses, and metadata.
"""

import logging
import os
from typing import Any, Dict, Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

# LangSmith imports with fallback
try:
    from langsmith import Client
    from langsmith.run_helpers import traceable
    from langchain.callbacks import LangChainTracer
    from langchain.callbacks.manager import CallbackManager
    LANGSMITH_AVAILABLE = True
except ImportError:
    logger.warning("LangSmith not available. Install with: pip install langsmith")
    LANGSMITH_AVAILABLE = False
    
    # Fallback decorators/classes
    def traceable(name: str = None, **kwargs):
        """Fallback decorator when LangSmith is not available."""
        def decorator(func):
            return func
        return decorator
    
    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def create_run(self, *args, **kwargs):
            return None
        def update_run(self, *args, **kwargs):
            pass
    
    class LangChainTracer:
        def __init__(self, *args, **kwargs):
            pass
    
    class CallbackManager:
        def __init__(self, *args, **kwargs):
            pass


class LangSmithTracker:
    """
    Service for tracking LLM calls with LangSmith.
    
    Provides comprehensive monitoring of:
    - Question generation prompts and responses
    - Few-shot examples usage
    - Generation performance metrics
    - Error tracking and debugging
    """
    
    def __init__(self):
        """Initialize LangSmith tracker."""
        self.enabled = self._check_langsmith_enabled()
        self.client = None
        self.tracer = None
        
        if self.enabled and LANGSMITH_AVAILABLE:
            try:
                self.client = Client()
                self.tracer = LangChainTracer(
                    project_name=os.getenv("LANGCHAIN_PROJECT", "qcm-generator-pro")
                )
                logger.info("LangSmith tracking enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith client: {e}")
                self.enabled = False
        else:
            logger.info("LangSmith tracking disabled")
    
    def _check_langsmith_enabled(self) -> bool:
        """Check if LangSmith tracking is enabled via environment variables."""
        return (
            LANGSMITH_AVAILABLE and
            os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and
            os.getenv("LANGCHAIN_API_KEY") is not None
        )
    
    @contextmanager
    def trace_qcm_generation(
        self,
        session_id: str,
        topic: str,
        question_type: str,
        difficulty: str,
        examples_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing QCM generation.
        
        Uses the modern LangSmith traceable decorator approach.
        """
        if not self.enabled:
            yield None
            return
        
        # For the traceable decorator approach, we just yield a placeholder
        # The actual tracing is handled by the @traceable decorator on the LLM calls
        yield "traceable_context"
    
    def track_llm_call(
        self,
        run_id: Optional[str],
        prompt: str,
        system_prompt: Optional[str],
        response: str,
        model: str,
        provider: str,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track individual LLM call within a generation run.
        
        Args:
            run_id: Parent run ID from trace_qcm_generation
            prompt: User prompt sent to LLM
            system_prompt: System prompt used
            response: LLM response received
            model: Model name used
            provider: Provider (openai, anthropic, ollama)
            usage: Token usage information
            metadata: Additional metadata
        """
        if not self.enabled or not run_id:
            return
            
        try:
            # Create child run for this LLM call
            self.client.create_run(
                name="llm_call",
                run_type="llm",
                parent_run_id=run_id,
                inputs={
                    "prompt": prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
                    "system_prompt": system_prompt[:500] + "..." if system_prompt and len(system_prompt) > 500 else system_prompt,
                    "model": model,
                    "provider": provider
                },
                outputs={
                    "response": response[:1000] + "..." if len(response) > 1000 else response
                },
                tags=["llm", provider, model],
                extra={
                    "token_usage": usage or {},
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    **(metadata or {})
                },
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error tracking LLM call: {e}")
    
    def track_generation_result(
        self,
        run_id: Optional[str],
        question_text: str,
        options: list,
        correct_answers: list,
        explanation: str,
        validation_passed: bool,
        generation_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track the final generation result.
        
        Args:
            run_id: Parent run ID
            question_text: Generated question text
            options: Question options
            correct_answers: Correct answer indices
            explanation: Question explanation
            validation_passed: Whether validation passed
            generation_time: Time taken to generate
            metadata: Additional metadata
        """
        if not self.enabled or not run_id:
            return
            
        try:
            # Update the main run with results
            self.client.update_run(
                run_id,
                outputs={
                    "question_text": question_text,
                    "options_count": len(options),
                    "correct_answers_count": len(correct_answers),
                    "explanation_length": len(explanation),
                    "validation_passed": validation_passed,
                    "generation_time_seconds": generation_time
                },
                extra={
                    "quality_metrics": {
                        "has_explanation": bool(explanation.strip()),
                        "options_balanced": self._check_options_balance(options),
                        "validation_passed": validation_passed
                    },
                    "performance_metrics": {
                        "generation_time": generation_time,
                        "question_length": len(question_text),
                        "explanation_length": len(explanation)
                    },
                    **(metadata or {})
                }
            )
            
        except Exception as e:
            logger.error(f"Error tracking generation result: {e}")
    
    def _check_options_balance(self, options: list) -> bool:
        """Check if options are reasonably balanced in length."""
        if len(options) < 2:
            return False
            
        lengths = [len(str(opt)) for opt in options]
        avg_length = sum(lengths) / len(lengths)
        
        # Check if all options are within 50% of average length
        return all(abs(length - avg_length) / avg_length < 0.5 for length in lengths)
    
    def get_callback_manager(self) -> Optional[CallbackManager]:
        """Get LangChain callback manager with LangSmith tracer."""
        if not self.enabled or not self.tracer:
            return None
            
        return CallbackManager([self.tracer])
    
    def create_traceable_function(self, name: str, **trace_kwargs):
        """Create a traceable function decorator."""
        if not self.enabled:
            return lambda func: func
            
        return traceable(name=name, **trace_kwargs)


# Global tracker instance
_langsmith_tracker: Optional[LangSmithTracker] = None


def get_langsmith_tracker() -> LangSmithTracker:
    """Get the global LangSmith tracker instance."""
    global _langsmith_tracker
    if _langsmith_tracker is None:
        _langsmith_tracker = LangSmithTracker()
    return _langsmith_tracker


# Convenience decorators
def track_qcm_generation(func):
    """Decorator for tracking QCM generation functions."""
    tracker = get_langsmith_tracker()
    if not tracker.enabled:
        return func
        
    return tracker.create_traceable_function(
        name=f"qcm_{func.__name__}",
        tags=["qcm", "generation"]
    )(func)


def track_llm_interaction(func):
    """Decorator for tracking LLM interaction functions."""
    tracker = get_langsmith_tracker()
    if not tracker.enabled:
        return func
        
    return tracker.create_traceable_function(
        name=f"llm_{func.__name__}",
        tags=["llm", "interaction"]
    )(func)