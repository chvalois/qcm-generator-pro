"""
Progressive Workflow Manager

Handles the progressive generation workflow (1 → 5 → all questions).
Follows SRP by focusing solely on workflow orchestration.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from src.models.enums import Difficulty, QuestionType
from src.models.schemas import GenerationConfig, QuestionCreate

logger = logging.getLogger(__name__)


class ProgressiveWorkflowError(Exception):
    """Exception raised when progressive workflow fails."""
    pass


class ProgressiveWorkflowManager:
    """
    Service responsible for managing the progressive generation workflow.
    
    This class orchestrates the 1 → 5 → all workflow with validation checkpoints.
    """
    
    def __init__(self):
        """Initialize the workflow manager."""
        self.phase_sizes = [1, 5]  # Phase 1: 1 question, Phase 2: 5 questions
    
    async def execute_progressive_workflow(
        self,
        total_questions: int,
        generation_callback: Callable[[int], List[QuestionCreate]],
        validation_callback: Optional[Callable] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the progressive generation workflow with user validation.
        
        For 1 question: Direct generation without validation
        For 2+ questions: Progressive workflow with user confirmation:
        - Phase 1: Generate 1 question → show to user → await confirmation
        - Phase 2: Generate min(5, remaining) questions → show to user → await confirmation  
        - Phase 3: Generate remaining questions (if any)
        
        Args:
            total_questions: Total number of questions to generate
            generation_callback: Function to generate questions (takes batch_size)
            validation_callback: Optional validation function (required for >1 questions)
            session_id: Session ID for tracking
            
        Returns:
            Workflow results with all phases
            
        Raises:
            ProgressiveWorkflowError: If workflow fails
        """
        # Generate session_id if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        logger.info(f"Starting progressive workflow for {total_questions} questions (session: {session_id})")
        
        results = {
            "session_id": session_id,
            "total_requested": total_questions,
            "phases": [],
            "final_questions": [],
            "generation_stats": {},
            "requires_user_validation": total_questions > 1
        }
        
        all_questions = []
        
        try:
            # Handle single question requests - no progressive validation needed
            if total_questions <= 1:
                logger.info(f"Single question: generating directly without progressive validation")
                single_phase_questions = await self._execute_phase(
                    phase_number=1,
                    batch_size=total_questions,
                    generation_callback=generation_callback,
                    validation_callback=None  # Skip validation for single questions
                )
                
                results["phases"].append(single_phase_questions["phase_result"])
                all_questions.extend(single_phase_questions["questions"])
                
            else:
                # Multi-question requests: Always use progressive validation
                logger.info(f"Multi-question request: using progressive validation workflow")
                
                if not validation_callback:
                    raise ProgressiveWorkflowError("validation_callback is required for multi-question requests")
                
                # Phase 1: Generate 1 test question for user validation
                logger.info("Phase 1: Generating 1 test question for user validation")
                phase1_questions = await self._execute_phase(
                    phase_number=1,
                    batch_size=1,
                    generation_callback=generation_callback,
                    validation_callback=validation_callback
                )
                
                results["phases"].append(phase1_questions["phase_result"])
                
                if not phase1_questions["approved"]:
                    logger.warning("Phase 1 validation failed - stopping workflow")
                    results["status"] = "stopped_at_phase_1"
                    results["final_questions"] = all_questions
                    return results
                    
                all_questions.extend(phase1_questions["questions"])
                remaining_questions = total_questions - 1
                
                # Phase 2: Generate small batch (up to 5 questions) for validation
                if remaining_questions > 0:
                    phase2_batch_size = min(5, remaining_questions)
                    logger.info(f"Phase 2: Generating {phase2_batch_size} questions for user validation")
                    
                    phase2_questions = await self._execute_phase(
                        phase_number=2,
                        batch_size=phase2_batch_size,
                        generation_callback=generation_callback,
                        validation_callback=validation_callback
                    )
                    
                    results["phases"].append(phase2_questions["phase_result"])
                    
                    if not phase2_questions["approved"]:
                        logger.warning("Phase 2 validation failed - stopping workflow")
                        results["status"] = "stopped_at_phase_2"
                        results["final_questions"] = all_questions
                        return results
                    
                    all_questions.extend(phase2_questions["questions"])
                    remaining_questions -= phase2_batch_size
                
                # Phase 3: Generate remaining questions (if any) - usually auto-approved
                if remaining_questions > 0:
                    logger.info(f"Phase 3: Generating remaining {remaining_questions} questions")
                    
                    # For large batches, still ask for user confirmation
                    phase3_validation = validation_callback if remaining_questions > 5 else None
                    
                    phase3_questions = await self._execute_phase(
                        phase_number=3,
                        batch_size=remaining_questions,
                        generation_callback=generation_callback,
                        validation_callback=phase3_validation
                    )
                    
                    results["phases"].append(phase3_questions["phase_result"])
                    
                    if phase3_validation and not phase3_questions["approved"]:
                        logger.warning("Phase 3 validation failed - stopping workflow")
                        results["status"] = "stopped_at_phase_3"
                        results["final_questions"] = all_questions
                        return results
                    
                    all_questions.extend(phase3_questions["questions"])
            
            # Finalize results
            results["final_questions"] = all_questions
            results["generation_stats"] = self._calculate_stats(all_questions, total_questions)
            results["status"] = "completed"
            
            logger.info(f"Progressive workflow completed: {len(all_questions)}/{total_questions} questions generated")
            return results
            
        except Exception as e:
            logger.error(f"Progressive workflow failed: {e}")
            raise ProgressiveWorkflowError(f"Workflow execution failed: {e}")
    
    async def _execute_phase(
        self,
        phase_number: int,
        batch_size: int,
        generation_callback: Callable[[int], List[QuestionCreate]],
        validation_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a single phase of the workflow.
        
        Args:
            phase_number: Phase number (1, 2, or 3)
            batch_size: Number of questions to generate
            generation_callback: Function to generate questions
            validation_callback: Optional validation function
            
        Returns:
            Phase execution results
        """
        logger.info(f"Phase {phase_number}: Generating {batch_size} questions")
        
        try:
            # Generate questions for this phase
            questions = await generation_callback(batch_size)
            
            phase_result = {
                "phase": phase_number,
                "batch_size": batch_size,
                "questions_generated": len(questions),
                "questions": questions,
                "validation_required": validation_callback is not None
            }
            
            # Validation step
            approved = True
            if validation_callback:
                validation_result = await validation_callback(questions, phase_number)
                phase_result["validation_result"] = validation_result
                approved = validation_result.get("approved", False)
            
            return {
                "phase_result": phase_result,
                "questions": questions,
                "approved": approved
            }
            
        except Exception as e:
            logger.error(f"Phase {phase_number} failed: {e}")
            raise ProgressiveWorkflowError(f"Phase {phase_number} execution failed: {e}")
    
    def _calculate_stats(self, questions: List[QuestionCreate], total_requested: int) -> Dict[str, Any]:
        """
        Calculate generation statistics.
        
        Args:
            questions: Generated questions
            total_requested: Total number of questions requested
            
        Returns:
            Statistics dictionary
        """
        if not questions:
            return {
                "total_generated": 0,
                "success_rate": 0.0,
                "topics_covered": 0,
                "difficulty_distribution": {},
                "type_distribution": {}
            }
        
        # Extract themes/topics
        question_themes = []
        for q in questions:
            if hasattr(q, 'generation_params') and q.generation_params and 'topic' in q.generation_params:
                question_themes.append(q.generation_params['topic'])
            elif hasattr(q, 'theme'):
                question_themes.append(q.theme)
            else:
                question_themes.append("Unknown")
        
        # Calculate difficulty distribution
        difficulty_dist = {}
        for difficulty in Difficulty:
            count = len([q for q in questions if q.difficulty == difficulty])
            if count > 0:
                difficulty_dist[difficulty.value] = count
        
        # Calculate type distribution
        type_dist = {}
        for qtype in QuestionType:
            count = len([q for q in questions if q.question_type == qtype])
            if count > 0:
                type_dist[qtype.value] = count
        
        return {
            "total_generated": len(questions),
            "success_rate": len(questions) / total_requested if total_requested > 0 else 0,
            "topics_covered": len(set(question_themes)),
            "difficulty_distribution": difficulty_dist,
            "type_distribution": type_dist
        }
    
    def get_phase_info(self, phase_number: int) -> Dict[str, Any]:
        """
        Get information about a specific phase.
        
        Args:
            phase_number: Phase number (1, 2, or 3)
            
        Returns:
            Phase information
        """
        if phase_number == 1:
            return {
                "phase": 1,
                "description": "Test question generation",
                "batch_size": 1,
                "validation_required": True,
                "purpose": "Validate generation quality before proceeding"
            }
        elif phase_number == 2:
            return {
                "phase": 2,
                "description": "Small batch generation",
                "batch_size": 5,
                "validation_required": True,
                "purpose": "Confirm consistency across multiple questions"
            }
        elif phase_number == 3:
            return {
                "phase": 3,
                "description": "Full batch generation",
                "batch_size": "remaining",
                "validation_required": False,
                "purpose": "Generate all remaining questions automatically"
            }
        else:
            raise ValueError(f"Invalid phase number: {phase_number}")


# Global instance
_workflow_manager: ProgressiveWorkflowManager | None = None


def get_progressive_workflow_manager() -> ProgressiveWorkflowManager:
    """Get the global workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = ProgressiveWorkflowManager()
    return _workflow_manager