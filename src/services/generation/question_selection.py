"""
Question Selection Service

Handles the selection of question types and difficulty levels based on configuration.
Follows SRP by focusing solely on selection logic.
"""

import logging
import random
from typing import List, Tuple

from src.models.enums import Difficulty, QuestionType
from src.models.schemas import GenerationConfig

logger = logging.getLogger(__name__)


class QuestionSelector:
    """
    Service responsible for selecting question types and difficulty levels.
    
    This class encapsulates the logic for probabilistic selection
    based on configuration parameters.
    """
    
    def select_question_type(self, config: GenerationConfig) -> QuestionType:
        """
        Select question type based on configuration probabilities.
        
        Args:
            config: Generation configuration with type distribution
            
        Returns:
            Selected question type
        """
        rand = random.random()
        cumulative = 0.0
        
        for q_type, probability in config.question_types.items():
            cumulative += probability
            if rand <= cumulative:
                # Handle both enum and string keys
                if isinstance(q_type, QuestionType):
                    logger.debug(f"Selected question type: {q_type.value}")
                    return q_type
                else:
                    # q_type is a string, convert to enum
                    enum_type = QuestionType(q_type)
                    logger.debug(f"Selected question type: {enum_type.value}")
                    return enum_type
                
        # Fallback to unique choice
        logger.debug("Using fallback question type: unique-choice")
        return QuestionType.UNIQUE_CHOICE
    
    def select_difficulty(self, config: GenerationConfig) -> Difficulty:
        """
        Select difficulty based on configuration probabilities.
        
        Args:
            config: Generation configuration with difficulty distribution
            
        Returns:
            Selected difficulty level
        """
        rand = random.random()
        cumulative = 0.0
        
        for difficulty, probability in config.difficulty_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                # Handle both enum and string keys
                if isinstance(difficulty, Difficulty):
                    logger.debug(f"Selected difficulty: {difficulty.value}")
                    return difficulty
                else:
                    # difficulty is a string, convert to enum
                    enum_difficulty = Difficulty(difficulty)
                    logger.debug(f"Selected difficulty: {enum_difficulty.value}")
                    return enum_difficulty
                
        # Fallback to medium
        logger.debug("Using fallback difficulty: medium")
        return Difficulty.MEDIUM
    
    def select_batch_parameters(
        self, 
        config: GenerationConfig, 
        batch_size: int
    ) -> List[Tuple[QuestionType, Difficulty]]:
        """
        Select parameters for a batch of questions.
        
        Args:
            config: Generation configuration
            batch_size: Number of questions in the batch
            
        Returns:
            List of (question_type, difficulty) tuples
        """
        parameters = []
        
        for _ in range(batch_size):
            question_type = self.select_question_type(config)
            difficulty = self.select_difficulty(config)
            parameters.append((question_type, difficulty))
        
        logger.debug(f"Selected parameters for batch of {batch_size} questions")
        return parameters
    
    def validate_distributions(self, config: GenerationConfig) -> bool:
        """
        Validate that probability distributions sum to 1.0.
        
        Args:
            config: Generation configuration to validate
            
        Returns:
            True if distributions are valid, False otherwise
        """
        # Check question type distribution
        type_total = sum(config.question_types.values())
        if abs(type_total - 1.0) > 0.01:  # Allow small floating point errors
            logger.warning(f"Question type distribution sums to {type_total}, expected 1.0")
            return False
        
        # Check difficulty distribution
        diff_total = sum(config.difficulty_distribution.values())
        if abs(diff_total - 1.0) > 0.01:
            logger.warning(f"Difficulty distribution sums to {diff_total}, expected 1.0")
            return False
        
        return True


# Global instance
_question_selector: QuestionSelector | None = None


def get_question_selector() -> QuestionSelector:
    """Get the global question selector instance."""
    global _question_selector
    if _question_selector is None:
        _question_selector = QuestionSelector()
    return _question_selector