"""
Question Prompt Builder Service

Handles the creation of prompts for question generation.
Follows SRP by focusing solely on prompt construction.
"""

import logging
from typing import Dict, Any

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import GenerationConfig, QuestionContext
from src.prompts.templates import generate_question_prompt, get_system_prompt

logger = logging.getLogger(__name__)


class QuestionPromptBuilder:
    """
    Service responsible for building prompts for question generation.
    
    This class encapsulates all logic related to prompt creation,
    making it easy to modify prompt templates without affecting
    other parts of the system.
    """
    
    def build_generation_prompt(
        self, 
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty,
        language: Language = Language.FR
    ) -> str:
        """
        Build a prompt for question generation.
        
        Args:
            context: Question context from RAG
            config: Generation configuration
            question_type: Type of question to generate
            difficulty: Question difficulty
            language: Generation language
            
        Returns:
            Formatted prompt for question generation
        """
        logger.debug(f"Building prompt for {question_type.value} question, difficulty: {difficulty.value}")
        
        return generate_question_prompt(
            language=language,
            context=context,
            config=config,
            question_type=question_type,
            difficulty=difficulty
        )
    
    def build_system_prompt(self, language: Language = Language.FR) -> str:
        """
        Build system prompt for the given language.
        
        Args:
            language: Target language for the system prompt
            
        Returns:
            System prompt string
        """
        return get_system_prompt(language)
    
    def get_prompt_metadata(
        self,
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty
    ) -> Dict[str, Any]:
        """
        Get metadata about the prompt for logging/debugging.
        
        Args:
            context: Question context
            config: Generation configuration  
            question_type: Question type
            difficulty: Difficulty level
            
        Returns:
            Dictionary with prompt metadata
        """
        return {
            "context_length": len(context.context_text),
            "themes_count": len(context.themes),
            "confidence_score": context.confidence_score,
            "question_type": question_type.value,
            "difficulty": difficulty.value,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }


# Global instance
_prompt_builder: QuestionPromptBuilder | None = None


def get_question_prompt_builder() -> QuestionPromptBuilder:
    """Get the global prompt builder instance."""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = QuestionPromptBuilder()
    return _prompt_builder