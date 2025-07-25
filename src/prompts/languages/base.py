"""
Base template class for multilingual QCM generation prompts.

This module defines the abstract base class that all language-specific
prompt templates must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict

from ...models.enums import Difficulty, QuestionType
from ...models.schemas import QuestionContext, GenerationConfig


class LanguageTemplate(ABC):
    """
    Abstract base class for language-specific prompt templates.
    
    Each language implementation must provide all the required prompt templates
    and helper methods for generating QCM questions in that language.
    """
    
    @property
    @abstractmethod
    def language_code(self) -> str:
        """Return the ISO language code (e.g., 'fr', 'en')."""
        pass
    
    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the full language name (e.g., 'Français', 'English')."""
        pass
    
    @abstractmethod
    def get_question_type_descriptions(self) -> Dict[QuestionType, str]:
        """Return descriptions for each question type in this language."""
        pass
    
    @abstractmethod
    def get_difficulty_descriptions(self) -> Dict[Difficulty, str]:
        """Return descriptions for each difficulty level in this language."""
        pass
    
    @abstractmethod
    def get_question_generation_prompt(
        self,
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty
    ) -> str:
        """
        Generate the main prompt for QCM question generation.
        
        Args:
            context: Question context from RAG
            config: Generation configuration
            question_type: Type of question to generate
            difficulty: Question difficulty level
            
        Returns:
            Formatted prompt string for LLM
        """
        pass
    
    @abstractmethod
    def get_validation_prompt(self, question_data: Dict) -> str:
        """
        Generate prompt for question validation.
        
        Args:
            question_data: The question data to validate
            
        Returns:
            Formatted validation prompt string
        """
        pass
    
    @abstractmethod
    def get_theme_extraction_prompt(self, text_content: str) -> str:
        """
        Generate prompt for theme extraction from PDF content.
        
        Args:
            text_content: Text content from PDF document
            
        Returns:
            Formatted theme extraction prompt string
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM in this language.
        
        Returns:
            System prompt string
        """
        pass
    
    def get_options_count_range(self, question_type: QuestionType) -> str:
        """
        Get the expected number of options for a question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            String describing the options count (e.g., "3-4", "4-6")
        """
        if question_type == QuestionType.MULTIPLE_CHOICE:
            return "3-4"
        else:  # MULTIPLE_SELECTION
            return "4-6"
    
    def get_correct_answers_count(self, question_type: QuestionType) -> str:
        """
        Get the expected number of correct answers for a question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            String describing the correct answers count
        """
        if question_type == QuestionType.MULTIPLE_CHOICE:
            return "1"
        else:  # MULTIPLE_SELECTION
            return "2-3"