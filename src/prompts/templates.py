"""
Main templates module for QCM generation prompts.

This module provides the main interface for accessing language-specific
prompt templates and managing multilingual prompt generation.
"""

from typing import Dict, Type

from ..models.enums import Language
from .languages.base import LanguageTemplate
from .languages.fr import FrenchTemplate
from .languages.en import EnglishTemplate


class PromptTemplateManager:
    """
    Main manager for accessing and using language-specific prompt templates.
    """
    
    def __init__(self):
        """Initialize the template manager with available languages."""
        self._templates: Dict[Language, LanguageTemplate] = {
            Language.FR: FrenchTemplate(),
            Language.EN: EnglishTemplate(),
        }
        
        # Default fallback language
        self._default_language = Language.FR
        
    def get_template(self, language: Language) -> LanguageTemplate:
        """
        Get the template for a specific language.
        
        Args:
            language: The requested language
            
        Returns:
            The language template instance
            
        Raises:
            ValueError: If the language is not supported
        """
        if language not in self._templates:
            # Fallback to default language if requested language is not available
            return self._templates[self._default_language]
        
        return self._templates[language]
    
    def get_available_languages(self) -> list[Language]:
        """
        Get the list of available languages.
        
        Returns:
            List of supported language enums
        """
        return list(self._templates.keys())
    
    def add_template(self, language: Language, template: LanguageTemplate):
        """
        Add or update a language template.
        
        Args:
            language: The language enum
            template: The template instance
        """
        self._templates[language] = template
        
    def set_default_language(self, language: Language):
        """
        Set the default fallback language.
        
        Args:
            language: The default language to use
        """
        if language in self._templates:
            self._default_language = language
        else:
            raise ValueError(f"Language {language} is not available as a template")


# Global template manager instance
_template_manager: PromptTemplateManager | None = None


def get_template_manager() -> PromptTemplateManager:
    """
    Get the global template manager instance.
    
    Returns:
        The global PromptTemplateManager instance
    """
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager


def get_language_template(language: Language) -> LanguageTemplate:
    """
    Convenience function to get a language template.
    
    Args:
        language: The requested language
        
    Returns:
        The language template instance
    """
    manager = get_template_manager()
    return manager.get_template(language)


# Additional convenience functions for common template operations
def generate_question_prompt(
    language: Language,
    context,
    config,
    question_type,
    difficulty
) -> str:
    """
    Generate a question generation prompt for any language.
    
    Args:
        language: Target language
        context: Question context from RAG
        config: Generation configuration
        question_type: Type of question
        difficulty: Question difficulty
        
    Returns:
        Formatted prompt string
    """
    template = get_language_template(language)
    return template.get_question_generation_prompt(
        context, config, question_type, difficulty
    )


def generate_validation_prompt(language: Language, question_data: Dict) -> str:
    """
    Generate a validation prompt for any language.
    
    Args:
        language: Target language
        question_data: Question data to validate
        
    Returns:
        Formatted validation prompt string
    """
    template = get_language_template(language)
    return template.get_validation_prompt(question_data)


def generate_theme_extraction_prompt(language: Language, text_content: str) -> str:
    """
    Generate a theme extraction prompt for any language.
    
    Args:
        language: Target language
        text_content: Text content to analyze
        
    Returns:
        Formatted theme extraction prompt string
    """
    template = get_language_template(language) 
    return template.get_theme_extraction_prompt(text_content)


def get_system_prompt(language: Language) -> str:
    """
    Get system prompt for any language.
    
    Args:
        language: Target language
        
    Returns:
        System prompt string
    """
    template = get_language_template(language)
    return template.get_system_prompt()