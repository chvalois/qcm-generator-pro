"""
QCM Generator Pro - Prompts Module

This module provides multilingual prompt templates for QCM generation,
validation, and theme extraction.
"""

from .templates import (
    PromptTemplateManager,
    get_template_manager,
    get_language_template,
    generate_question_prompt,
    generate_validation_prompt,
    generate_theme_extraction_prompt,
    get_system_prompt
)

__all__ = [
    "PromptTemplateManager",
    "get_template_manager", 
    "get_language_template",
    "generate_question_prompt",
    "generate_validation_prompt",
    "generate_theme_extraction_prompt",
    "get_system_prompt"
]