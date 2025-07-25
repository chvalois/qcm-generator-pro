"""
Language-specific prompt templates for QCM generation.

This module contains implementations of prompt templates for different
languages, all inheriting from the base LanguageTemplate class.
"""

from .base import LanguageTemplate
from .fr import FrenchTemplate
from .en import EnglishTemplate

__all__ = [
    "LanguageTemplate",
    "FrenchTemplate", 
    "EnglishTemplate"
]