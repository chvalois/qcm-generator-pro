"""
Question Prompt Builder Service

Handles the creation of prompts for question generation.
Follows SRP by focusing solely on prompt construction.
"""

import logging
from typing import Dict, Any, List, Optional

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import GenerationConfig, QuestionContext
from src.prompts.templates import generate_question_prompt, get_system_prompt
from .simple_examples_loader import get_examples_loader

logger = logging.getLogger(__name__)


class QuestionPromptBuilder:
    """
    Service responsible for building prompts for question generation.
    
    This class encapsulates all logic related to prompt creation,
    making it easy to modify prompt templates without affecting
    other parts of the system.
    """
    
    def __init__(self):
        """Initialize the prompt builder with examples loader."""
        self.examples_loader = get_examples_loader()
    
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

    
    def build_generation_prompt_with_examples(
        self, 
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty,
        language: Language = Language.FR,
        examples_file: Optional[str] = None,
        max_examples: int = 3
    ) -> str:
        """
        Build a prompt for question generation with few-shot examples.
        
        Args:
            context: Question context from RAG
            config: Generation configuration
            question_type: Type of question to generate
            difficulty: Question difficulty
            language: Generation language
            examples_file: Name of JSON file with examples (e.g., "python_advanced.json")
            max_examples: Maximum number of examples to include
            
        Returns:
            Formatted prompt with examples for question generation
        """
        logger.debug(f"Building prompt with examples for {question_type.value} question")
        
        # Get base prompt
        base_prompt = generate_question_prompt(
            language=language,
            context=context,
            config=config,
            question_type=question_type,
            difficulty=difficulty
        )
        
        # If no examples file specified, return base prompt
        if not examples_file:
            return base_prompt
            
        # Load examples
        examples = self.examples_loader.get_examples_for_context(
            project_file=examples_file,
            question_type=question_type.value if hasattr(question_type, 'value') else str(question_type),
            difficulty=difficulty.value if hasattr(difficulty, 'value') else str(difficulty),
            max_examples=max_examples
        )
        
        # If no examples found, return base prompt
        if not examples:
            logger.debug(f"No examples found for {question_type}/{difficulty}, using base prompt")
            return base_prompt
            
        # Build examples section
        examples_section = self._build_examples_section(examples, language)
        
        # Combine examples with base prompt
        enhanced_prompt = f"""Voici des exemples de questions de qualité pour vous guider :

{examples_section}

---

{base_prompt}

IMPORTANT: Inspirez-vous des exemples ci-dessus pour:
- Le style de formulation des questions
- La structure des options de réponse  
- La qualité des distracteurs
- Le niveau de détail des explications
- L'orientation pratique et hands-on quand approprié
- La cohérence entre les réponses correctes et l'explication"""
        
        logger.debug(f"Enhanced prompt with {len(examples)} examples")
        return enhanced_prompt
    
    def _build_examples_section(self, examples: List[Dict], language: Language) -> str:
        """
        Build the examples section for the prompt.
        
        Args:
            examples: List of example dictionaries
            language: Target language
            
        Returns:
            Formatted examples section
        """
        if language == Language.FR:
            section = "EXEMPLES DE QUESTIONS DE RÉFÉRENCE:\n\n"
        else:
            section = "REFERENCE QUESTION EXAMPLES:\n\n"
            
        for i, example in enumerate(examples, 1):
            theme = example.get('theme', 'N/A')
            difficulty = example.get('difficulty', 'medium')
            question = example.get('question', '')
            options = example.get('options', [])
            correct = example.get('correct', [])
            explanation = example.get('explanation', '')
            
            if language == Language.FR:
                section += f"EXEMPLE {i} - {difficulty.upper()} - {theme}:\n"
                section += f"Question: {question}\n"
                section += "Options:\n"
                for option in options:
                    section += f"  {option}\n"
                section += f"Réponse(s) correcte(s): {[chr(65+idx) for idx in correct]}\n"
                section += f"Explication: {explanation}\n\n"
            else:
                section += f"EXAMPLE {i} - {difficulty.upper()} - {theme}:\n"
                section += f"Question: {question}\n"
                section += "Options:\n"
                for option in options:
                    section += f"  {option}\n"
                section += f"Correct answer(s): {[chr(65+idx) for idx in correct]}\n"
                section += f"Explanation: {explanation}\n\n"
                
        return section
    
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