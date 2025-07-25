"""
English language template for QCM generation prompts.

This module implements English-specific prompts for generating
QCM questions, validating them, and extracting themes.
"""

from typing import Dict

from ...models.enums import Difficulty, QuestionType
from ...models.schemas import QuestionContext, GenerationConfig
from .base import LanguageTemplate


class EnglishTemplate(LanguageTemplate):
    """English language implementation of QCM generation prompts."""
    
    @property
    def language_code(self) -> str:
        return "en"
    
    @property
    def language_name(self) -> str:
        return "English"
    
    def get_question_type_descriptions(self) -> Dict[QuestionType, str]:
        return {
            QuestionType.MULTIPLE_CHOICE: "multiple choice (single correct answer)",
            QuestionType.MULTIPLE_SELECTION: "multiple selection (multiple correct answers possible)"
        }
    
    def get_difficulty_descriptions(self) -> Dict[Difficulty, str]:
        return {
            Difficulty.EASY: "easy (basic concepts)",
            Difficulty.MEDIUM: "medium (concept application)",
            Difficulty.HARD: "hard (analysis and synthesis)"
        }
    
    def get_question_generation_prompt(
        self,
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty
    ) -> str:
        """Generate the main English prompt for QCM question generation."""
        type_descriptions = self.get_question_type_descriptions()
        difficulty_descriptions = self.get_difficulty_descriptions()
        
        # Safely extract enum values
        question_type_value = question_type.value if hasattr(question_type, 'value') else question_type
        difficulty_value = difficulty.value if hasattr(difficulty, 'value') else difficulty
        
        options_count = self.get_options_count_range(question_type)
        correct_count = self.get_correct_answers_count(question_type)
        
        prompt = f"""Generate a QCM question based on the following context.

CONTEXT:
{context.context_text}

THEME: {context.topic}

REQUIREMENTS:
- Type: {type_descriptions[question_type]}
- Difficulty: {difficulty_descriptions[difficulty]}
- Language: English
- {options_count} answer options
- {correct_count} correct answer(s)
- Clear and precise question
- Plausible answer options
- Detailed explanation

RESPOND ONLY in the following JSON format:
{{
  "question_text": "Question text",
  "question_type": "{question_type_value}",
  "difficulty": "{difficulty_value}",
  "language": "en",
  "theme": "{context.topic}",
  "options": [
    "Option 1",
    "Option 2", 
    "Option 3",
    "Option 4"
  ],
  "correct_answers": [0, 2],
  "explanation": "Detailed explanation of why these answers are correct"
}}"""

        return prompt
    
    def get_validation_prompt(self, question_data: Dict) -> str:
        """Generate English prompt for question validation."""
        return f"""Validate the quality of this QCM question in English.

QUESTION TO VALIDATE:
{question_data}

VALIDATION CRITERIA:
1. STRUCTURE: Correct JSON format, all required fields present
2. CONTENT: Clear question, unambiguous, appropriate level
3. OPTIONS: Plausible distractors, valid correct answers
4. LANGUAGE: Correct English, appropriate terminology
5. PEDAGOGY: Useful question for learning
6. CLARITY AND PRECISION: Avoid ambiguous questions or double negatives, Use precise and technical vocabulary appropriate to the target level
7. BALANCED LENGTH: Maintain similar length options to avoid clues, Concise but complete questions
8. BALANCED COVERAGE: Distribute questions across the entire document, Vary the types of knowledge tested
9. VARIED DIFFICULTY: Mix difficulty levels, Include questions on key concepts vs specific details
10. COHERENT DISTRACTORS: Use terms from the same technical domain, Avoid obviously absurd or out-of-context answers
11. PEDAGOGICAL TRAPS: Include common confusions or related concepts, Use elements mentioned in the document but in another context
12. STATISTICS AND FIGURES: Maximum 5% of questions containing inquiries about figures or stats

RESPOND in JSON format:
{{
  "is_valid": true/false,
  "score": 0-10,
  "issues": ["list of identified problems"],
  "suggestions": ["improvement suggestions"],
  "validation_details": {{
    "structure": true/false,
    "content": true/false,
    "options": true/false,
    "language": true/false,
    "pedagogy": true/false
  }}
}}"""
    
    def get_theme_extraction_prompt(self, text_content: str) -> str:
        """Generate English prompt for theme extraction from PDF content."""
        return f"""Analyze the following content and extract the main themes.

CONTENT TO ANALYZE:
{text_content[:2000]}...

INSTRUCTIONS:
- Identify 3-8 main themes from the document
- For each theme, provide representative keywords
- Estimate a confidence score (0.0-1.0)
- Organize by order of importance

RESPOND in JSON format:
{{
  "themes": [
    {{
      "name": "Theme name",
      "keywords": ["keyword1", "keyword2", "keyword3"],
      "confidence": 0.95,
      "description": "Short theme description"
    }}
  ],
  "document_summary": "General document summary",
  "language_detected": "en",
  "extraction_confidence": 0.90
}}"""
    
    def get_system_prompt(self) -> str:
        """Get the English system prompt for the LLM."""
        return """You are an expert in creating educational QCM questions in English.

YOUR SKILLS:
- Generate clear and pedagogical questions
- Create plausible but incorrect distractors
- Adapt difficulty level according to instructions
- Strictly follow the requested JSON format
- Perfect mastery of academic English

GENERAL INSTRUCTIONS:
- Always generate responses in valid JSON format
- Ensure questions are educational and useful
- Avoid unnecessary traps or ambiguities
- Use vocabulary appropriate to the requested level
- Provide complete pedagogical explanations"""