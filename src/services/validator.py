"""
QCM Generator Pro - Question Validator Service

This module handles validation of generated QCM questions for quality,
structure, and educational value.
"""

import logging
import re
from typing import Any

from ..models.enums import QuestionType, ValidationStatus
from ..models.schemas import QuestionCreate, ValidationResult

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


class QuestionValidator:
    """
    Service for validating QCM questions.
    """
    
    def __init__(self):
        """Initialize question validator."""
        self.min_question_length = 10
        self.max_question_length = 500
        self.min_option_length = 1
        self.max_option_length = 200
        self.min_explanation_length = 10
        
    def validate_structure(self, question: QuestionCreate) -> ValidationResult:
        """
        Validate question structure and format.
        
        Args:
            question: Question to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        score = 1.0
        
        # Validate question text
        if not question.question_text or not question.question_text.strip():
            errors.append("Question text is empty")
            score -= 0.3
        elif len(question.question_text) < self.min_question_length:
            errors.append(f"Question text too short (minimum {self.min_question_length} characters)")
            score -= 0.2
        elif len(question.question_text) > self.max_question_length:
            warnings.append(f"Question text very long ({len(question.question_text)} characters)")
            score -= 0.1
            
        # Validate options
        if not question.options or len(question.options) < 2:
            errors.append("Question must have at least 2 options")
            score -= 0.4
        else:
            # Check option count based on question type
            if question.question_type == QuestionType.MULTIPLE_CHOICE:
                if len(question.options) < 3:
                    errors.append("Multiple choice questions should have at least 3 options")
                    score -= 0.2
                elif len(question.options) > 5:
                    warnings.append(f"Many options ({len(question.options)}) - consider reducing")
                    score -= 0.05
            elif question.question_type == QuestionType.MULTIPLE_SELECTION:
                if len(question.options) < 4:
                    warnings.append("Multiple selection questions work better with 4+ options")
                    score -= 0.1
                    
            # Validate individual options
            for i, option in enumerate(question.options):
                # Handle both string and QuestionOption objects
                option_text = option.text if hasattr(option, 'text') else str(option)
                
                if not option_text or not option_text.strip():
                    errors.append(f"Option {i+1} is empty")
                    score -= 0.1
                elif len(option_text) < self.min_option_length:
                    errors.append(f"Option {i+1} too short")
                    score -= 0.05
                elif len(option_text) > self.max_option_length:
                    warnings.append(f"Option {i+1} very long ({len(option_text)} characters)")
                    score -= 0.02
                    
            # Check for duplicate options
            option_texts = [option.text if hasattr(option, 'text') else str(option) for option in question.options]
            unique_options = set(option_texts)
            if len(unique_options) < len(option_texts):
                errors.append("Duplicate options found")
                score -= 0.2
                
        # Validate correct answers (check both correct_answers attribute and is_correct flags)
        correct_indices = []
        
        if hasattr(question, 'correct_answers') and question.correct_answers:
            # Use correct_answers if available
            correct_indices = question.correct_answers
        else:
            # Extract from QuestionOption is_correct flags
            correct_indices = [i for i, opt in enumerate(question.options) if hasattr(opt, 'is_correct') and opt.is_correct]
        
        if not correct_indices:
            errors.append("No correct answers specified")
            score -= 0.4
        else:
            # Check indices are valid
            for idx in correct_indices:
                if not isinstance(idx, int) or idx < 0 or idx >= len(question.options):
                    errors.append(f"Invalid correct answer index: {idx}")
                    score -= 0.1
                    
            # Check answer count based on question type
            if question.question_type == QuestionType.MULTIPLE_CHOICE:
                if len(correct_indices) != 1:
                    errors.append("Multiple choice questions must have exactly 1 correct answer")
                    score -= 0.3
            elif question.question_type == QuestionType.MULTIPLE_SELECTION:
                if len(correct_indices) < 2:
                    warnings.append("Multiple selection questions typically have 2+ correct answers")
                    score -= 0.1
                elif len(correct_indices) >= len(question.options):
                    errors.append("Too many correct answers (all or almost all options)")
                    score -= 0.2
                    
        # Validate explanation
        if not question.explanation or not question.explanation.strip():
            warnings.append("No explanation provided")
            score -= 0.1
        elif len(question.explanation) < self.min_explanation_length:
            warnings.append("Explanation is very short")
            score -= 0.05
            
        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            validation_type="structure",
            details={"checks_performed": ["structure", "format", "completeness"]}
        )
        
    def validate_content_quality(self, question: QuestionCreate) -> ValidationResult:
        """
        Validate question content quality.
        
        Args:
            question: Question to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        score = 1.0
        
        # Check question clarity
        question_text = question.question_text.lower()
        
        # Look for unclear language
        unclear_patterns = [
            r'\b(peut-être|possiblement|probablement)\b',  # French uncertain words
            r'\b(maybe|possibly|probably)\b',  # English uncertain words
            r'\?.*\?',  # Multiple question marks
            r'\.\.\.+',  # Multiple dots indicating uncertainty
        ]
        
        for pattern in unclear_patterns:
            if re.search(pattern, question_text):
                warnings.append("Question contains uncertain language")
                score -= 0.1
                break
                
        # Check for complete sentences
        if not question_text.strip().endswith(('?', '.', ':', '!')):
            warnings.append("Question should end with proper punctuation")
            score -= 0.05
            
        # Validate options quality
        if question.options:
            option_texts = [opt.text if hasattr(opt, 'text') else str(opt) for opt in question.options]
            option_lengths = [len(opt.strip()) for opt in option_texts]
            
            # Check for balanced option lengths
            if max(option_lengths) > 2 * min(option_lengths):
                warnings.append("Option lengths vary significantly")
                score -= 0.1
                
            # Check for obvious wrong answers
            obvious_wrong = ['aucune de ces réponses', 'none of the above', 'toutes les réponses']
            for option in option_texts:
                if any(wrong in option.lower() for wrong in obvious_wrong):
                    warnings.append("Avoid 'none/all of the above' options")
                    score -= 0.05
                    break
                    
            # Check for grammatical consistency
            starts_with_capital = [opt[0].isupper() for opt in option_texts if opt]
            if len(set(starts_with_capital)) > 1:
                warnings.append("Inconsistent capitalization in options")
                score -= 0.05
                
        # Validate explanation quality
        if question.explanation:
            explanation_lower = question.explanation.lower()
            
            # Check if explanation actually explains
            if len(explanation_lower.split()) < 5:
                warnings.append("Explanation seems too brief")
                score -= 0.1
                
            # Check if explanation references correct answer
            correct_options = []
            correct_indices = []
            
            if hasattr(question, 'correct_answers') and question.correct_answers:
                correct_indices = question.correct_answers
            else:
                # Extract from QuestionOption is_correct flags
                correct_indices = [i for i, opt in enumerate(question.options) if hasattr(opt, 'is_correct') and opt.is_correct]
            
            for i in correct_indices:
                if i < len(question.options):
                    opt = question.options[i]
                    opt_text = opt.text if hasattr(opt, 'text') else str(opt)
                    correct_options.append(opt_text)
            
            explanation_mentions_answer = any(
                opt.lower()[:10] in explanation_lower for opt in correct_options
            )
            
            if not explanation_mentions_answer:
                warnings.append("Explanation doesn't clearly reference the correct answer")
                score -= 0.1
                
        # Check thematic coherence
        theme = getattr(question, 'theme', None) or (question.generation_params.get('topic') if hasattr(question, 'generation_params') and question.generation_params else None)
        theme_words = theme.lower().split() if theme else []
        
        option_texts = [opt.text if hasattr(opt, 'text') else str(opt) for opt in question.options]
        question_content = (question.question_text + ' ' + ' '.join(option_texts)).lower()
        
        theme_relevance = sum(1 for word in theme_words if word in question_content)
        if theme_words and theme_relevance == 0:
            warnings.append("Question doesn't seem related to specified theme")
            score -= 0.15
            
        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            validation_type="content",
            details={"checks_performed": ["clarity", "completeness", "coherence"]}
        )
        
    def validate_educational_value(self, question: QuestionCreate) -> ValidationResult:
        """
        Validate educational value of the question.
        
        Args:
            question: Question to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        score = 1.0
        
        # Check question type appropriateness
        question_lower = question.question_text.lower()
        
        # Questions that should be multiple choice
        mc_indicators = ['qui', 'quoi', 'où', 'quand', 'combien', 'what', 'who', 'where', 'when', 'how many']
        has_mc_indicator = any(indicator in question_lower for indicator in mc_indicators)
        
        if (question.question_type == QuestionType.MULTIPLE_SELECTION and 
            has_mc_indicator and 
            len(question.correct_answers) == 1):
            warnings.append("This might work better as a multiple choice question")
            score -= 0.1
            
        # Check difficulty appropriateness
        complexity_indicators = {
            'easy': ['définir', 'identifier', 'nommer', 'define', 'identify', 'name'],
            'medium': ['expliquer', 'comparer', 'analyser', 'explain', 'compare', 'analyze'],
            'hard': ['évaluer', 'synthétiser', 'critiquer', 'evaluate', 'synthesize', 'critique']
        }
        
        detected_difficulty = None
        for level, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                detected_difficulty = level
                break
                
        if detected_difficulty:
            difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
            difficulty_value = question.difficulty.value if hasattr(question.difficulty, 'value') else question.difficulty
            question_difficulty_level = difficulty_map.get(difficulty_value, 2)
            detected_level = difficulty_map[detected_difficulty]
            
            if abs(question_difficulty_level - detected_level) > 1:
                warnings.append(f"Question complexity seems {detected_difficulty} but marked as {difficulty_value}")
                score -= 0.1
                
        # Check for educational best practices
        if question.options:
            option_texts = [opt.text if hasattr(opt, 'text') else str(opt) for opt in question.options]
            
            # All options should be plausible
            very_short_options = [opt for opt in option_texts if len(opt.strip()) < 3]
            if len(very_short_options) > 1:
                warnings.append("Multiple very short options may not be plausible distractors")
                score -= 0.1
                
            # Check for options that are obviously wrong
            obvious_patterns = [
                r'\b(impossible|jamais|never|always|toujours)\b',
                r'\b(100%|0%|tous|aucun|all|none)\b'
            ]
            
            obvious_count = 0
            for option in option_texts:
                for pattern in obvious_patterns:
                    if re.search(pattern, option.lower()):
                        obvious_count += 1
                        break
                        
            if obvious_count > 1:
                warnings.append("Multiple options contain absolute terms (may be too obvious)")
                score -= 0.1
                
        # Check explanation educational value
        if question.explanation:
            explanation_lower = question.explanation.lower()
            
            # Good explanations should teach, not just state
            teaching_indicators = [
                'parce que', 'car', 'puisque', 'because', 'since', 'due to',
                'permet', 'permet de', 'allows', 'enables',
                'contrairement', 'unlike', 'contrary to'
            ]
            
            has_teaching_element = any(indicator in explanation_lower for indicator in teaching_indicators)
            if not has_teaching_element:
                warnings.append("Explanation could be more educational (explain why, not just what)")
                score -= 0.1
                
        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            validation_type="educational",
            details={"checks_performed": ["appropriateness", "difficulty", "educational_value"]}
        )
        
    def validate_question(self, question: QuestionCreate) -> ValidationResult:
        """
        Perform complete validation of a question.
        
        Args:
            question: Question to validate
            
        Returns:
            Combined validation result
        """
        logger.debug(f"Validating question: {question.question_text[:50]}...")
        
        # Run all validation checks
        structure_result = self.validate_structure(question)
        content_result = self.validate_content_quality(question)
        educational_result = self.validate_educational_value(question)
        
        # Combine results
        all_errors = structure_result.errors + content_result.errors + educational_result.errors
        all_warnings = structure_result.warnings + content_result.warnings + educational_result.warnings
        
        # Calculate weighted score
        weights = {"structure": 0.4, "content": 0.35, "educational": 0.25}
        combined_score = (
            structure_result.score * weights["structure"] +
            content_result.score * weights["content"] +
            educational_result.score * weights["educational"]
        )
        
        # Determine overall validity
        is_valid = len(all_errors) == 0 and combined_score >= 0.6
        
        # Determine validation status
        if combined_score >= 0.8 and len(all_errors) == 0:
            status = ValidationStatus.APPROVED
        elif combined_score >= 0.6 and len(all_errors) == 0:
            status = ValidationStatus.PENDING
        else:
            status = ValidationStatus.REJECTED
            
        combined_result = ValidationResult(
            is_valid=is_valid,
            score=combined_score,
            errors=all_errors,
            warnings=all_warnings,
            validation_type="complete",
            details={
                "structure_score": structure_result.score,
                "content_score": content_result.score,
                "educational_score": educational_result.score,
                "recommended_status": status.value,
                "validation_components": ["structure", "content", "educational"]
            }
        )
        
        logger.debug(f"Validation completed: score={combined_score:.2f}, status={status.value}")
        return combined_result
        
    def validate_question_batch(self, questions: list[QuestionCreate]) -> list[ValidationResult]:
        """
        Validate a batch of questions.
        
        Args:
            questions: List of questions to validate
            
        Returns:
            List of validation results
        """
        logger.info(f"Validating batch of {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions):
            try:
                result = self.validate_question(question)
                results.append(result)
                logger.debug(f"Question {i+1}: score={result.score:.2f}, valid={result.is_valid}")
            except Exception as e:
                logger.error(f"Validation failed for question {i+1}: {e}")
                error_result = ValidationResult(
                    is_valid=False,
                    score=0.0,
                    errors=[f"Validation error: {str(e)}"],
                    warnings=[],
                    validation_type="error",
                    details={"error": str(e)}
                )
                results.append(error_result)
                
        # Log batch statistics
        valid_count = sum(1 for r in results if r.is_valid)
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        
        logger.info(f"Batch validation completed: {valid_count}/{len(questions)} valid, avg score: {avg_score:.2f}")
        
        return results


# Global validator instance
_validator: QuestionValidator | None = None


def get_question_validator() -> QuestionValidator:
    """Get the global question validator instance."""
    global _validator
    if _validator is None:
        _validator = QuestionValidator()
    return _validator


# Convenience functions
def validate_question(question: QuestionCreate) -> ValidationResult:
    """Validate a single question."""
    validator = get_question_validator()
    return validator.validate_question(question)


def validate_questions_batch(questions: list[QuestionCreate]) -> list[ValidationResult]:
    """Validate a batch of questions."""
    validator = get_question_validator()
    return validator.validate_question_batch(questions)