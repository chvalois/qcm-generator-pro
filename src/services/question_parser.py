"""
Question Parser Service

Handles parsing and validation of LLM responses into structured question data.
Follows SRP by focusing solely on response parsing and data transformation.
"""

import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import GenerationConfig, QuestionCreate, QuestionOption
from src.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class QuestionParsingError(Exception):
    """Exception raised when question parsing fails."""
    pass


class QuestionParser:
    """
    Service responsible for parsing LLM responses into structured question data.
    
    This class handles all the complexity of parsing JSON responses,
    validating structure, and transforming data into the correct format.
    """
    
    # Constants
    MAX_PROMPT_LENGTH = 1000
    REQUIRED_FIELDS = ["question_text", "question_type", "options", "correct_answers", "explanation"]
    
    @staticmethod
    def _clean_quotes(text: str) -> str:
        """
        Remove unwanted quotes and apostrophes from text fields.
        
        Args:
            text: Input text that may contain extra quotes
            
        Returns:
            Cleaned text without extra quotes
        """
        if not isinstance(text, str):
            return text
            
        # Remove leading and trailing quotes of various types
        text = text.strip()
        
        # Remove double quotes at beginning and end
        if text.startswith('"') and text.endswith('"') and len(text) > 2:
            text = text[1:-1]
        
        # Remove single quotes at beginning and end    
        if text.startswith("'") and text.endswith("'") and len(text) > 2:
            text = text[1:-1]
            
        # Remove French quotes
        if text.startswith("«") and text.endswith("»"):
            text = text[1:-1].strip()
            
        # Clean up internal double quotes that might cause CSV issues
        # Replace any remaining double quotes with single quotes for CSV compatibility
        text = text.replace('"', "'")
        
        return text.strip()
    
    @staticmethod
    def _normalize_options(options: list) -> list:
        """
        Normalize options to ensure they start with A., B., C., D. format.
        
        Args:
            options: List of option strings
            
        Returns:
            List of normalized options with A., B., C., D. prefixes
        """
        if not isinstance(options, list):
            return options
            
        normalized = []
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # Support up to 8 options
        
        for i, option in enumerate(options):
            if not isinstance(option, str):
                normalized.append(option)
                continue
                
            option = option.strip()
            
            # Check if option already starts with correct format (A., B., etc.)
            expected_prefix = f"{letters[i]}."
            if option.startswith(expected_prefix):
                # Already correctly formatted
                normalized.append(option)
            elif len(option) >= 2 and option[0].upper() in letters and option[1] == '.':
                # Has some letter format, but wrong letter - replace with correct one
                normalized.append(f"{expected_prefix} {option[2:].lstrip()}")
            else:
                # No letter format - add the correct prefix
                normalized.append(f"{expected_prefix} {option}")
        
        return normalized
    
    def parse_llm_response(
        self,
        response: Any,
        config: GenerationConfig,
        document_id: int,
        session_id: str,
        topic: str,
        prompt: str,
        source_chunks: Optional[List[Any]] = None,
        context_confidence: float = 0.0
    ) -> QuestionCreate:
        """
        Parse LLM response into a QuestionCreate object.
        
        Args:
            response: LLM response object
            config: Generation configuration
            document_id: Document ID for the question
            session_id: Session ID for the question
            topic: Question topic
            prompt: Generation prompt used
            source_chunks: Source chunks from RAG
            context_confidence: Confidence score from context
            
        Returns:
            QuestionCreate object
            
        Raises:
            QuestionParsingError: If parsing fails
        """
        try:
            # Extract JSON from response
            question_data = self._extract_json_from_response(response)
            
            # Validate required fields
            self._validate_required_fields(question_data)
            
            # Parse and validate options (may update question type)
            options, updated_question_type = self._parse_options(question_data)
            
            # Get enum values safely
            language = self._parse_language(question_data, config)
            difficulty = self._parse_difficulty(question_data)
            question_type = self._parse_question_type(question_data)  # Uses updated question_data
            
            # Process source chunks
            processed_chunks = self._process_source_chunks(source_chunks)
            
            # Create QuestionCreate object
            return self._create_question_object(
                question_data=question_data,
                options=options,
                language=language,
                difficulty=difficulty,
                question_type=question_type,
                document_id=document_id,
                session_id=session_id,
                topic=topic,
                prompt=prompt,
                source_chunks=processed_chunks,
                context_confidence=context_confidence,
                response=response,
                config=config
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise QuestionParsingError(f"Question parsing failed: {e}")
    
    def _extract_json_from_response(self, response: Any) -> Dict[str, Any]:
        """Extract JSON data from LLM response with robust error handling."""
        try:
            content = response.content.strip()
            logger.debug(f"Parsing LLM response content (length: {len(content)})")
            
            # Try direct JSON parsing first
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.debug(f"Direct JSON parsing failed: {e}")
                
            # Try to extract JSON from response content
            content = response.content.strip()
            
            # Remove markdown code fences if present
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Find the first complete JSON object
            start = content.find('{')
            if start == -1:
                raise QuestionParsingError(f"No JSON object found in response: {content[:200]}...")
            
            # Find matching closing brace
            brace_count = 0
            end = start
            
            for i, char in enumerate(content[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if brace_count != 0:
                logger.warning(f"JSON appears incomplete, attempting repair: brace_count={brace_count}")
                # Try to repair the JSON
                json_part = content[start:]
                repaired_json = self._attempt_json_repair(json_part)
                
                if repaired_json:
                    try:
                        result = json.loads(repaired_json)
                        logger.info("Successfully repaired and parsed JSON")
                        return result
                    except json.JSONDecodeError as repair_error:
                        logger.error(f"JSON repair failed: {repair_error}")
                
                # If repair fails, provide more detailed error information
                raise QuestionParsingError(
                    f"Invalid JSON response from LLM: {str(e)}. "
                    f"Content preview: {content[:500]}..."
                )
            
            try:
                json_str = content[start:end]
                result = json.loads(json_str)
                logger.debug("Successfully parsed JSON from response")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed even after extraction: {e}")
                # Try one more repair attempt
                repaired_json = self._attempt_json_repair(json_str)
                if repaired_json:
                    try:
                        return json.loads(repaired_json)
                    except:
                        pass
                
                raise QuestionParsingError(
                    f"Invalid JSON response from LLM: {str(e)}. "
                    f"Extracted JSON: {json_str[:200]}..."
                )
                
        except QuestionParsingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during JSON extraction: {e}")
            raise QuestionParsingError(f"Unexpected error parsing LLM response: {e}")
    
    def _attempt_json_repair(self, json_str: str) -> Optional[str]:
        """Attempt to repair common JSON truncation issues."""
        try:
            # Remove any markdown code fences first
            json_str = json_str.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.startswith('```'):
                json_str = json_str[3:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            # First, try to parse as-is in case it's already valid
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
            
            # Find the outermost JSON object boundaries
            start = json_str.find('{')
            if start == -1:
                return None
                
            # Count braces to find the end
            brace_count = 0
            end = len(json_str)
            
            for i in range(start, len(json_str)):
                if json_str[i] == '{':
                    brace_count += 1
                elif json_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            # Extract the JSON part
            json_part = json_str[start:end]
            
            # If braces are unbalanced, try to fix
            if brace_count > 0:
                # Add missing closing braces
                json_part += '}' * brace_count
            
            # Try parsing the repaired JSON
            try:
                json.loads(json_part)
                return json_part
            except json.JSONDecodeError as e:
                logger.debug(f"JSON repair attempt failed: {e}")
                # If it still fails, try to truncate at the last valid comma or closing brace
                last_comma = json_part.rfind(',')
                last_brace = json_part.rfind('}')
                
                if last_brace > last_comma and last_brace > 0:
                    # Try truncating after the last complete field
                    truncated = json_part[:last_brace + 1]
                    try:
                        json.loads(truncated)
                        return truncated
                    except:
                        pass
                
                return None
                
        except Exception as e:
            logger.debug(f"JSON repair failed with exception: {e}")
            return None
    
    def _validate_required_fields(self, question_data: Dict[str, Any]) -> None:
        """Validate that all required fields are present and valid."""
        missing_fields = [
            field for field in self.REQUIRED_FIELDS 
            if field not in question_data
        ]
        
        if missing_fields:
            raise QuestionParsingError(f"Missing required fields: {missing_fields}")
        
        # Clean text fields before validation
        question_data["question_text"] = self._clean_quotes(question_data.get("question_text", ""))
        question_data["explanation"] = self._clean_quotes(question_data.get("explanation", ""))
        # Clean options text (normalization is done later in _parse_options)
        if isinstance(question_data.get("options"), list):
            question_data["options"] = [self._clean_quotes(opt) for opt in question_data["options"]]
        
        # Additional validation for field types and contents
        if not isinstance(question_data.get("question_text", ""), str) or not question_data["question_text"].strip():
            raise QuestionParsingError("question_text must be a non-empty string")
            
        if not isinstance(question_data.get("options", []), list) or len(question_data["options"]) < 2:
            raise QuestionParsingError("options must be a list with at least 2 items")
            
        if not isinstance(question_data.get("correct_answers", []), list) or len(question_data["correct_answers"]) == 0:
            raise QuestionParsingError("correct_answers must be a non-empty list")
            
        if not isinstance(question_data.get("explanation", ""), str) or not question_data["explanation"].strip():
            raise QuestionParsingError("explanation must be a non-empty string")
            
        # Validate correct_answers indices
        correct_indices = question_data["correct_answers"]
        max_index = len(question_data["options"]) - 1
        
        for idx in correct_indices:
            if not isinstance(idx, int) or idx < 0 or idx > max_index:
                raise QuestionParsingError(f"Invalid correct_answer index {idx}, must be between 0 and {max_index}")
        
        logger.debug("All required fields validated successfully")
    
    def _parse_options(self, question_data: Dict[str, Any]) -> tuple[List[QuestionOption], str]:
        """
        Parse options from question data and potentially adjust question type.
        
        Returns:
            Tuple of (options_list, potentially_updated_question_type)
        """
        option_texts = question_data["options"]
        correct_indices = question_data["correct_answers"]
        
        # Validate and adjust correct answers based on question type
        question_type = question_data.get("question_type", "multiple-choice")
        correct_indices, updated_question_type = self._validate_correct_answers(correct_indices, question_type, len(option_texts))
        
        # Update question_data with the potentially new question type
        question_data["question_type"] = updated_question_type
        
        # Normalize option texts before creating QuestionOption objects
        normalized_option_texts = self._normalize_options(option_texts)
        
        # Create QuestionOption objects
        options = []
        for i, option_text in enumerate(normalized_option_texts):
            is_correct = i in correct_indices
            options.append(QuestionOption(
                text=option_text,
                is_correct=is_correct
            ))
        
        return options, updated_question_type
    
    def _validate_correct_answers(
        self, 
        correct_indices: List[int], 
        question_type: str, 
        options_count: int
    ) -> tuple[List[int], str]:
        """
        Validate and intelligently adjust question type based on correct answers.
        
        Returns:
            Tuple of (corrected_indices, potentially_updated_question_type)
        """
        original_type = question_type
        
        if question_type in ["multiple-choice", "unique-choice"]:
            if len(correct_indices) > 1:
                # LLM found multiple correct answers - convert to multiple-selection
                logger.info(f"Converting {question_type} to multiple-selection: found {len(correct_indices)} correct answers")
                return correct_indices, "multiple-selection"
            elif len(correct_indices) == 0:
                # No correct answers - default to first option
                logger.warning("No correct answers found, defaulting to first option")
                return [0], "unique-choice"  # Normalize to new format
        
        elif question_type == "multiple-selection":
            if len(correct_indices) == 1:
                # LLM found only one correct answer - convert to unique-choice
                logger.info(f"Converting multiple-selection to unique-choice: only {len(correct_indices)} correct answer")
                return correct_indices, "unique-choice"
            elif len(correct_indices) == 0:
                # No correct answers - create a reasonable multiple-selection
                logger.warning("No correct answers found for multiple-selection, defaulting to first two options")
                return [0, 1], question_type
            elif len(correct_indices) >= options_count:
                # All options are correct - not meaningful for multiple-selection
                logger.warning(f"All {len(correct_indices)} options marked correct, reducing to first two")
                return [0, 1], question_type
        
        # Default case - return as-is
        if original_type != question_type:
            logger.info(f"Question type adapted from {original_type} to {question_type}")
        
        return correct_indices, question_type
    
    def _parse_language(self, question_data: Dict[str, Any], config: GenerationConfig) -> Language:
        """Parse language from question data or config."""
        language_value = question_data.get("language", config.language)
        
        # Handle different types of language values
        if isinstance(language_value, Language):
            return language_value
        elif hasattr(language_value, 'value'):
            # It's an enum, get its value
            return Language(language_value.value)
        else:
            # It's already a string
            return Language(language_value)
    
    def _parse_difficulty(self, question_data: Dict[str, Any]) -> Difficulty:
        """Parse difficulty from question data."""
        difficulty_raw = question_data.get("difficulty", "medium")
        
        # Handle different types of difficulty values
        if isinstance(difficulty_raw, Difficulty):
            return difficulty_raw
        elif hasattr(difficulty_raw, 'value'):
            # It's an enum, get its value
            difficulty_raw = difficulty_raw.value
        
        # Map French terms to English enum values
        difficulty_mapping = {
            "facile": "easy",
            "moyen": "medium", 
            "difficile": "hard",
            "easy": "easy",
            "medium": "medium",
            "hard": "hard"
        }
        
        difficulty_value = difficulty_mapping.get(difficulty_raw, "medium")
        return Difficulty(difficulty_value)
    
    def _parse_question_type(self, question_data: Dict[str, Any]) -> QuestionType:
        """Parse question type from question data."""
        question_type_value = question_data["question_type"]
        
        # Handle different types of question type values
        if isinstance(question_type_value, QuestionType):
            return question_type_value
        elif hasattr(question_type_value, 'value'):
            # It's an enum, get its value
            return QuestionType(question_type_value.value)
        else:
            # It's a string
            return QuestionType(question_type_value)
    
    def _update_explanation_indices(self, explanation: str) -> str:
        """
        Update explanation text to use 1-based indexing for better user experience.
        
        Args:
            explanation: Raw explanation text with 0-based indices
            
        Returns:
            Updated explanation with 1-based indices
        """
        if not explanation:
            return explanation
        
        import re
        
        # Pattern to match common index references
        patterns = [
            # More comprehensive patterns for all answer reference formats
            # "La réponse 0", "La réponse 1", "L'option 0", "L'affirmation 0", "La proposition 0" etc.
            (r'\b(La?\s+)?(réponse|option|answer|choix|affirmation|proposition)\s+(\d+)\b', 
             lambda m: f"{m.group(1) or ''}{m.group(2)} {int(m.group(3)) + 1}"),
            
            # "réponses 0, 2 et 3", "affirmations 0 et 2", "propositions 0 et 2", "answers 0, 2 and 3" - Convert all numbers to 1-based
            (r'\b(Les?\s+)?(réponses?|answers?|options?|choix|affirmations?|propositions?)\s+([0-9,\s]+(?:et|and)\s+[0-9]+)', 
             lambda m: f"{m.group(1) or ''}{self._convert_number_list(m.group(0))[len(m.group(1) or ''):]}"),
            
            # "(0)", "(1)", etc. when referring to options - Convert to 1-based
            (r'\((\d+)\)', lambda m: f"({int(m.group(1)) + 1})"),
            
            # Handle complex patterns like "Les réponses 0 et 2", "Les affirmations 0 et 2", "Les propositions 0 et 2"
            (r'\b(Les?\s+)(réponses?|options?|answers?|choix|affirmations?|propositions?)\s+(\d+)(\s+et\s+)(\d+)\b',
             lambda m: f"{m.group(1)}{m.group(2)} {int(m.group(3)) + 1}{m.group(4)}{int(m.group(5)) + 1}"),
            
            # Handle even more complex patterns like "Les affirmations 0, 2 et 3", "Les propositions 0, 2 et 3"
            (r'\b(Les?\s+)(affirmations?|réponses?|options?|answers?|choix|propositions?)\s+((?:\d+,?\s*)+(?:et\s+\d+)?)\b',
             lambda m: f"{m.group(1)}{m.group(2)} {self._convert_number_sequence(m.group(3))}"),
        ]
        
        updated_explanation = explanation
        for pattern, replacement in patterns:
            updated_explanation = re.sub(pattern, replacement, updated_explanation, flags=re.IGNORECASE)
        
        return updated_explanation
    
    def _convert_number_list(self, text: str) -> str:
        """Convert a list of 0-based numbers to 1-based in text like 'réponses 0, 2 et 3'."""
        import re
        
        def replace_number(match):
            return str(int(match.group(0)) + 1)
        
        # Replace all individual numbers in the text
        return re.sub(r'\b\d+\b', replace_number, text)
    
    def _convert_number_sequence(self, sequence: str) -> str:
        """Convert a sequence of 0-based numbers to 1-based like '0, 2 et 3' -> '1, 3 et 4'."""
        import re
        
        def replace_number(match):
            return str(int(match.group(0)) + 1)
        
        # Replace all individual numbers in the sequence
        return re.sub(r'\b\d+\b', replace_number, sequence)

    def _process_source_chunks(self, source_chunks: Optional[List[Any]]) -> Optional[List[str]]:
        """Process source chunks into string format."""
        if not source_chunks:
            return None
        
        processed_chunks = []
        
        for i, chunk in enumerate(source_chunks[:5]):  # Limit to 5 chunks
            try:
                if isinstance(chunk, dict):
                    chunk_str = (
                        f"chunk_id: {chunk.get('chunk_id', 'unknown')}, "
                        f"doc_id: {chunk.get('document_id', 'unknown')}, "
                        f"similarity: {chunk.get('similarity', 0.0):.3f}"
                    )
                    processed_chunks.append(chunk_str)
                elif isinstance(chunk, str):
                    processed_chunks.append(chunk)
                else:
                    processed_chunks.append(str(chunk))
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk {i}: {e}")
                continue
        
        return processed_chunks if processed_chunks else None
    
    def _create_question_object(
        self,
        question_data: Dict[str, Any],
        options: List[QuestionOption],
        language: Language,
        difficulty: Difficulty,
        question_type: QuestionType,
        document_id: int,
        session_id: str,
        topic: str,
        prompt: str,
        source_chunks: Optional[List[str]],
        context_confidence: float,
        response: Any,
        config: GenerationConfig
    ) -> QuestionCreate:
        """Create QuestionCreate object with error handling."""
        
        # Truncate prompt if too long
        truncated_prompt = prompt[:self.MAX_PROMPT_LENGTH] if len(prompt) > self.MAX_PROMPT_LENGTH else prompt
        
        generation_params = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "context_confidence": context_confidence,
            "topic": topic,
            "usage": getattr(response, 'usage', {"total_tokens": 0})
        }
        
        try:
            return QuestionCreate(
                # Required fields from QuestionBase
                question_text=question_data["question_text"],  # Already cleaned
                question_type=question_type,
                language=language,
                difficulty=difficulty,
                options=options,  # QuestionOption objects with normalized text
                explanation=self._update_explanation_indices(question_data["explanation"]),  # Already cleaned
                
                # Required fields from QuestionCreate
                document_id=document_id,
                session_id=session_id,
                
                # Optional fields
                generation_order=0,
                source_chunks=source_chunks,
                generation_prompt=truncated_prompt,
                model_used=getattr(response, 'model', 'unknown'),
                generation_params=generation_params
            )
            
        except ValidationError as e:
            # If validation fails due to source_chunks, try without them
            if "source_chunks" in str(e) and source_chunks:
                logger.warning(f"Source chunks validation failed, retrying without them: {e}")
                return QuestionCreate(
                    question_text=question_data["question_text"],  # Already cleaned
                    question_type=question_type,
                    language=language,
                    difficulty=difficulty,
                    options=options,  # QuestionOption objects with normalized text
                    explanation=self._update_explanation_indices(question_data["explanation"]),  # Already cleaned
                    document_id=document_id,
                    session_id=session_id,
                    generation_order=0,
                    source_chunks=None,
                    generation_prompt=truncated_prompt,
                    model_used=getattr(response, 'model', 'unknown'),
                    generation_params=generation_params
                )
            else:
                raise
    



# Global instance
_question_parser: QuestionParser | None = None


def get_question_parser() -> QuestionParser:
    """Get the global question parser instance."""
    global _question_parser
    if _question_parser is None:
        _question_parser = QuestionParser()
    return _question_parser