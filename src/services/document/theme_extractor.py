"""
QCM Generator Pro - Theme Extraction Service

This module handles automatic theme extraction from documents using
LLM-based analysis for intelligent topic identification.
"""

import json
import logging
from typing import Any, Optional, Union, List, Dict

from src.core.config import settings
from src.models.schemas import ThemeDetection
from src.services.llm.llm_manager import generate_llm_response, LLMError

logger = logging.getLogger(__name__)


class ThemeExtractionError(Exception):
    """Exception raised when theme extraction fails."""
    pass


class LLMThemeExtractor:
    """
    LLM-based theme extractor for intelligent topic identification.
    """
    
    def __init__(self):
        """Initialize theme extractor."""
        self.min_confidence = settings.processing.min_theme_confidence
        self.max_themes = settings.processing.max_themes_per_document
        self.keywords_limit = settings.processing.theme_keywords_limit
        
    def create_theme_extraction_prompt(self, text: str, language: str = "fr") -> str:
        """
        Create prompt for LLM theme extraction.
        
        Args:
            text: Document text to analyze
            language: Document language
            
        Returns:
            Formatted prompt for theme extraction
        """
        if language == "fr":
            prompt = f"""Analysez le texte suivant et identifiez les thèmes principaux.

TEXTE À ANALYSER:
{text[:3000]}{"..." if len(text) > 3000 else ""}

INSTRUCTIONS:
- Identifiez entre 2 et {self.max_themes} thèmes principaux
- Pour chaque thème, fournissez:
  * Un nom de thème clair et descriptif
  * Un score de confiance entre 0.0 et 1.0
  * 5-10 mots-clés représentatifs
  * Une courte description (1-2 phrases)
- Répondez UNIQUEMENT au format JSON suivant:

{{
  "themes": [
    {{
      "theme_name": "Nom du thème",
      "confidence_score": 0.8,
      "keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"],
      "description": "Description courte du thème"
    }}
  ]
}}"""
        else:
            prompt = f"""Analyze the following text and identify the main themes.

TEXT TO ANALYZE:
{text[:3000]}{"..." if len(text) > 3000 else ""}

INSTRUCTIONS:
- Identify between 2 and {self.max_themes} main themes
- For each theme, provide:
  * A clear and descriptive theme name
  * A confidence score between 0.0 and 1.0
  * 5-10 representative keywords
  * A short description (1-2 sentences)
- Respond ONLY in the following JSON format:

{{
  "themes": [
    {{
      "theme_name": "Theme name",
      "confidence_score": 0.8,
      "keywords": ["word1", "word2", "word3", "word4", "word5"],
      "description": "Short theme description"
    }}
  ]
}}"""
        
        return prompt
        
    async def call_llm_api(self, prompt: str) -> dict:
        """
        Call LLM API for theme extraction using the centralized LLM manager.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            LLM response as dictionary
            
        Raises:
            ThemeExtractionError: If API call fails
        """
        try:
            # Use the centralized LLM manager
            system_prompt = "You are an expert at analyzing documents and identifying themes. Always respond with valid JSON."
            
            response = await generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                model=settings.llm.openai_model,  # Use OpenAI model specifically
                temperature=settings.llm.default_temperature,
                max_tokens=1500  # Increased for theme extraction (was default_max_tokens=512)
            )
            
            content = response.content.strip()
            
            # Parse JSON response with robust error handling
            try:
                return self._parse_json_response(content)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {content}")
                raise ThemeExtractionError(f"Invalid JSON response from LLM: {e}")
                
        except LLMError as e:
            logger.error(f"LLM API error: {e}")
            raise ThemeExtractionError(f"LLM API call failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling LLM API: {e}")
            raise ThemeExtractionError(f"LLM API call failed: {e}")
            
    def create_fallback_themes(self, text: str, pages_data: Optional[List[Dict]] = None) -> List[ThemeDetection]:
        """
        Create basic fallback themes when LLM is unavailable.
        
        Args:
            text: Document text
            pages_data: Page data
            
        Returns:
            List of basic themes
        """
        logger.warning("Creating fallback themes (LLM unavailable)")
        
        # Simple word frequency analysis for fallback
        words = text.lower().split()
        word_freq = {}
        
        # Basic stop words
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'du', 'dans',
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of'
        }
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Create generic themes
        themes = []
        if top_words:
            theme = ThemeDetection(
                theme_name="Contenu principal",
                confidence_score=0.6,
                keywords=[word for word, _ in top_words[:10]],
                start_page=1,
                end_page=len(pages_data) if pages_data else 1,
                page_coverage=1.0,
                word_count=len(words),
                description="Thème principal identifié par analyse de fréquence des mots"
            )
            themes.append(theme)
            
        return themes
        
    async def extract_themes(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        pages_data: Optional[List[Dict[str, Any]]] = None,
        language: str = "fr"
    ) -> List[ThemeDetection]:
        """
        Extract themes from document text using LLM.
        
        Args:
            text: Document text
            metadata: Document metadata
            pages_data: Page-level data
            language: Document language
            
        Returns:
            List of detected themes
            
        Raises:
            ThemeExtractionError: If extraction fails
        """
        try:
            logger.info("Starting LLM-based theme extraction")
            
            if not text.strip():
                raise ThemeExtractionError("No text provided for theme extraction")
                
            # Create extraction prompt
            prompt = self.create_theme_extraction_prompt(text, language)
            
            try:
                # Call LLM API
                llm_response = await self.call_llm_api(prompt)
                
                # Parse themes from LLM response
                themes = []
                if "themes" in llm_response:
                    total_pages = len(pages_data) if pages_data else 1
                    
                    for theme_data in llm_response["themes"][:self.max_themes]:
                        # Validate theme data
                        if not all(key in theme_data for key in ["theme_name", "confidence_score", "keywords"]):
                            logger.warning(f"Incomplete theme data: {theme_data}")
                            continue
                            
                        confidence = float(theme_data["confidence_score"])
                        if confidence < self.min_confidence:
                            logger.debug(f"Theme below confidence threshold: {theme_data['theme_name']}")
                            continue
                            
                        theme = ThemeDetection(
                            theme_name=theme_data["theme_name"],
                            confidence_score=confidence,
                            keywords=theme_data.get("keywords", [])[:self.keywords_limit],
                            start_page=1,  # Could be enhanced with page-level analysis
                            end_page=total_pages,
                            page_coverage=1.0,  # Could be calculated based on keyword distribution
                            word_count=len(text.split()),
                            description=theme_data.get("description", f"Thème: {theme_data['theme_name']}")
                        )
                        themes.append(theme)
                        
                    if themes:
                        logger.info(f"LLM theme extraction completed: {len(themes)} themes detected")
                        return themes
                        
            except ThemeExtractionError:
                logger.warning("LLM theme extraction failed, using fallback method")
                
            # Fallback to simple analysis if LLM fails
            themes = self.create_fallback_themes(text, pages_data)
            
            if not themes:
                # Ultimate fallback - create generic theme
                theme = ThemeDetection(
                    theme_name="Document général",
                    confidence_score=0.5,
                    keywords=["contenu", "document", "texte"],
                    start_page=1,
                    end_page=len(pages_data) if pages_data else 1,
                    page_coverage=1.0,
                    word_count=len(text.split()),
                    description="Thème générique pour le contenu du document"
                )
                themes = [theme]
                
            logger.info(f"Theme extraction completed: {len(themes)} themes detected")
            return themes
            
        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
            raise ThemeExtractionError(f"Failed to extract themes: {e}")
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response with robust error handling and repair."""
        try:
            # Try direct parsing first
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
            
        # Remove markdown code fences if present
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # Find JSON boundaries
        start = content.find('{')
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", content, 0)
            
        # Try to extract a valid JSON substring
        json_part = content[start:]
        
        # Multiple repair strategies
        repair_attempts = [
            self._repair_truncated_content,  # Try this first for your specific case
            self._repair_unterminated_strings,
            self._repair_missing_braces
        ]
        
        for repair_func in repair_attempts:
            try:
                repaired = repair_func(json_part)
                if repaired:
                    result = json.loads(repaired)
                    logger.info("Successfully repaired JSON with strategy: " + repair_func.__name__)
                    return result
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.debug(f"Repair strategy {repair_func.__name__} failed: {e}")
                continue
        
        logger.error(f"All JSON repair strategies failed. Content preview: {json_part[:500]}...")
        raise json.JSONDecodeError("Failed to repair malformed JSON", content, start)
    
    def _repair_unterminated_strings(self, json_str: str) -> str:
        """Repair unterminated strings in JSON."""
        lines = json_str.split('\n')
        repaired_lines = []
        
        for line in lines:
            # Count quotes to detect unterminated strings
            quote_count = line.count('"')
            if quote_count % 2 == 1:  # Odd number of quotes - might be unterminated
                # Find last quote position
                last_quote = line.rfind('"')
                if last_quote >= 0:
                    # Check if this looks like an unterminated string at end of line
                    after_quote = line[last_quote + 1:].strip()
                    if not after_quote or after_quote.startswith(',') or after_quote.startswith('}'):
                        # Add closing quote before comma/brace
                        if after_quote.startswith(','):
                            line = line[:last_quote + 1] + '"' + after_quote
                        elif after_quote.startswith('}'):
                            line = line[:last_quote + 1] + '","' + after_quote
                        else:
                            line = line + '"'
            
            repaired_lines.append(line)
        
        return '\n'.join(repaired_lines)
    
    def _repair_missing_braces(self, json_str: str) -> str:
        """Repair missing braces in JSON."""
        # Count braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        
        if open_braces > close_braces:
            # Add missing closing braces
            return json_str + '}' * (open_braces - close_braces)
        
        return json_str
    
    def _repair_truncated_content(self, json_str: str) -> str:
        """Repair truncated JSON by finding last valid structure."""
        # Remove any trailing incomplete text that might be causing issues
        lines = json_str.split('\n')
        complete_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Skip incomplete lines that end with incomplete strings or structures
            if (stripped.endswith('"') and stripped.count('"') % 2 == 1) or \
               stripped.endswith(',') or \
               ('"' in stripped and not stripped.endswith('",') and not stripped.endswith('"')):
                # This line looks incomplete, try to fix it or skip it
                if stripped.endswith('"') and not stripped.endswith('",'):
                    # Add missing comma after string
                    complete_lines.append(stripped[:-1] + '",')
                elif '"' in stripped and stripped.count('"') % 2 == 1:
                    # Incomplete string, truncate at last complete quote
                    last_quote = stripped.rfind('"', 0, -1)
                    if last_quote > 0:
                        complete_lines.append(stripped[:last_quote] + '"')
                # Otherwise skip this incomplete line
                break
            else:
                complete_lines.append(line)
        
        if not complete_lines:
            return json_str
        
        reconstructed = '\n'.join(complete_lines)
        
        # Find the last complete object or array
        last_brace = reconstructed.rfind('}')
        last_bracket = reconstructed.rfind(']')
        
        # Take the position that's furthest to the right
        last_valid = max(last_brace, last_bracket)
        
        if last_valid > 0:
            truncated = reconstructed[:last_valid + 1]
            # Ensure we have balanced braces
            return self._repair_missing_braces(truncated)
        
        # If no valid structure found, try to close what we have
        return self._repair_missing_braces(reconstructed)


# Convenience functions
async def extract_document_themes(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    pages_data: Optional[List[Dict[str, Any]]] = None,
    language: str = "fr"
) -> List[ThemeDetection]:
    """
    Extract themes from document using LLM.
    
    Args:
        text: Document text
        metadata: Document metadata  
        pages_data: Page-level data
        language: Document language
        
    Returns:
        List of detected themes
    """
    extractor = LLMThemeExtractor()
    return await extractor.extract_themes(text, metadata, pages_data, language)


def extract_document_themes_sync(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    pages_data: Optional[List[Dict[str, Any]]] = None,
    language: str = "fr"
) -> List[ThemeDetection]:
    """
    Synchronous wrapper for theme extraction.
    
    Args:
        text: Document text
        metadata: Document metadata  
        pages_data: Page-level data
        language: Document language
        
    Returns:
        List of detected themes
    """
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(
        extract_document_themes(text, metadata, pages_data, language)
    )