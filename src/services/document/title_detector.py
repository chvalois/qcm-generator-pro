"""
QCM Generator Pro - Intelligent Title Detection Service

This module provides automatic detection of document title hierarchy (H1-H4)
from PDF text using multiple heuristics and pattern recognition.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TitleCandidate:
    """Represents a potential title with its characteristics."""
    text: str
    line_number: int
    page_number: int
    level: int
    confidence: float
    features: Dict[str, Any]
    
    
@dataclass
class TitleHierarchy:
    """Represents the document's title hierarchy."""
    h1_title: Optional[str] = None
    h2_title: Optional[str] = None  
    h3_title: Optional[str] = None
    h4_title: Optional[str] = None
    
    def get_full_path(self) -> str:
        """Get the full hierarchical path as a string."""
        titles = [self.h1_title, self.h2_title, self.h3_title, self.h4_title]
        return " > ".join(title for title in titles if title)
    
    def get_level_title(self, level: int) -> Optional[str]:
        """Get title for a specific level."""
        if level == 1:
            return self.h1_title
        elif level == 2:
            return self.h2_title
        elif level == 3:
            return self.h3_title
        elif level == 4:
            return self.h4_title
        return None


class TitleDetector:
    """
    Intelligent title detection service using multiple heuristics.
    
    Analyzes text patterns, formatting, positioning, and content to automatically
    identify document title hierarchy (H1, H2, H3, H4).
    """
    
    def __init__(self, custom_config=None):
        """Initialize title detector with configurable patterns."""
        self.custom_config = custom_config
        # Common title patterns and indicators
        self.title_patterns = {
            'numbered': [
                r'^(\d+\.?\s+.+)$',  # "1. Introduction"
                r'^(\d+\.\d+\.?\s+.+)$',  # "1.1. Overview"
                r'^(\d+\.\d+\.\d+\.?\s+.+)$',  # "1.1.1. Details"
                r'^(\d+\.\d+\.\d+\.\d+\.?\s+.+)$',  # "1.1.1.1. Sub-details"
            ],
            'lettered': [
                r'^([A-Z]\.?\s+.+)$',  # "A. Section"
                r'^([a-z]\.?\s+.+)$',  # "a. Subsection"
            ],
            'roman': [
                r'^([IVX]+\.?\s+.+)$',  # "I. Introduction"
                r'^([ivx]+\.?\s+.+)$',  # "i. subsection"
            ],
            'keywords': [
                r'^(Parcours\s+\d+.*)$',           # Parcours = H1
                r'^(Module\s+\d+.*)$',             # Module = H2  
                r'^(Unit[ée]\s+\d+.*)$',           # Unité = H3
                r'^(Chapitre\s+\d+.*)$',
                r'^(Chapter\s+\d+.*)$',
                r'^(Leçon\s+\d+.*)$',
                r'^(Lesson\s+\d+.*)$',
                r'^(Section\s+\d+.*)$',
                r'^(Partie\s+\d+.*)$',
                r'^(Part\s+\d+.*)$',
            ]
        }
        
        # Font size and formatting indicators (estimated from text analysis)
        self.formatting_indicators = {
            'all_caps': r'^[A-Z\s\d\.\-\(\)]+$',
            'title_case': r'^[A-Z][a-z\s\d\.\-\(\)]*(?:[A-Z][a-z\s\d\.\-\(\)]*)*$',
            'short_line': lambda text: len(text.strip()) < 80,  # Titles are usually shorter
            'no_final_punctuation': lambda text: not text.strip().endswith(('.', '!', '?', ';', ':')),
        }
        
        # Common stop words for filtering non-titles
        self.stop_patterns = [
            r'^\s*$',  # Empty lines
            r'^Page\s+\d+',  # Page numbers
            r'^\d+\s*$',  # Just numbers
            r'^[^\w]+$',  # Only punctuation
            r'^(Figure|Table|Tableau|Image)\s+\d+',  # Figure/table captions
            r'^\(.*\)$',  # Text in parentheses only
        ]
        
    def detect_titles_in_text(self, text: str, pages_data: List[Dict[str, Any]]) -> List[TitleCandidate]:
        """
        Detect title candidates in the document text.
        
        Args:
            text: Full document text
            pages_data: Page-by-page text data
            
        Returns:
            List of title candidates with confidence scores
        """
        logger.info("Starting intelligent title detection")
        
        title_candidates = []
        
        # Process each page to maintain page context
        for page_data in pages_data:
            page_num = page_data['page_number']
            page_text = page_data.get('text', '')
            
            if not page_text.strip():
                continue
                
            # Split into lines for analysis
            lines = page_text.split('\n')
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                
                if not line or self._is_stop_pattern(line):
                    continue
                
                # Analyze this line as a potential title
                candidate = self._analyze_title_candidate(
                    line, line_num, page_num, lines, page_text
                )
                
                if candidate and candidate.confidence > 0.3:  # Minimum confidence threshold
                    title_candidates.append(candidate)
        
        # Debug: check Module 3 titles before refinement
        module3_before = [c for c in title_candidates if "module 3" in c.text.lower()]
        logger.debug(f"Found {len(module3_before)} Module 3 titles BEFORE refinement:")
        for title in module3_before:
            logger.debug(f"  Page {getattr(title, 'page_number', '?')}: {title.text}")
        
        # Post-process and refine candidates
        refined_candidates = self._refine_title_candidates(title_candidates)
        
        logger.info(f"Detected {len(refined_candidates)} title candidates after refinement")
        
        # Debug: log duplicate removal stats
        original_count = len(title_candidates)
        refined_count = len(refined_candidates)
        logger.debug(f"Title refinement: {original_count} -> {refined_count} candidates")
        
        # Debug: check for Module 3 titles
        module3_titles = [c for c in refined_candidates if "module 3" in c.text.lower()]
        if module3_titles:
            logger.debug(f"Found {len(module3_titles)} Module 3 titles after refinement:")
            for title in module3_titles:
                logger.debug(f"  Page {getattr(title, 'page_number', '?')}: {title.text}")
        return refined_candidates
    
    def _analyze_title_candidate(
        self, 
        line: str, 
        line_num: int, 
        page_num: int, 
        context_lines: List[str],
        page_text: str
    ) -> Optional[TitleCandidate]:
        """Analyze a single line as a potential title."""
        
        features = {}
        confidence = 0.0
        estimated_level = 1
        
        # Check custom patterns first (highest priority)
        custom_score, custom_level = self._analyze_custom_patterns(line)
        if custom_score > 0:
            confidence += custom_score
            estimated_level = custom_level
            features['custom_score'] = custom_score
            features['custom_level'] = custom_level
            # If we have a strong custom match, return early
            if custom_score > 0.9:
                return TitleCandidate(
                    text=line,
                    line_number=line_num,
                    page_number=page_num,
                    level=estimated_level,
                    confidence=custom_score,
                    features=features
                )
        
        # Check if we should use automatic detection
        use_auto_detection = True
        
        if self.custom_config:
            # Check if we have any patterns defined
            has_patterns = any([
                self.custom_config.h1_patterns,
                self.custom_config.h2_patterns, 
                self.custom_config.h3_patterns,
                self.custom_config.h4_patterns
            ])
            
            if has_patterns and not self.custom_config.use_auto_detection:
                # Strict custom-only mode: only proceed if we found a custom pattern match
                if custom_score == 0:
                    return None  # No custom pattern match, skip this line
                # For custom matches, only add minimal additional scoring
                confidence += 0.1  # Small boost for context
                features['pattern_score'] = 0
                features['format_score'] = 0
                features['context_score'] = 0
                features['position_score'] = 0
                features['structure_score'] = 0
                use_auto_detection = False
                logger.debug(f"Custom-only mode: accepted '{line[:50]}...' with score {confidence}")
            elif has_patterns and custom_score > 0:
                # Custom patterns found, but auto-detection is enabled
                # Still prioritize custom matches by skipping auto-detection for this line
                confidence += 0.1  # Small boost for context
                features['pattern_score'] = 0
                features['format_score'] = 0
                features['context_score'] = 0
                features['position_score'] = 0
                features['structure_score'] = 0
                use_auto_detection = False
                logger.debug(f"Custom priority mode: accepted '{line[:50]}...' with score {confidence}")
            # If no custom patterns defined or no custom match, will use auto-detection
        
        # Apply automatic detection if needed
        if use_auto_detection:
            # Standard analysis with automatic detection
            # Pattern matching analysis (automatic detection)
            pattern_score, pattern_level = self._analyze_patterns(line)
            if not custom_level:  # Only use if no custom level found
                confidence += pattern_score
                if pattern_level:
                    estimated_level = pattern_level
            features['pattern_score'] = pattern_score
            features['pattern_level'] = pattern_level
            
            # Formatting analysis
            format_score = self._analyze_formatting(line)
            confidence += format_score
            features['format_score'] = format_score
            
            # Context analysis (surrounding lines)
            context_score = self._analyze_context(line, line_num, context_lines)
            confidence += context_score
            features['context_score'] = context_score
            
            # Position analysis (early in page/document)
            position_score = self._analyze_position(line_num, page_num, len(context_lines))
            confidence += position_score
            features['position_score'] = position_score
            
            # Length and structure analysis
            structure_score = self._analyze_structure(line)
            confidence += structure_score
            features['structure_score'] = structure_score
        
        # Normalize confidence (max possible score is ~2.5)
        confidence = min(confidence / 2.5, 1.0)
        
        return TitleCandidate(
            text=line,
            line_number=line_num,
            page_number=page_num,
            level=estimated_level,
            confidence=confidence,
            features=features
        )
    
    def _analyze_patterns(self, line: str) -> Tuple[float, Optional[int]]:
        """Analyze title numbering and keyword patterns with hierarchical logic."""
        score = 0.0
        level = None
        
        # Hierarchical analysis based on numbering type and educational structure
        # Order of hierarchy: Roman uppercase > Arabic numbers > Roman lowercase > Letters
        # Educational structure: Parcours (H1) > Module (H2) > Unité (H3)
        
        # Check for educational structure keywords first (highest priority)
        # But only if they appear to be title-like (start of line or with numbers)
        if re.search(r'^parcours\s+\d+|^parcours\s*:', line, re.IGNORECASE):
            score += 0.9
            level = 1  # Parcours = H1
            return score, level
        
        if re.search(r'^module\s+\d+|^module\s*:', line, re.IGNORECASE):
            score += 0.9
            level = 2  # Module = H2
            return score, level
        
        # Enhanced pattern for "Unité" with various formats
        unite_patterns = [
            r'^unité\s+\d+[^\n]*$',     # "Unité 7: Titre"
            r'^unité\s*:\s*[^\n]*$',    # "Unité: Titre" 
            r'^\d+[\.]?\s*unité[^\n]*$', # "7. Unité Titre" or "7 Unité Titre"
        ]
        
        for pattern in unite_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                score += 0.95  # Higher confidence for Unité patterns
                level = 3  # Unité = H3
                return score, level
        
        # Check Roman numerals (uppercase) - typically H1 or H2 level
        roman_upper_pattern = r'^([IVXLCDM]+\.?\s+.+)$'
        if re.match(roman_upper_pattern, line):
            score += 0.8
            level = 1  # High-level sections
            return score, level
        
        # Check Arabic numbered patterns - multiple levels based on depth
        arabic_patterns = [
            (r'^(\d+\.?\s+.+)$', 2),           # "1. Title" -> H2
            (r'^(\d+\.\d+\.?\s+.+)$', 3),      # "1.1. Title" -> H3  
            (r'^(\d+\.\d+\.\d+\.?\s+.+)$', 4), # "1.1.1. Title" -> H4
            (r'^(\d+\.\d+\.\d+\.\d+\.?\s+.+)$', 4), # "1.1.1.1. Title" -> H4 (max depth)
        ]
        
        for pattern, suggested_level in arabic_patterns:
            if re.match(pattern, line):
                score += 0.8
                level = suggested_level
                return score, level
        
        # Check Roman numerals (lowercase) - typically H3 level
        roman_lower_pattern = r'^([ivxlcdm]+\.?\s+.+)$'
        if re.match(roman_lower_pattern, line):
            score += 0.6
            level = 3
            return score, level
        
        # Check lettered patterns - typically H4 level
        # Updated to avoid matching acronyms (2+ capital letters)
        letter_patterns = [
            (r'^([A-Z]\.\s+.+)$', 4),         # "A. Title" -> H4 (exactly one letter + dot)
            (r'^([A-Z]\s[a-z].+)$', 4),       # "A Title" -> H4 (one letter + space + lowercase)
            (r'^([a-z]\.\s+.+)$', 4),         # "a. Title" -> H4
        ]
        
        for pattern, suggested_level in letter_patterns:
            if re.match(pattern, line):
                # Additional check: make sure it's not an acronym
                first_word = line.split()[0]
                if len(first_word) == 1 or (len(first_word) == 2 and first_word.endswith('.')):
                    # It's a single letter or single letter with dot - valid
                    score += 0.4
                    level = suggested_level
                    return score, level
                else:
                    # It's an acronym or multi-letter word - skip this pattern
                    logger.debug(f"Skipped lettered pattern for acronym: '{first_word}'")
                    continue
        
        # Anti-false-positive filters - Check for patterns that should NOT be titles
        false_positive_patterns = [
            r'^[A-Z]{2,5}\s+est\s+appliqué',      # "DDM est appliqué", "PKI est appliqué", etc.
            r'^[A-Z]{2,5}\s+est\s+utilisé',       # "SQL est utilisé", etc.
            r'^[A-Z]{2,5}\s+doit\s+être',         # "DDM doit être", etc.
            r'^[A-Z]{2,5}\s+peut\s+être',         # "API peut être", etc.
            r'Au lieu de cela',                   # Content phrases with "Au lieu de cela"
            r'les règles de masquage',            # Specific content phrase
            r'sont appliquées aux résultats',     # Specific content phrase
        ]
        
        # Check if this line matches any false positive patterns
        for fp_pattern in false_positive_patterns:
            if re.search(fp_pattern, line, re.IGNORECASE):
                # This is likely content, not a title - reduce score significantly
                score = max(0.0, score - 0.9)
                level = None
                logger.debug(f"False positive filter triggered for: '{line[:50]}...' with pattern: {fp_pattern}")
                return score, level
        
        # Additional content detection filters
        if (len(line.strip()) > 100 and  # Long lines are rarely titles
            ('.' in line[10:] or ',' in line[10:])):  # Contains punctuation mid-sentence
            score = max(0.0, score - 0.3)  # Reduce confidence for long sentences
            logger.debug(f"Long sentence filter applied to: '{line[:50]}...'")
        
        # Check keyword patterns with educational hierarchy
        for pattern in self.title_patterns['keywords']:
            if re.match(pattern, line, re.IGNORECASE):
                score += 0.8  # Higher score for keyword patterns
                if not level:  # Only set if not already set by previous checks
                    # Educational hierarchy: Parcours > Module > Unité
                    if any(word in line.lower() for word in ['parcours']):
                        level = 1  # Parcours = H1
                    elif any(word in line.lower() for word in ['module']):
                        level = 2  # Module = H2
                    elif any(word in line.lower() for word in ['unité', 'unite']):
                        level = 3  # Unité = H3
                    # Other educational terms
                    elif any(word in line.lower() for word in ['chapitre', 'chapter']):
                        level = 1  # High-level
                    elif any(word in line.lower() for word in ['leçon', 'lesson', 'section']):
                        level = 2  # Mid-level
                    elif any(word in line.lower() for word in ['partie', 'part']):
                        level = 1  # High-level
                    else:
                        level = 2  # Default for other keywords
                break
        
        return score, level
    
    def _analyze_custom_patterns(self, line: str) -> Tuple[float, Optional[int]]:
        """Analyze line using user-defined custom patterns."""
        if not self.custom_config:
            return 0.0, None
        
        # Check each level's patterns
        for level, patterns in [
            (1, self.custom_config.h1_patterns),
            (2, self.custom_config.h2_patterns), 
            (3, self.custom_config.h3_patterns),
            (4, self.custom_config.h4_patterns)
        ]:
            for pattern in patterns:
                # Create regex pattern for exact matching
                if self._matches_custom_pattern(line, pattern):
                    return 0.95, level  # Very high confidence for user patterns
        
        return 0.0, None
    
    def _matches_custom_pattern(self, line: str, pattern: str) -> bool:
        """Check if line matches a custom pattern with intelligent placeholder support."""
        line_stripped = line.strip()
        pattern_stripped = pattern.strip()
        
        # Convert pattern to regex with intelligent placeholders
        regex_pattern = self._convert_pattern_to_regex(pattern_stripped)
        
        # Test the regex pattern against the line
        if re.match(regex_pattern, line_stripped, re.IGNORECASE):
            return True
        
        # Also test with bullet points removed
        if line_stripped.lower().startswith(('• ', '- ', '* ', '▪ ', '▫ ')):
            clean_line = re.sub(r'^[•\-*▪▫]\s*', '', line_stripped)
            if re.match(regex_pattern, clean_line, re.IGNORECASE):
                return True
        
        return False
    
    def _convert_pattern_to_regex(self, pattern: str) -> str:
        """Convert user pattern to regex with intelligent placeholders."""
        # Escape special regex characters except our placeholders
        escaped_pattern = re.escape(pattern)
        
        # Define intelligent placeholders and their regex equivalents
        placeholders = {
            # Numbers: 1, 2, 3, etc.
            r'\\d\+': r'\d+',  # \d+ was escaped, restore it
            
            # Roman numerals (uppercase): I, II, III, IV, V, etc.
            r'I+': r'[IVX]+',
            
            # Roman numerals (lowercase): i, ii, iii, iv, v, etc.
            r'i+': r'[ivx]+',
            
            # Letters (uppercase): A, B, C, etc.
            r'[A-Z]': r'[A-Z]',
            
            # Letters (lowercase): a, b, c, etc.
            r'[a-z]': r'[a-z]',
        }
        
        # Handle {X} placeholders first (before escaping)
        if '{X}' in pattern:
            # For {X}, we should use the combined pattern since it's meant to be flexible
            # But limit each type to 20 maximum and exclude conflicts
            number_pattern = r'(?:20|1[0-9]|[1-9])(?!\\d)'  # 1-20
            roman_upper_pattern = r'(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)(?![IVX])'  # I-XX
            roman_lower_pattern = r'(?:xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)(?![ivx])'  # i-xx
            
            # Letters: A-T (20 letters) - for {X} we allow all letters A-T
            letters_upper_pattern = r'[A-T]'  # A-T (20 letters)
            letters_lower_pattern = r'[a-t]'  # a-t (20 letters)
            
            replacement_pattern = f'(?:{number_pattern}|{roman_upper_pattern}|{roman_lower_pattern}|{letters_upper_pattern}|{letters_lower_pattern})'
            escaped_pattern = re.sub(r'\\\{X\\\}', replacement_pattern, escaped_pattern)
        
        # Only do auto-detection if {X} placeholder was NOT used
        # (to avoid overriding the limited patterns)
        if '{X}' not in pattern:
            # 1. Pattern "I." => template for roman numerals I to XX (I., II., III., ..., XX.) - CHECK FIRST!
            if re.match(r'^[IVX]+\.$', pattern):
                # Replace the roman numeral with the full I-XX pattern
                roman_part = pattern.split('.')[0]  # Get "I", "V", "X", etc.
                escaped_pattern = escaped_pattern.replace(roman_part, '(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)')
            # 2. Pattern "i." => template for roman numerals i to xx (i., ii., iii., ..., xx.)
            elif re.match(r'^[ivx]+\.$', pattern):
                # Replace the roman numeral with the full i-xx pattern  
                roman_part = pattern.split('.')[0]  # Get "i", "v", "x", etc.
                escaped_pattern = escaped_pattern.replace(roman_part, '(?:xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)')
            # 3. Pattern "1." => template for arabic numbers 1 to 20 (1., 2., 3., ..., 20.)
            elif re.match(r'^\d+\.$', pattern):
                # Replace the number with 1-20 pattern
                number_part = re.search(r'\d+', pattern).group()  # Get "1", "2", etc.
                escaped_pattern = escaped_pattern.replace(number_part, '(?:20|1[0-9]|[1-9])')
            # 4. Pattern "A." => template for letters A to T (A., B., C., ..., T.) - CHECK AFTER ROMAN NUMERALS!
            elif re.match(r'^[A-Z]\.$', pattern):
                # Replace the specific letter (e.g., "A") with [A-T] pattern
                letter = pattern[0]  # Get the letter (A, B, C, etc.)
                escaped_pattern = escaped_pattern.replace(letter, '[A-T]')
            # 5. Pattern "Parcours 1" => template for "Parcours 1" to "Parcours 20"
            elif re.search(r'\d+', pattern):
                # Replace any number in the pattern with 1-20 range
                number_match = re.search(r'\d+', pattern)
                if number_match:
                    number_part = number_match.group()
                    escaped_pattern = escaped_pattern.replace(number_part, '(?:20|1[0-9]|[1-9])')
            # General roman numeral patterns (not just dots) - for backwards compatibility
            elif re.search(r'\b[IVX]+\b', pattern):
                escaped_pattern = re.sub(r'\\[IVX]+', r'(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)(?![IVX])', escaped_pattern)
            elif re.search(r'\b[ivx]+\b', pattern):
                escaped_pattern = re.sub(r'\\[ivx]+', r'(?:xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)(?![ivx])', escaped_pattern)
            # Single letter patterns (not dots, not roman numerals) - for backwards compatibility
            elif re.search(r'\b[A-Z]\b', pattern) and not re.search(r'[IVX]', pattern):
                escaped_pattern = re.sub(r'\\[A-Z]', r'[A-T]', escaped_pattern)
            elif re.search(r'\b[a-z]\b', pattern) and not re.search(r'[ivx]', pattern):
                escaped_pattern = re.sub(r'\\[a-z]', r'[a-t]', escaped_pattern)
        
        # Restore escaped special characters we need
        escaped_pattern = escaped_pattern.replace(r'\\.', r'\\.')  # Periods
        escaped_pattern = escaped_pattern.replace(r'\\:', r':')     # Colons
        escaped_pattern = escaped_pattern.replace(r'\\ ', r'\\s+')  # Spaces (flexible)
        
        # Ensure pattern starts at beginning of line and allows continuation
        final_pattern = f"^{escaped_pattern}\\s*[:\\-.]?\\s*.*"
        
        return final_pattern
    
    def _analyze_formatting(self, line: str) -> float:
        """Analyze text formatting indicators."""
        score = 0.0
        
        # All caps (often titles)
        if re.match(self.formatting_indicators['all_caps'], line):
            score += 0.4
        
        # Title case
        elif re.match(self.formatting_indicators['title_case'], line):
            score += 0.2
        
        # Short line (titles are usually shorter)
        if self.formatting_indicators['short_line'](line):
            score += 0.2
        
        # No final punctuation (titles often don't end with periods)
        if self.formatting_indicators['no_final_punctuation'](line):
            score += 0.1
        
        # Contains digits (often section numbers)
        if re.search(r'\d', line):
            score += 0.1
        
        return score
    
    def _analyze_context(self, line: str, line_num: int, context_lines: List[str]) -> float:
        """Analyze surrounding context."""
        score = 0.0
        
        # Check if followed by empty line (common for titles)
        if line_num + 1 < len(context_lines):
            next_line = context_lines[line_num + 1].strip()
            if not next_line:
                score += 0.2
        
        # Check if preceded by empty line
        if line_num > 0:
            prev_line = context_lines[line_num - 1].strip()
            if not prev_line:
                score += 0.1
        
        # Check if followed by paragraph text (longer lines)
        following_text_lines = 0
        for i in range(line_num + 1, min(line_num + 4, len(context_lines))):
            next_line = context_lines[i].strip()
            if next_line and len(next_line) > 50:  # Paragraph-like text
                following_text_lines += 1
        
        if following_text_lines >= 2:
            score += 0.3
        
        return score
    
    def _analyze_position(self, line_num: int, page_num: int, total_lines: int) -> float:
        """Analyze position in page/document."""
        score = 0.0
        
        # Earlier in page
        if line_num < total_lines * 0.2:  # First 20% of page
            score += 0.2
        elif line_num < total_lines * 0.5:  # First 50% of page
            score += 0.1
        
        # Earlier pages are more likely to have main titles
        if page_num <= 3:
            score += 0.1
        
        # Top of page (excluding header area)
        if 2 <= line_num <= 8:
            score += 0.1
        
        return score
    
    def _analyze_structure(self, line: str) -> float:
        """Analyze line structure and content."""
        score = 0.0
        
        # Appropriate length for titles (not too short, not too long)
        length = len(line.strip())
        if 10 <= length <= 100:
            score += 0.2
        elif 5 <= length <= 150:
            score += 0.1
        
        # Contains meaningful words (not just numbers/symbols)
        word_count = len(re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', line))
        if word_count >= 2:
            score += 0.2
        elif word_count >= 1:
            score += 0.1
        
        # Balanced punctuation
        if line.count('(') == line.count(')'):
            score += 0.05
        
        return score
    
    def _is_stop_pattern(self, line: str) -> bool:
        """Check if line matches stop patterns (should not be a title)."""
        for pattern in self.stop_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _refine_title_candidates(self, candidates: List[TitleCandidate]) -> List[TitleCandidate]:
        """Refine and filter title candidates."""
        if not candidates:
            return []
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicates and very similar titles, but allow duplicates on distant pages
        refined = []
        seen_titles = []  # Store tuples of (normalized_text, page_number, candidate)
        
        for candidate in candidates:
            # Normalize text for comparison
            normalized = re.sub(r'\s+', ' ', candidate.text.lower().strip())
            current_page = getattr(candidate, 'page_number', 0)
            
            # Check for exact duplicates on nearby pages (within 5 pages)
            is_nearby_duplicate = False
            for seen_text, seen_page, seen_candidate in seen_titles:
                if normalized == seen_text:
                    page_distance = abs(current_page - seen_page) if current_page and seen_page else float('inf')
                    if page_distance <= 5:  # Only consider duplicates if within 5 pages
                        # Keep the one with higher confidence
                        if candidate.confidence <= seen_candidate.confidence:
                            is_nearby_duplicate = True
                            break
                        else:
                            # Remove the lower confidence duplicate
                            seen_titles = [(t, p, c) for t, p, c in seen_titles if not (t == seen_text and p == seen_page)]
                            refined = [c for c in refined if c != seen_candidate]
            
            if not is_nearby_duplicate:
                # Check for substring duplicates (but allow distant exact matches)
                is_substring_duplicate = False
                for seen_text, seen_page, seen_candidate in seen_titles:
                    if normalized != seen_text:  # Only check substrings if not exact match
                        if (normalized in seen_text and len(normalized) < len(seen_text) * 0.8) or \
                           (seen_text in normalized and len(seen_text) < len(normalized) * 0.8):
                            is_substring_duplicate = True
                            break
                
                if not is_substring_duplicate:
                    seen_titles.append((normalized, current_page, candidate))
                    refined.append(candidate)
        
        # Return all refined candidates without limits
        return refined
    
    def build_title_hierarchy_for_chunks(
        self, 
        chunks: List[str], 
        title_candidates: List[TitleCandidate],
        full_text: str,
        pages_data: List[Dict[str, Any]],
        chunks_with_pages: List[Dict[str, Any]] = None
    ) -> List[TitleHierarchy]:
        """
        Build title hierarchy for each chunk.
        
        Args:
            chunks: List of text chunks
            title_candidates: Detected title candidates
            full_text: Complete document text
            pages_data: Page-by-page data for accurate positioning
            
        Returns:
            List of TitleHierarchy objects, one per chunk
        """
        chunk_hierarchies = []
        
        # Filter title candidates with more flexible criteria
        filtered_titles = []
        for title in title_candidates:
            confidence = title.confidence
            pattern_score = title.features.get('pattern_score', 0)
            format_score = title.features.get('format_score', 0)
            
            # General criteria for keeping titles (no hard-coded exceptions):
            keep_title = (
                # Strong pattern recognition (numbered, lettered titles)
                (confidence > 0.7 and pattern_score > 0.5) or
                
                # Strong formatting indicators (ALL CAPS, good structure)
                (confidence > 0.6 and format_score > 0.6) or
                
                # Moderate confidence with some pattern recognition
                (confidence > 0.5 and pattern_score > 0.3) or
                
                # High confidence alone (for well-formatted titles)
                (confidence > 0.8)
            )
            
            if keep_title:
                filtered_titles.append(title)
        
        logger.info(f"Filtered {len(title_candidates)} candidates to {len(filtered_titles)} real titles")
        
        # Use chunks_with_pages if available, otherwise fallback to legacy method
        if chunks_with_pages:
            return self._build_hierarchy_with_page_info(chunks_with_pages, filtered_titles)
        else:
            return self._build_hierarchy_legacy(chunks, filtered_titles, full_text, pages_data)
    
    def _build_hierarchy_with_page_info(
        self,
        chunks_with_pages: List[Dict[str, Any]],
        title_candidates: List[TitleCandidate]
    ) -> List[TitleHierarchy]:
        """Build hierarchy using chunks with page information."""
        chunk_hierarchies = []
        
        # Sort titles by page number for proper hierarchy assignment
        sorted_titles = sorted(
            title_candidates,
            key=lambda t: (getattr(t, 'page_number', 1), getattr(t, 'text', ''))
        )
        
        logger.info(f"Processing {len(chunks_with_pages)} chunks with page info")
        
        # DEBUG: Afficher les détails complets des titres pour diagnostiquer
        logger.info("=== DEBUG TITLE ATTRIBUTION ===")
        for title in sorted_titles:
            logger.info(f"Title: '{title.text[:50]}...' | Level: {title.level} | Page: {getattr(title, 'page_number', 'MISSING')}")
        logger.info("=== END DEBUG ===")
        logger.info(f"Available titles: {[(t.text[:30], t.level, getattr(t, 'page_number', '?')) for t in sorted_titles]}")
        
        for i, chunk_data in enumerate(chunks_with_pages):
            hierarchy = TitleHierarchy()
            
            # Get chunk page information
            chunk_start_page = chunk_data.get('start_page', 1)
            chunk_end_page = chunk_data.get('end_page', chunk_start_page)
            
            logger.info(f"DEBUG Chunk {i}: pages {chunk_start_page}-{chunk_end_page}, text: {chunk_data.get('text', '')[:50]}...")
            
            # DEBUG: Afficher tous les titres candidats pour ce chunk
            applicable_titles = [t for t in sorted_titles if getattr(t, 'page_number', 1) <= chunk_end_page]
            logger.info(f"  -> Applicable titles for chunk {i}: {[(t.text[:30], t.level, getattr(t, 'page_number', '?')) for t in applicable_titles]}")
            
            # Find applicable titles for this chunk's page range
            current_titles = {1: None, 2: None, 3: None, 4: None}
            
            for title in sorted_titles:
                title_page = getattr(title, 'page_number', 1)
                
                # Title applies if it's on or before the chunk's page range
                # BUT: We need to prevent later titles from overriding earlier ones inappropriately
                if title_page <= chunk_end_page:
                    level = title.level
                    
                    # For better accuracy, prefer titles that are closer to the chunk
                    # If we already have a title at this level, only replace it if:
                    # 1. The new title is closer to the chunk start, OR
                    # 2. The new title is within the chunk range (more specific)
                    should_replace = True
                    if current_titles[level] is not None:
                        # Find the page of the current title
                        current_title_page = None
                        for existing_title in sorted_titles:
                            if existing_title.text == current_titles[level]:
                                current_title_page = getattr(existing_title, 'page_number', 1)
                                break
                        
                        if current_title_page is not None:
                            # Calculate distances from chunk START (not end)
                            current_distance = abs(chunk_start_page - current_title_page)
                            new_distance = abs(chunk_start_page - title_page)
                            
                            # FIXED LOGIC: Only replace if the new title is closer to chunk START
                            # AND either:
                            # - The new title is within the chunk range, OR  
                            # - The new title is significantly closer (at least 2 pages)
                            if new_distance < current_distance:
                                # New title is closer to chunk start
                                if (chunk_start_page <= title_page <= chunk_end_page or  # Title within chunk
                                    current_distance - new_distance >= 2):  # Significantly closer
                                    should_replace = True
                                else:
                                    should_replace = False
                            else:
                                # New title is farther or same distance - don't replace
                                should_replace = False
                    
                    if should_replace:
                        current_titles[level] = title.text
                        
                        # FIXED: Only clear lower-level titles if this title comes from a LATER page
                        # If titles are on the same page, they likely form a proper hierarchy
                        should_clear_lower_levels = False
                        
                        # Check if any existing lower-level titles are from earlier pages
                        for lower_level in range(level + 1, 5):
                            if current_titles[lower_level] is not None:
                                # Find the page of the existing lower-level title
                                existing_title_page = None
                                for existing_title in sorted_titles:
                                    if existing_title.text == current_titles[lower_level]:
                                        existing_title_page = getattr(existing_title, 'page_number', 1)
                                        break
                                
                                # Only clear if the current title is from a later page
                                if existing_title_page is not None and title_page > existing_title_page:
                                    should_clear_lower_levels = True
                                    break
                        
                        # Clear lower-level titles only if they're from earlier pages
                        if should_clear_lower_levels:
                            for lower_level in range(level + 1, 5):
                                if current_titles[lower_level] is not None:
                                    # Check if this lower-level title is from an earlier page
                                    existing_title_page = None
                                    for existing_title in sorted_titles:
                                        if existing_title.text == current_titles[lower_level]:
                                            existing_title_page = getattr(existing_title, 'page_number', 1)
                                            break
                                    
                                    if existing_title_page is not None and title_page > existing_title_page:
                                        current_titles[lower_level] = None
                            
                        logger.info(
                            f"Chunk pages {chunk_start_page}-{chunk_end_page}: "
                            f"Applied H{level} '{title.text[:30]}...' from page {title_page}"
                        )
                    else:
                        logger.info(
                            f"Chunk pages {chunk_start_page}-{chunk_end_page}: "
                            f"Applied H{level} '{title.text[:30]}...' from page {title_page}"
                        )
            
            # Post-process: Check if the chunk itself contains title-like content that wasn't detected
            # BUT only if we're not in strict custom-only mode
            if not (self.custom_config and not self.custom_config.use_auto_detection):
                chunk_text = chunk_data.get('chunk_text', '')
                
                # Look for "Unité X" patterns directly in chunk text - more aggressive approach
                # Check for titles that start the chunk (first line with content)
                first_lines = [line.strip() for line in chunk_text.split('\n')[:5] if line.strip()]
                
                for line in first_lines:
                    # Look for Unité patterns at the beginning of chunks
                    unite_match = re.match(r'unité\s+(\d+)[:\s]*([^\n]{0,80})', line, re.IGNORECASE)
                    if unite_match:
                        full_unite_title = unite_match.group(0).strip()
                        
                        # Always override H3 if we find a direct Unité match in the chunk
                        # This handles cases where chunk boundaries cut titles
                        if len(full_unite_title) < 120:  # Reasonable title length
                            # Check if this is different from current H3 or if we should force override
                            should_override = (
                                not current_titles[3] or  # No H3 yet
                                full_unite_title not in (current_titles[3] or '') or  # Different title
                                (current_titles[3] and len(current_titles[3]) > len(full_unite_title) + 20)  # Current title too long/generic
                            )
                            
                            if should_override:
                                current_titles[3] = full_unite_title
                                logger.debug(f"Chunk {i}: Override H3 with chunk-specific Unité title: '{full_unite_title}'")
                            break
                    
                    # Look for Module patterns
                    module_match = re.match(r'module\s+(\d+)[:\s]*([^\n]{0,100})', line, re.IGNORECASE)
                    if module_match:
                        full_module_title = module_match.group(0).strip()
                        
                        if len(full_module_title) < 150 and not current_titles[2]:
                            logger.info(f"Chunk {i}: POST-PROCESS applying Module from chunk text: '{full_module_title}'")
                            current_titles[2] = full_module_title
                            logger.debug(f"Chunk {i}: Found Module title in chunk text: '{full_module_title}'")
                            break
            
            # Set the hierarchy
            hierarchy.h1_title = current_titles[1]
            hierarchy.h2_title = current_titles[2]
            hierarchy.h3_title = current_titles[3]
            hierarchy.h4_title = current_titles[4]
            
            # DEBUG: Log si H2 est None mais il devrait y avoir Module 1
            if current_titles[2] is None and i == 0:
                logger.error(f"*** BUG CHUNK 0: H2 is None! Final current_titles: {current_titles}")
            
            logger.info(f"Chunk {i} final hierarchy: H1={current_titles[1]}, H2={current_titles[2]}, H3={current_titles[3]}, H4={current_titles[4]}")
            
            chunk_hierarchies.append(hierarchy)
            
        return chunk_hierarchies
    
    def _build_hierarchy_legacy(
        self,
        chunks: List[str],
        title_candidates: List[TitleCandidate],
        full_text: str,
        pages_data: List[Dict[str, Any]]
    ) -> List[TitleHierarchy]:
        """Legacy method for building hierarchy without chunk page info."""
        chunk_hierarchies = []
        
        # Use filtered titles for hierarchy building
        title_candidates = title_candidates
        
        # Create position mapping for chunks in full text with improved search
        chunk_positions = []
        search_start = 0
        
        for i, chunk in enumerate(chunks):
            # Try multiple approaches to find chunk position
            chunk_start = full_text.find(chunk, search_start)
            
            if chunk_start == -1:
                # Try searching for first line of chunk
                chunk_lines = chunk.split('\n')
                if chunk_lines:
                    first_line = chunk_lines[0].strip()
                    if first_line:
                        line_pos = full_text.find(first_line, search_start)
                        if line_pos != -1:
                            chunk_start = line_pos
                
            if chunk_start == -1:
                # Fallback: search from beginning  
                chunk_start = full_text.find(chunk)
                
            if chunk_start == -1:
                # Last resort: estimate position based on chunk order
                estimated_pos = (len(full_text) // len(chunks)) * i
                chunk_start = estimated_pos
            
            chunk_end = chunk_start + len(chunk) if chunk_start != -1 else chunk_start + 100
            chunk_positions.append((chunk_start, chunk_end))
            search_start = max(chunk_end, search_start + 1)
        
        # Create position mapping for titles using page information for accuracy
        title_positions = []
        
        # Build cumulative character positions for pages
        page_char_positions = {}
        cumulative_chars = 0
        
        for page_data in pages_data:
            page_num = page_data['page_number']
            page_text = page_data.get('text', '')
            page_char_positions[page_num] = cumulative_chars
            cumulative_chars += len(page_text) + 20  # Add buffer for page separators
        
        for title in title_candidates:
            title_start = -1
            
            # Use page number information for more accurate positioning
            if hasattr(title, 'page_number') and title.page_number:
                page_start_pos = page_char_positions.get(title.page_number, 0)
                
                # Search for title within the specific page text
                page_data = next((p for p in pages_data if p['page_number'] == title.page_number), None)
                if page_data:
                    page_text = page_data.get('text', '')
                    local_pos = page_text.find(title.text.strip())
                    
                    if local_pos != -1:
                        title_start = page_start_pos + local_pos
                    else:
                        # Fallback: search for partial match in page
                        title_words = title.text.strip().split()
                        if title_words:
                            first_words = ' '.join(title_words[:3])  # First 3 words
                            local_pos = page_text.find(first_words)
                            if local_pos != -1:
                                title_start = page_start_pos + local_pos
            
            # Fallback to original method if page-based search failed
            if title_start == -1:
                # For duplicates, search within a page range rather than the entire document
                if hasattr(title, 'page_number') and title.page_number:
                    # Calculate search range around the title's page
                    page_start_pos = page_char_positions.get(title.page_number, 0)
                    next_page_pos = page_char_positions.get(title.page_number + 1, len(full_text))
                    
                    # Search for title within this page range
                    page_text_section = full_text[page_start_pos:next_page_pos]
                    local_pos = page_text_section.find(title.text.strip())
                    
                    if local_pos != -1:
                        title_start = page_start_pos + local_pos
                    else:
                        # Try with first few words in the page range
                        title_words = title.text.strip().split()
                        if len(title_words) >= 2:
                            first_words = ' '.join(title_words[:2])
                            local_pos = page_text_section.find(first_words)
                            if local_pos != -1:
                                title_start = page_start_pos + local_pos
                
                # Only use global search as absolute last resort
                if title_start == -1:
                    title_start = full_text.find(title.text.strip())
                    
                    # If still not found, search for first few words globally
                    if title_start == -1:
                        title_words = title.text.strip().split()
                        if len(title_words) >= 2:
                            first_words = ' '.join(title_words[:2])
                            title_start = full_text.find(first_words)
            
            # Last resort: estimate based on page number
            if title_start == -1 and hasattr(title, 'page_number') and title.page_number:
                title_start = page_char_positions.get(title.page_number, 0)
            
            title_positions.append((title_start, title))
            
            # Debug information
            if title_start != -1:
                logger.debug(f"Title '{title.text[:30]}...' positioned at char {title_start} (page {getattr(title, 'page_number', 'unknown')})")
            else:
                logger.warning(f"Could not position title '{title.text[:30]}...' (page {getattr(title, 'page_number', 'unknown')})")
        
        # Sort titles by position, handling -1 positions
        title_positions.sort(key=lambda x: x[0] if x[0] != -1 else float('inf'))
        
        # Build hierarchy for each chunk using page-based logic
        for i, (chunk_start, chunk_end) in enumerate(chunk_positions):
            hierarchy = TitleHierarchy()
            
            # Estimate chunk page number based on its position
            chunk_page = self._estimate_chunk_page(chunk_start, pages_data, full_text)
            
            # Find the most recent titles before or on this chunk's page for each level
            # Following the rule: titles persist until a new title of the same level is encountered
            current_titles = {1: None, 2: None, 3: None, 4: None}
            
            # Sort titles by page number first, then by position within page
            sorted_titles = sorted(
                [(title.page_number if hasattr(title, 'page_number') else 1, title) for title in title_candidates],
                key=lambda x: x[0]
            )
            
            for title_page, title in sorted_titles:
                if title_page <= chunk_page:
                    # This title comes before or on the chunk's page
                    level = title.level
                    current_titles[level] = title.text
                    
                    # Clear only LOWER level titles when a higher or equal level title is found
                    # H1 clears H2, H3, H4
                    # H2 clears H3, H4 (but keeps H1)
                    # H3 clears H4 (but keeps H1, H2)
                    # H4 doesn't clear anything (keeps H1, H2, H3)
                    for lower_level in range(level + 1, 5):
                        current_titles[lower_level] = None
                else:
                    # We've passed the chunk's page
                    break
            
            # Set the hierarchy
            hierarchy.h1_title = current_titles[1]
            hierarchy.h2_title = current_titles[2]
            hierarchy.h3_title = current_titles[3]
            hierarchy.h4_title = current_titles[4]
            
            chunk_hierarchies.append(hierarchy)
        
        return chunk_hierarchies

    def _estimate_chunk_page(self, chunk_start: int, pages_data: List[Dict[str, Any]], full_text: str) -> int:
        """
        Estimate the page number for a chunk based on its character position.
        
        Args:
            chunk_start: Starting character position of the chunk
            pages_data: Page-by-page data
            full_text: Complete document text
            
        Returns:
            Estimated page number (1-based)
        """
        if chunk_start == -1:
            return 1
            
        # Build cumulative character positions for pages
        cumulative_chars = 0
        
        for page_data in pages_data:
            page_num = page_data['page_number']
            page_text = page_data.get('text', '')
            
            # Check if chunk starts within this page
            if chunk_start <= cumulative_chars + len(page_text):
                return page_num
                
            cumulative_chars += len(page_text) + 20  # Add buffer for page separators
        
        # Fallback: return last page
        return len(pages_data) if pages_data else 1


# Convenience function
def detect_document_titles(
    text: str, 
    pages_data: List[Dict[str, Any]], 
    custom_config=None
) -> List[TitleCandidate]:
    """
    Detect titles in document text.
    
    Args:
        text: Full document text
        pages_data: Page-by-page data
        custom_config: Optional custom title structure configuration
        
    Returns:
        List of detected title candidates
    """
    detector = TitleDetector(custom_config=custom_config)
    return detector.detect_titles_in_text(text, pages_data)


def build_chunk_title_hierarchies(
    chunks: List[str], 
    text: str, 
    pages_data: List[Dict[str, Any]],
    custom_config=None,
    chunks_with_pages: List[Dict[str, Any]] = None
) -> List[TitleHierarchy]:
    """
    Build title hierarchies for document chunks.
    
    Args:
        chunks: List of text chunks
        text: Full document text
        pages_data: Page-by-page data
        custom_config: Optional custom title structure configuration
        
    Returns:
        List of title hierarchies for each chunk
    """
    detector = TitleDetector(custom_config=custom_config)
    title_candidates = detector.detect_titles_in_text(text, pages_data)
    return detector.build_title_hierarchy_for_chunks(chunks, title_candidates, text, pages_data, chunks_with_pages)