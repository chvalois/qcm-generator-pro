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
    
    def __init__(self):
        """Initialize title detector with configurable patterns."""
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
                r'^(Module\s+\d+.*)$',
                r'^(Chapitre\s+\d+.*)$',
                r'^(Chapter\s+\d+.*)$',
                r'^(Leçon\s+\d+.*)$',
                r'^(Lesson\s+\d+.*)$',
                r'^(Unit[ée]?\s+\d+.*)$',
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
        
        # Post-process and refine candidates
        refined_candidates = self._refine_title_candidates(title_candidates)
        
        logger.info(f"Detected {len(refined_candidates)} title candidates")
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
        
        # Pattern matching analysis
        pattern_score, pattern_level = self._analyze_patterns(line)
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
        """Analyze title numbering and keyword patterns."""
        score = 0.0
        level = None
        
        # Check numbered patterns (highest confidence)
        for i, pattern in enumerate(self.title_patterns['numbered']):
            if re.match(pattern, line, re.IGNORECASE):
                score += 0.8
                level = i + 1  # H1, H2, H3, H4
                break
        
        # Check keyword patterns
        for pattern in self.title_patterns['keywords']:
            if re.match(pattern, line, re.IGNORECASE):
                score += 0.6
                if not level:  # Only set if not already set by numbering
                    level = 1 if any(word in line.lower() for word in ['module', 'chapitre', 'chapter']) else 2
                break
        
        # Check lettered patterns
        for pattern in self.title_patterns['lettered']:
            if re.match(pattern, line, re.IGNORECASE):
                score += 0.4
                if not level:
                    level = 3 if pattern == self.title_patterns['lettered'][0] else 4
                break
        
        # Check roman numerals
        for pattern in self.title_patterns['roman']:
            if re.match(pattern, line, re.IGNORECASE):
                score += 0.3
                if not level:
                    level = 2 if pattern == self.title_patterns['roman'][0] else 3
                break
        
        return score, level
    
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
        
        # Remove duplicates and very similar titles
        refined = []
        seen_texts = set()
        
        for candidate in candidates:
            # Normalize text for comparison
            normalized = re.sub(r'\s+', ' ', candidate.text.lower().strip())
            
            # Skip if we've seen very similar text
            if normalized in seen_texts:
                continue
            
            # Check for substring duplicates
            is_duplicate = False
            for seen in seen_texts:
                if (normalized in seen and len(normalized) < len(seen) * 0.8) or \
                   (seen in normalized and len(seen) < len(normalized) * 0.8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(normalized)
                refined.append(candidate)
        
        # Limit to reasonable number of titles per level
        level_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        final_candidates = []
        
        for candidate in refined:
            level = candidate.level
            if level_counts[level] < 20:  # Max 20 titles per level
                level_counts[level] += 1
                final_candidates.append(candidate)
        
        return final_candidates
    
    def build_title_hierarchy_for_chunks(
        self, 
        chunks: List[str], 
        title_candidates: List[TitleCandidate],
        full_text: str
    ) -> List[TitleHierarchy]:
        """
        Build title hierarchy for each chunk.
        
        Args:
            chunks: List of text chunks
            title_candidates: Detected title candidates
            full_text: Complete document text
            
        Returns:
            List of TitleHierarchy objects, one per chunk
        """
        chunk_hierarchies = []
        
        # Create position mapping for chunks in full text
        chunk_positions = []
        search_start = 0
        
        for chunk in chunks:
            # Find chunk position in full text
            chunk_start = full_text.find(chunk, search_start)
            if chunk_start == -1:
                # Fallback: search from beginning
                chunk_start = full_text.find(chunk)
            
            chunk_end = chunk_start + len(chunk) if chunk_start != -1 else 0
            chunk_positions.append((chunk_start, chunk_end))
            search_start = chunk_end
        
        # Create position mapping for titles
        title_positions = []
        for title in title_candidates:
            title_start = full_text.find(title.text)
            title_positions.append((title_start, title))
        
        # Sort titles by position
        title_positions.sort(key=lambda x: x[0])
        
        # Build hierarchy for each chunk
        for chunk_start, chunk_end in chunk_positions:
            hierarchy = TitleHierarchy()
            
            # Find the most recent titles before this chunk
            current_titles = {1: None, 2: None, 3: None, 4: None}
            
            for title_pos, title in title_positions:
                if title_pos != -1 and title_pos <= chunk_start:
                    # This title comes before the chunk
                    current_titles[title.level] = title.text
                    
                    # Clear lower level titles when a higher level title is found
                    for level in range(title.level + 1, 5):
                        current_titles[level] = None
                else:
                    # We've passed the chunk position
                    break
            
            # Set the hierarchy
            hierarchy.h1_title = current_titles[1]
            hierarchy.h2_title = current_titles[2]
            hierarchy.h3_title = current_titles[3]
            hierarchy.h4_title = current_titles[4]
            
            chunk_hierarchies.append(hierarchy)
        
        return chunk_hierarchies


# Convenience function
def detect_document_titles(text: str, pages_data: List[Dict[str, Any]]) -> List[TitleCandidate]:
    """
    Detect titles in document text.
    
    Args:
        text: Full document text
        pages_data: Page-by-page data
        
    Returns:
        List of detected title candidates
    """
    detector = TitleDetector()
    return detector.detect_titles_in_text(text, pages_data)


def build_chunk_title_hierarchies(
    chunks: List[str], 
    text: str, 
    pages_data: List[Dict[str, Any]]
) -> List[TitleHierarchy]:
    """
    Build title hierarchies for document chunks.
    
    Args:
        chunks: List of text chunks
        text: Full document text
        pages_data: Page-by-page data
        
    Returns:
        List of title hierarchies for each chunk
    """
    detector = TitleDetector()
    title_candidates = detector.detect_titles_in_text(text, pages_data)
    return detector.build_title_hierarchy_for_chunks(chunks, title_candidates, text)