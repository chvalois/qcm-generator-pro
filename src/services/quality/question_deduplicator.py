"""
Question Deduplication Service

Handles detection and prevention of duplicate questions using multiple similarity metrics.
Follows SRP by focusing solely on deduplication logic.
"""

import hashlib
import logging
import re
from typing import List, Set, Dict, Tuple
from difflib import SequenceMatcher

from src.models.schemas import QuestionCreate

logger = logging.getLogger(__name__)


class QuestionDeduplicator:
    """
    Service responsible for detecting and preventing duplicate questions.
    
    Uses multiple similarity metrics:
    - Exact text hash matching
    - Semantic similarity of question text
    - Answer option similarity
    - Topic similarity
    """
    
    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering questions similar (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.question_hashes: Set[str] = set()
        self.question_signatures: List[Dict] = []
        
    def is_duplicate(self, question: QuestionCreate) -> bool:
        """
        Check if question is a duplicate of previously seen questions.
        
        Args:
            question: Question to check
            
        Returns:
            True if question is considered a duplicate
        """
        # 1. Check exact hash match (fastest)
        question_hash = self._get_exact_hash(question)
        if question_hash in self.question_hashes:
            logger.debug("Exact duplicate detected via hash")
            return True
        
        # 2. Check semantic similarity (more thorough)
        if self._is_semantically_similar(question):
            logger.debug("Semantic duplicate detected")
            return True
        
        return False
    
    def add_question(self, question: QuestionCreate) -> None:
        """
        Add question to tracking system.
        
        Args:
            question: Question to track
        """
        # Add exact hash
        question_hash = self._get_exact_hash(question)
        self.question_hashes.add(question_hash)
        
        # Add semantic signature
        signature = self._create_question_signature(question)
        self.question_signatures.append(signature)
        
        logger.debug(f"Added question to deduplication tracking: {question.question_text[:50]}...")
    
    def _get_exact_hash(self, question: QuestionCreate) -> str:
        """Generate exact hash for question content."""
        # Normalize text for hashing
        normalized_text = self._normalize_text(question.question_text)
        
        # Include options if available
        options_text = ""
        if hasattr(question, 'options') and question.options:
            option_texts = []
            for opt in question.options:
                if hasattr(opt, 'text'):
                    option_texts.append(self._normalize_text(opt.text))
                elif isinstance(opt, str):
                    option_texts.append(self._normalize_text(opt))
            sorted_options = sorted(option_texts)
            options_text = "|".join(sorted_options)
        
        content = f"{normalized_text}|{options_text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_question_signature(self, question: QuestionCreate) -> Dict:
        """Create semantic signature for similarity comparison."""
        normalized_text = self._normalize_text(question.question_text)
        
        # Extract key features
        signature = {
            "text": normalized_text,
            "words": set(normalized_text.split()),
            "keywords": self._extract_keywords(normalized_text),
            "length": len(normalized_text),
            "topic": getattr(question, 'theme', ''),
            "question_type": getattr(question, 'question_type', ''),
            "options_count": len(getattr(question, 'options', []))
        }
        
        if hasattr(question, 'options') and question.options:
            option_texts = []
            signature["option_words"] = set()
            for opt in question.options:
                if hasattr(opt, 'text'):
                    text = self._normalize_text(opt.text)
                    option_texts.append(text)
                    signature["option_words"].update(text.split())
                elif isinstance(opt, str):
                    text = self._normalize_text(opt)
                    option_texts.append(text)
                    signature["option_words"].update(text.split())
            signature["options"] = option_texts
        
        return signature
    
    def _is_semantically_similar(self, question: QuestionCreate) -> bool:
        """Check if question is semantically similar to existing questions."""
        current_signature = self._create_question_signature(question)
        
        for existing_signature in self.question_signatures:
            similarity = self._calculate_similarity(current_signature, existing_signature)
            
            if similarity >= self.similarity_threshold:
                logger.debug(f"High similarity detected: {similarity:.2f}")
                return True
        
        return False
    
    def _calculate_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate comprehensive similarity between two question signatures."""
        similarities = []
        
        # 1. Text similarity using sequence matcher
        text_sim = SequenceMatcher(None, sig1["text"], sig2["text"]).ratio()
        similarities.append(("text", text_sim, 0.4))  # 40% weight
        
        # 2. Word overlap similarity
        words1, words2 = sig1["words"], sig2["words"]
        if words1 and words2:
            word_intersection = len(words1.intersection(words2))
            word_union = len(words1.union(words2))
            word_sim = word_intersection / word_union if word_union > 0 else 0
            similarities.append(("words", word_sim, 0.3))  # 30% weight
        
        # 3. Keyword similarity
        keywords1, keywords2 = sig1["keywords"], sig2["keywords"]
        if keywords1 and keywords2:
            keyword_intersection = len(keywords1.intersection(keywords2))
            keyword_union = len(keywords1.union(keywords2))
            keyword_sim = keyword_intersection / keyword_union if keyword_union > 0 else 0
            similarities.append(("keywords", keyword_sim, 0.2))  # 20% weight
        
        # 4. Options similarity (if both have options)
        if "options" in sig1 and "options" in sig2:
            options_sim = self._calculate_options_similarity(sig1["options"], sig2["options"])
            similarities.append(("options", options_sim, 0.1))  # 10% weight
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in similarities)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(sim * weight for _, sim, weight in similarities)
        return weighted_sum / total_weight
    
    def _calculate_options_similarity(self, options1: List[str], options2: List[str]) -> float:
        """Calculate similarity between option lists."""
        if not options1 or not options2:
            return 0.0
        
        # Check for exact matches
        set1, set2 = set(options1), set(options2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation that doesn't affect meaning
        text = re.sub(r'[^\w\s\?\.\!]', '', text)
        
        # Remove common question prefixes/suffixes
        prefixes = ['quel est', 'quelle est', 'quels sont', 'quelles sont', 'comment', 'pourquoi']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        return text.strip()
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract important keywords from text."""
        # Common stop words to ignore
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'est', 'sont',
            'dans', 'sur', 'avec', 'pour', 'par', 'ce', 'cette', 'ces', 'que', 'qui',
            'dont', 'où', 'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'se', 'sa', 'son'
        }
        
        words = text.split()
        # Keep words that are longer than 3 characters and not stop words
        keywords = {word for word in words 
                   if len(word) > 3 and word.lower() not in stop_words}
        
        return keywords
    
    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        return {
            "total_questions_tracked": len(self.question_signatures),
            "unique_hashes": len(self.question_hashes),
            "similarity_threshold": self.similarity_threshold
        }
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.question_hashes.clear()
        self.question_signatures.clear()
        logger.info("Question deduplicator reset")


# Global instance
_question_deduplicator: QuestionDeduplicator | None = None


def get_question_deduplicator() -> QuestionDeduplicator:
    """Get the global question deduplicator instance."""
    global _question_deduplicator
    if _question_deduplicator is None:
        _question_deduplicator = QuestionDeduplicator()
    return _question_deduplicator


def reset_deduplicator() -> None:
    """Reset the global deduplicator."""
    global _question_deduplicator
    if _question_deduplicator is not None:
        _question_deduplicator.reset()