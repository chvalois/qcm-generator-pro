"""
Chunk Variety Validator Service

Validates that questions generated from the same chunk exhibit sufficient variety
in terms of cognitive levels, question aspects, and linguistic patterns.
"""

import logging
import re
from typing import Dict, List, Set, Tuple
from difflib import SequenceMatcher

from src.models.schemas import QuestionCreate

logger = logging.getLogger(__name__)


class ChunkVarietyMetrics:
    """Metrics for measuring question variety within a chunk."""
    
    def __init__(self):
        """Initialize variety metrics."""
        self.cognitive_levels = {
            "remember": ["identifier", "nommer", "lister", "définir", "reconnaître"],
            "understand": ["expliquer", "décrire", "résumer", "interpréter", "classer"],
            "apply": ["utiliser", "appliquer", "démontrer", "implémenter", "exécuter"],
            "analyze": ["analyser", "comparer", "contraster", "examiner", "différencier"],
            "evaluate": ["évaluer", "critiquer", "justifier", "recommander", "choisir"],
            "create": ["créer", "concevoir", "développer", "construire", "planifier"]
        }
        
        self.question_starters = {
            "what": ["qu'est-ce que", "que", "quoi", "quel", "quelle", "quels", "quelles"],
            "how": ["comment", "de quelle manière", "par quel moyen"],
            "why": ["pourquoi", "pour quelle raison", "dans quel but"],
            "when": ["quand", "à quel moment", "dans quelles circonstances"],
            "where": ["où", "dans quel endroit", "à quel endroit"],
            "which": ["lequel", "laquelle", "lesquels", "lesquelles", "parmi"]
        }
        
    def calculate_cognitive_diversity(self, questions: List[QuestionCreate]) -> Dict:
        """Calculate cognitive level diversity score."""
        if not questions:
            return {"score": 0.0, "distribution": {}, "analysis": "No questions provided"}
        
        level_counts = {level: 0 for level in self.cognitive_levels.keys()}
        question_levels = []
        
        for question in questions:
            detected_level = self._detect_cognitive_level(question.question_text)
            level_counts[detected_level] += 1
            question_levels.append(detected_level)
        
        # Calculate diversity score (higher when levels are distributed)
        total_questions = len(questions)
        used_levels = sum(1 for count in level_counts.values() if count > 0)
        max_possible_levels = len(self.cognitive_levels)
        
        # Diversity score based on level distribution
        diversity_score = used_levels / max_possible_levels
        
        # Bonus for even distribution
        if used_levels > 1:
            expected_per_level = total_questions / used_levels
            evenness_penalty = sum(abs(count - expected_per_level) for count in level_counts.values() if count > 0)
            evenness_score = max(0, 1 - (evenness_penalty / (total_questions * 2)))
            diversity_score = (diversity_score + evenness_score) / 2
        
        return {
            "score": round(diversity_score, 3),
            "distribution": {k: v for k, v in level_counts.items() if v > 0},
            "question_levels": question_levels,
            "used_levels": used_levels,
            "analysis": self._analyze_cognitive_distribution(level_counts, total_questions)
        }
    
    def calculate_linguistic_diversity(self, questions: List[QuestionCreate]) -> Dict:
        """Calculate linguistic pattern diversity."""
        if not questions:
            return {"score": 0.0, "patterns": {}, "analysis": "No questions provided"}
        
        starter_counts = {starter_type: 0 for starter_type in self.question_starters.keys()}
        question_starters_detected = []
        
        for question in questions:
            detected_starter = self._detect_question_starter(question.question_text)
            starter_counts[detected_starter] += 1
            question_starters_detected.append(detected_starter)
        
        # Calculate diversity
        total_questions = len(questions)
        used_starters = sum(1 for count in starter_counts.values() if count > 0)
        max_possible_starters = len(self.question_starters)
        
        diversity_score = used_starters / max_possible_starters
        
        # Check for over-repetition
        max_repetition = max(starter_counts.values())
        if max_repetition > total_questions * 0.6:  # More than 60% same pattern
            diversity_score *= 0.5
        
        return {
            "score": round(diversity_score, 3),
            "patterns": {k: v for k, v in starter_counts.items() if v > 0},
            "question_starters": question_starters_detected,
            "analysis": self._analyze_linguistic_distribution(starter_counts, total_questions)
        }
    
    def calculate_content_diversity(self, questions: List[QuestionCreate]) -> Dict:
        """Calculate content similarity diversity."""
        if len(questions) < 2:
            return {"score": 1.0, "similarities": [], "analysis": "Insufficient questions for comparison"}
        
        similarities = []
        total_comparisons = 0
        
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                q1_text = self._normalize_question_text(questions[i].question_text)
                q2_text = self._normalize_question_text(questions[j].question_text)
                
                similarity = SequenceMatcher(None, q1_text, q2_text).ratio()
                similarities.append({
                    "question_1": i,
                    "question_2": j,
                    "similarity": round(similarity, 3)
                })
                total_comparisons += 1
        
        # Calculate average similarity
        avg_similarity = sum(s["similarity"] for s in similarities) / len(similarities)
        
        # Diversity score (lower similarity = higher diversity)
        diversity_score = max(0, 1 - avg_similarity)
        
        # Penalty for very high similarities
        high_similarity_count = sum(1 for s in similarities if s["similarity"] > 0.7)
        if high_similarity_count > 0:
            penalty = min(0.5, high_similarity_count / total_comparisons)
            diversity_score *= (1 - penalty)
        
        return {
            "score": round(diversity_score, 3),
            "average_similarity": round(avg_similarity, 3),
            "similarities": similarities,
            "high_similarity_pairs": [s for s in similarities if s["similarity"] > 0.7],
            "analysis": self._analyze_content_diversity(similarities, avg_similarity)
        }
    
    def _detect_cognitive_level(self, question_text: str) -> str:
        """Detect cognitive level based on question text."""
        normalized_text = question_text.lower()
        
        level_scores = {}
        for level, keywords in self.cognitive_levels.items():
            score = sum(1 for keyword in keywords if keyword in normalized_text)
            if score > 0:
                level_scores[level] = score
        
        if level_scores:
            return max(level_scores, key=level_scores.get)
        
        # Default classification based on question structure
        if any(word in normalized_text for word in ["qu'est-ce que", "définir", "quel est"]):
            return "remember"
        elif any(word in normalized_text for word in ["comment", "expliquer", "décrire"]):
            return "understand"
        elif any(word in normalized_text for word in ["utiliser", "appliquer", "dans quel cas"]):
            return "apply"
        elif any(word in normalized_text for word in ["comparer", "différence", "analyser"]):
            return "analyze"
        elif any(word in normalized_text for word in ["meilleur", "recommander", "choisir"]):
            return "evaluate"
        else:
            return "understand"  # Default
    
    def _detect_question_starter(self, question_text: str) -> str:
        """Detect question starter pattern."""
        normalized_text = question_text.lower().strip()
        
        for starter_type, patterns in self.question_starters.items():
            for pattern in patterns:
                if normalized_text.startswith(pattern):
                    return starter_type
        
        return "what"  # Default
    
    def _normalize_question_text(self, text: str) -> str:
        """Normalize question text for comparison."""
        # Remove punctuation, extra spaces, convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def _analyze_cognitive_distribution(self, level_counts: Dict, total_questions: int) -> str:
        """Analyze cognitive level distribution."""
        used_levels = [level for level, count in level_counts.items() if count > 0]
        
        if len(used_levels) == 1:
            return f"Questions concentrées sur le niveau cognitif '{used_levels[0]}'. Considérez diversifier avec d'autres niveaux."
        elif len(used_levels) >= 4:
            return f"Excellente diversité cognitive avec {len(used_levels)} niveaux différents."
        else:
            return f"Diversité cognitive modérée avec {len(used_levels)} niveaux. Peut être améliorée."
    
    def _analyze_linguistic_distribution(self, starter_counts: Dict, total_questions: int) -> str:
        """Analyze linguistic pattern distribution."""
        used_patterns = [pattern for pattern, count in starter_counts.items() if count > 0]
        max_count = max(starter_counts.values())
        
        if len(used_patterns) == 1:
            return "Questions utilisent toutes le même type de formulation. Diversifiez les structures."
        elif max_count > total_questions * 0.6:
            dominant_pattern = max(starter_counts, key=starter_counts.get)
            return f"Sur-utilisation du pattern '{dominant_pattern}'. Équilibrez les formulations."
        else:
            return f"Bonne diversité linguistique avec {len(used_patterns)} patterns différents."
    
    def _analyze_content_diversity(self, similarities: List[Dict], avg_similarity: float) -> str:
        """Analyze content diversity."""
        if avg_similarity > 0.7:
            return "Questions très similaires. Risque de répétition élevé."
        elif avg_similarity > 0.5:
            return "Similarité modérée. Quelques questions peuvent être trop proches."
        elif avg_similarity > 0.3:
            return "Bonne diversité de contenu avec similarité acceptable."
        else:
            return "Excellente diversité de contenu."


class ChunkVarietyValidator:
    """
    Main validator for chunk-based question variety.
    """
    
    def __init__(self, min_diversity_score: float = 0.6):
        """
        Initialize variety validator.
        
        Args:
            min_diversity_score: Minimum acceptable diversity score (0.0-1.0)
        """
        self.min_diversity_score = min_diversity_score
        self.metrics = ChunkVarietyMetrics()
    
    def validate_chunk_questions(
        self, 
        questions: List[QuestionCreate], 
        chunk_id: str
    ) -> Dict:
        """
        Validate variety of questions generated from a chunk.
        
        Args:
            questions: List of questions from the same chunk
            chunk_id: ID of the source chunk
            
        Returns:
            Validation results with scores and recommendations
        """
        logger.debug(f"Validating variety for {len(questions)} questions from chunk {chunk_id}")
        
        if len(questions) < 2:
            return {
                "valid": True,
                "overall_score": 1.0,
                "chunk_id": chunk_id,
                "analysis": "Single question, no variety validation needed"
            }
        
        # Calculate diversity metrics
        cognitive_diversity = self.metrics.calculate_cognitive_diversity(questions)
        linguistic_diversity = self.metrics.calculate_linguistic_diversity(questions)
        content_diversity = self.metrics.calculate_content_diversity(questions)
        
        # Calculate overall score (weighted average)
        overall_score = (
            cognitive_diversity["score"] * 0.4 +  # 40% weight
            linguistic_diversity["score"] * 0.3 +  # 30% weight
            content_diversity["score"] * 0.3       # 30% weight
        )
        
        # Determine if validation passes
        is_valid = overall_score >= self.min_diversity_score
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            cognitive_diversity, linguistic_diversity, content_diversity, overall_score
        )
        
        result = {
            "valid": is_valid,
            "overall_score": round(overall_score, 3),
            "chunk_id": chunk_id,
            "question_count": len(questions),
            "cognitive_diversity": cognitive_diversity,
            "linguistic_diversity": linguistic_diversity,
            "content_diversity": content_diversity,
            "recommendations": recommendations,
            "analysis": f"Variété {'acceptable' if is_valid else 'insuffisante'} (score: {overall_score:.3f})"
        }
        
        if not is_valid:
            logger.warning(f"Chunk {chunk_id} questions failed variety validation (score: {overall_score:.3f})")
        else:
            logger.debug(f"Chunk {chunk_id} questions passed variety validation (score: {overall_score:.3f})")
        
        return result
    
    def validate_multiple_chunks(
        self, 
        chunk_questions: Dict[str, List[QuestionCreate]]
    ) -> Dict:
        """
        Validate variety across multiple chunks.
        
        Args:
            chunk_questions: Dict mapping chunk_id to list of questions
            
        Returns:
            Overall validation results
        """
        results = {}
        total_score = 0.0
        valid_chunks = 0
        
        for chunk_id, questions in chunk_questions.items():
            chunk_result = self.validate_chunk_questions(questions, chunk_id)
            results[chunk_id] = chunk_result
            
            total_score += chunk_result["overall_score"]
            if chunk_result["valid"]:
                valid_chunks += 1
        
        overall_analysis = {
            "total_chunks": len(chunk_questions),
            "valid_chunks": valid_chunks,
            "invalid_chunks": len(chunk_questions) - valid_chunks,
            "average_score": round(total_score / len(chunk_questions), 3) if chunk_questions else 0.0,
            "overall_valid": valid_chunks == len(chunk_questions),
            "chunk_results": results
        }
        
        return overall_analysis
    
    def _generate_recommendations(
        self, 
        cognitive_div: Dict, 
        linguistic_div: Dict, 
        content_div: Dict,
        overall_score: float
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Cognitive diversity recommendations
        if cognitive_div["score"] < 0.5:
            recommendations.append(
                f"Diversifier les niveaux cognitifs (actuellement {cognitive_div['used_levels']} niveaux utilisés)"
            )
        
        # Linguistic diversity recommendations
        if linguistic_div["score"] < 0.4:
            recommendations.append(
                "Varier les structures de questions (Comment, Pourquoi, Quel, etc.)"
            )
        
        # Content diversity recommendations
        if content_div["score"] < 0.5:
            high_sim_count = len(content_div.get("high_similarity_pairs", []))
            if high_sim_count > 0:
                recommendations.append(
                    f"Réduire la similarité entre questions ({high_sim_count} paires très similaires détectées)"
                )
        
        # Overall recommendations
        if overall_score < self.min_diversity_score:
            recommendations.append(
                "Augmenter la créativité et l'originalité des questions pour ce chunk"
            )
        
        if not recommendations:
            recommendations.append("Excellente variété, continuer sur cette voie")
        
        return recommendations


# Global instance
_chunk_variety_validator: ChunkVarietyValidator | None = None


def get_chunk_variety_validator() -> ChunkVarietyValidator:
    """Get the global chunk variety validator instance."""
    global _chunk_variety_validator
    if _chunk_variety_validator is None:
        _chunk_variety_validator = ChunkVarietyValidator()
    return _chunk_variety_validator


# Convenience functions
def validate_chunk_question_variety(
    questions: List[QuestionCreate], 
    chunk_id: str
) -> Dict:
    """Validate variety of questions from a chunk."""
    validator = get_chunk_variety_validator()
    return validator.validate_chunk_questions(questions, chunk_id)


def validate_multiple_chunk_varieties(
    chunk_questions: Dict[str, List[QuestionCreate]]
) -> Dict:
    """Validate variety across multiple chunks."""
    validator = get_chunk_variety_validator()
    return validator.validate_multiple_chunks(chunk_questions)