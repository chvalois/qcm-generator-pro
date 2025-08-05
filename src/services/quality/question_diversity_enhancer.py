"""
Question Diversity Enhancer Service

Provides advanced prompt engineering and context variation techniques
to enhance question diversity and prevent repetitive content.
"""

import logging
import random
from typing import Dict, List, Optional

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import GenerationConfig, QuestionContext

logger = logging.getLogger(__name__)


class QuestionDiversityEnhancer:
    """
    Service for enhancing question diversity through advanced prompt engineering
    and context variation techniques.
    """
    
    def __init__(self):
        """Initialize diversity enhancer with variation strategies."""
        self.question_angles = {
            'theoretical': [
                "Analysez les concepts théoriques de",
                "Expliquez les principes fondamentaux de",
                "Décrivez la théorie derrière",
                "Quels sont les fondements conceptuels de"
            ],
            'practical': [
                "Dans un contexte professionnel, comment",
                "Lors de l'implémentation de",
                "En pratique, comment utilise-t-on",
                "Dans quelles situations applique-t-on"
            ],
            'comparative': [
                "Comparez les différentes approches de",
                "Quelle est la différence entre",
                "Contrastez les méthodes de",
                "Évaluez les alternatives pour"
            ],
            'problem_solving': [
                "Comment résoudre un problème de",
                "Quelle stratégie adopter pour",
                "Face à un défi de",
                "Pour optimiser"
            ],
            'architectural': [
                "Analysez l'architecture de",
                "Décrivez l'organisation structurelle de",
                "Comment est conçu",
                "Quelle est la structure de"
            ]
        }
        
        self.context_modifiers = {
            'enterprise': "dans un contexte d'entreprise",
            'technical': "d'un point de vue technique",
            'strategic': "du point de vue stratégique",
            'operational': "au niveau opérationnel",
            'integration': "dans une perspective d'intégration",
            'security': "en considérant la sécurité",
            'performance': "en termes de performance",
            'scalability': "concernant la scalabilité"
        }
        
        self.difficulty_enhancers = {
            Difficulty.EASY: [
                "Identifiez",
                "Listez",
                "Nommez",
                "Reconnaissez"
            ],
            Difficulty.MEDIUM: [
                "Analysez",
                "Expliquez",
                "Décrivez",
                "Comparez"
            ],
            Difficulty.HARD: [
                "Évaluez de manière critique",
                "Synthétisez",
                "Justifiez votre choix",
                "Optimisez la stratégie"
            ]
        }
        
    def enhance_prompt_diversity(
        self,
        base_prompt: str,
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty,
        diversity_level: int = 0
    ) -> str:
        """
        Enhance prompt diversity by applying various transformation techniques.
        
        Args:
            base_prompt: Original prompt
            context: Question context
            config: Generation configuration
            question_type: Type of question
            difficulty: Question difficulty
            diversity_level: Level of diversity enhancement (0-3)
            
        Returns:
            Enhanced prompt with improved diversity
        """
        enhanced_prompt = base_prompt
        
        if diversity_level == 0:
            return enhanced_prompt
        
        # Level 1: Add question angle variation
        if diversity_level >= 1:
            enhanced_prompt = self._add_question_angle(enhanced_prompt, context.topic)
        
        # Level 2: Add context modifier
        if diversity_level >= 2:
            enhanced_prompt = self._add_context_modifier(enhanced_prompt)
        
        # Level 3: Add difficulty-specific enhancement
        if diversity_level >= 3:
            enhanced_prompt = self._add_difficulty_enhancement(enhanced_prompt, difficulty)
        
        # Add specific diversity instructions
        diversity_instruction = self._get_diversity_instruction(diversity_level)
        enhanced_prompt = f"{enhanced_prompt}\n\n{diversity_instruction}"
        
        logger.debug(f"Enhanced prompt with diversity level {diversity_level}")
        return enhanced_prompt
    
    def _add_question_angle(self, prompt: str, topic: str) -> str:
        """Add a specific questioning angle to vary approach."""
        angle_type = random.choice(list(self.question_angles.keys()))
        angle_prompt = random.choice(self.question_angles[angle_type])
        
        # Insert angle-specific instruction
        angle_instruction = f"\nApproche recommandée: {angle_prompt} {topic}."
        return prompt + angle_instruction
    
    def _add_context_modifier(self, prompt: str) -> str:
        """Add context modifier to vary perspective."""
        modifier_key = random.choice(list(self.context_modifiers.keys()))
        modifier_text = self.context_modifiers[modifier_key]
        
        context_instruction = f"\nPerspective: Considérez cette question {modifier_text}."
        return prompt + context_instruction
    
    def _add_difficulty_enhancement(self, prompt: str, difficulty: Difficulty) -> str:
        """Add difficulty-specific enhancement."""
        enhancers = self.difficulty_enhancers.get(difficulty, [])
        if enhancers:
            enhancer = random.choice(enhancers)
            difficulty_instruction = f"\nStyle de question: {enhancer} de manière {difficulty.value}."
            return prompt + difficulty_instruction
        
        return prompt
    
    def _get_diversity_instruction(self, diversity_level: int) -> str:
        """Get diversity-specific instruction for the LLM."""
        instructions = {
            1: (
                "IMPORTANT: Créez une question unique qui aborde le sujet "
                "sous un angle différent des questions standard."
            ),
            2: (
                "IMPORTANT: Formulez une question originale qui explore "
                "des aspects moins évidents du sujet. Évitez les formulations "
                "trop communes ou génériques."
            ),
            3: (
                "IMPORTANT: Générez une question innovante et spécifique "
                "qui se distingue clairement des approches conventionnelles. "
                "Privilégiez les nuances et les détails techniques."
            )
        }
        
        return instructions.get(diversity_level, instructions[1])
    
    def create_topic_variations(self, base_topic: str, count: int = 5) -> List[str]:
        """
        Create semantic variations of a topic to enhance context diversity.
        
        Args:
            base_topic: Original topic
            count: Number of variations to create
            
        Returns:
            List of topic variations
        """
        variations = [base_topic]  # Include original
        
        # Topic enhancement strategies
        strategies = [
            self._add_technical_focus,
            self._add_practical_focus,
            self._add_architectural_focus,
            self._add_implementation_focus,
            self._add_strategic_focus
        ]
        
        for i in range(min(count - 1, len(strategies))):
            strategy = strategies[i]
            variation = strategy(base_topic)
            if variation and variation not in variations:
                variations.append(variation)
        
        # If we need more variations, use generic enhancers
        if len(variations) < count:
            generic_enhancers = [
                "concepts avancés",
                "aspects pratiques",
                "fonctionnalités clés",
                "composants principaux",
                "méthodes d'utilisation"
            ]
            
            for enhancer in generic_enhancers:
                if len(variations) >= count:
                    break
                variation = f"{base_topic} {enhancer}"
                if variation not in variations:
                    variations.append(variation)
        
        return variations[:count]
    
    def _add_technical_focus(self, topic: str) -> str:
        """Add technical focus to topic."""
        technical_terms = ["architecture", "implémentation", "configuration", "optimisation"]
        term = random.choice(technical_terms)
        return f"{topic} - {term} technique"
    
    def _add_practical_focus(self, topic: str) -> str:
        """Add practical focus to topic."""
        practical_terms = ["cas d'usage", "applications réelles", "mise en pratique", "exemples concrets"]
        term = random.choice(practical_terms)
        return f"{topic} - {term}"
    
    def _add_architectural_focus(self, topic: str) -> str:
        """Add architectural focus to topic."""
        arch_terms = ["structure", "composants", "organisation", "conception"]
        term = random.choice(arch_terms)
        return f"{topic} - {term} architecturale"
    
    def _add_implementation_focus(self, topic: str) -> str:
        """Add implementation focus to topic."""
        impl_terms = ["déploiement", "mise en œuvre", "intégration", "configuration"]
        term = random.choice(impl_terms)
        return f"{topic} - {term}"
    
    def _add_strategic_focus(self, topic: str) -> str:
        """Add strategic focus to topic."""
        strategic_terms = ["stratégie", "planification", "adoption", "évolution"]
        term = random.choice(strategic_terms)
        return f"{topic} - {term} stratégique"
    
    def get_diversity_metrics(self) -> Dict:
        """Get metrics about diversity enhancement usage."""
        return {
            "available_angles": len(self.question_angles),
            "available_modifiers": len(self.context_modifiers),
            "total_combinations": len(self.question_angles) * len(self.context_modifiers)
        }


# Global instance
_diversity_enhancer: QuestionDiversityEnhancer | None = None


def get_diversity_enhancer() -> QuestionDiversityEnhancer:
    """Get the global diversity enhancer instance."""
    global _diversity_enhancer
    if _diversity_enhancer is None:
        _diversity_enhancer = QuestionDiversityEnhancer()
    return _diversity_enhancer