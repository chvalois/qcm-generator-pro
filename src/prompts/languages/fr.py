"""
French language template for QCM generation prompts.

This module implements French-specific prompts for generating
QCM questions, validating them, and extracting themes.
"""

from typing import Dict

from ...models.enums import Difficulty, QuestionType
from ...models.schemas import QuestionContext, GenerationConfig
from .base import LanguageTemplate


class FrenchTemplate(LanguageTemplate):
    """French language implementation of QCM generation prompts."""
    
    @property
    def language_code(self) -> str:
        return "fr"
    
    @property
    def language_name(self) -> str:
        return "Français"
    
    def get_question_type_descriptions(self) -> Dict[QuestionType, str]:
        return {
            QuestionType.MULTIPLE_CHOICE: "choix multiple (une seule bonne réponse)",
            QuestionType.MULTIPLE_SELECTION: "sélection multiple (plusieurs bonnes réponses possibles)"
        }
    
    def get_difficulty_descriptions(self) -> Dict[Difficulty, str]:
        return {
            Difficulty.EASY: "facile (concepts de base)",
            Difficulty.MEDIUM: "moyen (application des concepts)",
            Difficulty.HARD: "difficile (analyse et synthèse)"
        }
    
    def get_question_generation_prompt(
        self,
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty
    ) -> str:
        """Generate the main French prompt for QCM question generation."""
        type_descriptions = self.get_question_type_descriptions()
        difficulty_descriptions = self.get_difficulty_descriptions()
        
        # Safely extract enum values
        question_type_value = question_type.value if hasattr(question_type, 'value') else question_type
        difficulty_value = difficulty.value if hasattr(difficulty, 'value') else difficulty
        
        options_count = self.get_options_count_range(question_type)
        correct_count = self.get_correct_answers_count(question_type)
        
        if question_type == QuestionType.MULTIPLE_CHOICE:
            correct_desc = f"{correct_count} seule"
        else:
            correct_desc = f"{correct_count}"
        
        prompt = f"""Générez une question QCM basée sur le contexte suivant.

CONTEXTE:
{context.context_text}

THÈME: {context.topic}

EXIGENCES:
- Type: {type_descriptions[question_type]}
- Difficulté: {difficulty_descriptions[difficulty]}
- Langue: Français
- {options_count} options de réponse
- {correct_desc} bonne(s) réponse(s)
- Question claire et précise
- Options de réponse plausibles
- Explication détaillée de la réponse

RÉPONDEZ UNIQUEMENT au format JSON suivant:
{{
  "question_text": "Texte de la question",
  "question_type": "{question_type_value}",
  "difficulty": "{difficulty_value}",
  "language": "fr",
  "theme": "{context.topic}",
  "options": [
    "Option 1",
    "Option 2", 
    "Option 3",
    "Option 4"
  ],
  "correct_answers": [0, 2],
  "explanation": "Explication détaillée de pourquoi ces réponses sont correctes"
}}"""

        return prompt
    
    def get_validation_prompt(self, question_data: Dict) -> str:
        """Generate French prompt for question validation."""
        return f"""Validez la qualité de cette question QCM en français.

QUESTION À VALIDER:
{question_data}

CRITÈRES DE VALIDATION:
1. STRUCTURE: Format JSON correct, tous les champs requis présents
2. CONTENU: Question claire, sans ambiguïté, niveau approprié  
3. OPTIONS: Distracteurs plausibles, réponses correctes valides
4. LANGUE: Français correct, terminologie appropriée
5. PÉDAGOGIE: Question utile pour l'apprentissage
6. CLARTÉ ET PRÉCISION: Éviter les questions ambiguës ou à double négation, Utiliser un vocabulaire précis et technique approprié au niveau cible
7. LONGUEUR ÉQUILIBRÉE: Maintenir des propositions de longueur similaire pour éviter les indices, Questions concises mais complètes
8. COUVERTURE ÉQUILIBRÉE: Répartir les questions sur l'ensemble du document, Varier les types de connaissances testées
9. DIFFICULTÉ VARIÉE: Mélanger les niveaux de difficulté, Inclure des questions sur les concepts clés vs détails spécifiques
10. DISTRACTEURS COHÉRENTS: Utiliser des termes du même domaine technique, Éviter les réponses évidemment absurdes ou hors contexte
11. PIÈGES PÉDAGOGIQUES: Inclure des confusions courantes ou des concepts proches, Utiliser des éléments mentionnés dans le document mais dans un autre contexte
12. STATISTIQUES ET CHIFFRES: Maximum de 5% de questions contenant des interrogations sur des chiffres ou des stats

RÉPONDEZ au format JSON:
{{
  "is_valid": true/false,
  "score": 0-10,
  "issues": ["liste des problèmes identifiés"],
  "suggestions": ["suggestions d'amélioration"],
  "validation_details": {{
    "structure": true/false,
    "content": true/false,
    "options": true/false,
    "language": true/false,
    "pedagogy": true/false
  }}
}}"""
    
    def get_theme_extraction_prompt(self, text_content: str) -> str:
        """Generate French prompt for theme extraction from PDF content."""
        return f"""Analysez le contenu suivant et extrayez les thèmes principaux.

CONTENU À ANALYSER:
{text_content[:2000]}...

INSTRUCTIONS:
- Identifiez 3-8 thèmes principaux du document
- Pour chaque thème, fournissez des mots-clés représentatifs
- Estimez un score de confiance (0.0-1.0)
- Organisez par ordre d'importance

RÉPONDEZ au format JSON:
{{
  "themes": [
    {{
      "name": "Nom du thème",
      "keywords": ["mot-clé1", "mot-clé2", "mot-clé3"],
      "confidence": 0.95,
      "description": "Description courte du thème"
    }}
  ],
  "document_summary": "Résumé général du document",
  "language_detected": "fr",
  "extraction_confidence": 0.90
}}"""
    
    def get_system_prompt(self) -> str:
        """Get the French system prompt for the LLM."""
        return """Tu es un expert en création de questions QCM éducatives en français. 

TES COMPÉTENCES:
- Génération de questions claires et pédagogiques
- Création de distracteurs plausibles mais incorrect
- Adaptation du niveau de difficulté selon les consignes
- Respect strict du format JSON demandé
- Maîtrise parfaite du français académique

INSTRUCTIONS GÉNÉRALES:
- Génère toujours des réponses au format JSON valide
- Assure-toi que les questions sont éducatives et utiles
- Évite les pièges ou les ambiguïtés inutiles
- Utilise un vocabulaire approprié au niveau demandé
- Fournis des explications pédagogiques complètes"""