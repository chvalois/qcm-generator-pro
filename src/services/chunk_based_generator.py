"""
Chunk-Based Question Generator

Generates questions directly from document chunks with proper distribution and variety.
This approach ensures comprehensive coverage and natural diversity.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import DocumentChunk, GenerationConfig, QuestionCreate, QuestionContext
from src.services.llm_manager import get_llm_manager
from src.services.question_prompt_builder import get_question_prompt_builder
from src.services.question_parser import get_question_parser
from src.services.question_selection import get_question_selector
from src.services.document_manager import get_document_manager
from src.services.progress_tracker import update_progress, increment_progress

logger = logging.getLogger(__name__)


class ChunkBasedGenerationError(Exception):
    """Exception raised when chunk-based generation fails."""
    pass


class ChunkQuestionVarietyValidator:
    """
    Validates and ensures variety in questions generated from the same chunk.
    """
    
    def __init__(self):
        """Initialize variety validator."""
        self.question_aspects = [
            "definition",      # Qu'est-ce que...
            "function",        # À quoi sert...
            "application",     # Comment utilise-t-on...
            "comparison",      # Quelle différence...
            "advantage",       # Quel avantage...
            "implementation",  # Comment implémenter...
            "problem_solving", # Comment résoudre...
            "technical",       # Aspect technique...
        ]
        
    def generate_varied_questions_plan(
        self, 
        chunk: DocumentChunk, 
        num_questions: int,
        config: GenerationConfig
    ) -> List[Dict]:
        """
        Generate a plan for creating varied questions from a chunk.
        
        Args:
            chunk: Source chunk
            num_questions: Number of questions to generate
            config: Generation configuration
            
        Returns:
            List of question generation plans
        """
        plans = []
        
        # Ensure we have enough aspects for variety
        aspects_pool = self.question_aspects * (num_questions // len(self.question_aspects) + 1)
        selected_aspects = aspects_pool[:num_questions]
        
        for i, aspect in enumerate(selected_aspects):
            # Select question type and difficulty
            question_type = self._select_question_type_for_aspect(aspect, config)
            difficulty = self._select_difficulty_for_aspect(aspect, config, i, num_questions)
            
            plan = {
                "aspect": aspect,
                "question_type": question_type,
                "difficulty": difficulty,
                "focus": self._get_aspect_focus(aspect),
                "instruction": self._get_aspect_instruction(aspect),
                "order": i
            }
            plans.append(plan)
            
        return plans
    
    def _select_question_type_for_aspect(self, aspect: str, config: GenerationConfig) -> QuestionType:
        """Select appropriate question type for the aspect."""
        # Some aspects work better with certain question types
        aspect_preferences = {
            "definition": [QuestionType.UNIQUE_CHOICE],
            "function": [QuestionType.UNIQUE_CHOICE],
            "application": [QuestionType.MULTIPLE_SELECTION, QuestionType.UNIQUE_CHOICE],
            "comparison": [QuestionType.UNIQUE_CHOICE],
            "advantage": [QuestionType.MULTIPLE_SELECTION, QuestionType.UNIQUE_CHOICE],
            "implementation": [QuestionType.MULTIPLE_SELECTION],
            "problem_solving": [QuestionType.UNIQUE_CHOICE],
            "technical": [QuestionType.UNIQUE_CHOICE, QuestionType.MULTIPLE_SELECTION],
        }
        
        preferred_types = aspect_preferences.get(aspect, list(config.question_types.keys()))
        
        # Filter by configured types
        available_types = [qt for qt in preferred_types if qt in config.question_types]
        if not available_types:
            available_types = list(config.question_types.keys())
            
        # Select based on config probabilities among available types
        import random
        return random.choice(available_types)
    
    def _select_difficulty_for_aspect(
        self, 
        aspect: str, 
        config: GenerationConfig, 
        question_order: int, 
        total_questions: int
    ) -> Difficulty:
        """Select difficulty based on aspect and position."""
        # Some aspects are naturally more complex
        aspect_difficulty_bias = {
            "definition": [-0.1, 0.1, 0.0],      # Easier
            "function": [-0.1, 0.1, 0.0],        # Easier  
            "application": [0.0, 0.0, 0.1],      # Medium to Hard
            "comparison": [0.0, 0.1, 0.0],       # Medium
            "advantage": [0.0, 0.0, 0.1],        # Medium to Hard
            "implementation": [0.0, 0.0, 0.2],   # Harder
            "problem_solving": [0.0, 0.0, 0.2],  # Harder
            "technical": [0.0, 0.0, 0.1],        # Medium to Hard
        }
        
        # Base distribution from config
        base_dist = config.difficulty_distribution
        
        # Apply aspect bias
        bias = aspect_difficulty_bias.get(aspect, [0.0, 0.0, 0.0])
        
        # Adjust probabilities
        adjusted_dist = {
            Difficulty.EASY: base_dist.get(Difficulty.EASY, 0.3) + bias[0],
            Difficulty.MEDIUM: base_dist.get(Difficulty.MEDIUM, 0.5) + bias[1], 
            Difficulty.HARD: base_dist.get(Difficulty.HARD, 0.2) + bias[2]
        }
        
        # Normalize
        total = sum(adjusted_dist.values())
        if total > 0:
            adjusted_dist = {k: v/total for k, v in adjusted_dist.items()}
        
        # Select based on adjusted probabilities
        import random
        rand = random.random()
        cumulative = 0.0
        for difficulty, prob in adjusted_dist.items():
            cumulative += prob
            if rand <= cumulative:
                return difficulty
                
        return Difficulty.MEDIUM
    
    def _get_aspect_focus(self, aspect: str) -> str:
        """Get focus description for the aspect."""
        focuses = {
            "definition": "Définition et concepts de base",
            "function": "Rôle et fonctionnalité",
            "application": "Applications pratiques et cas d'usage",
            "comparison": "Comparaisons et différences",
            "advantage": "Avantages et bénéfices",
            "implementation": "Mise en œuvre et implémentation",
            "problem_solving": "Résolution de problèmes",
            "technical": "Aspects techniques et architecture"
        }
        return focuses.get(aspect, "Aspect général")
    
    def _get_aspect_instruction(self, aspect: str) -> str:
        """Get specific instruction for the aspect."""
        instructions = {
            "definition": "Créez une question qui teste la compréhension des définitions et concepts fondamentaux.",
            "function": "Formulez une question sur le rôle et les fonctionnalités principales.",
            "application": "Développez une question sur les applications pratiques et les cas d'usage concrets.",
            "comparison": "Construisez une question qui compare différents éléments ou approches.",
            "advantage": "Élaborez une question sur les avantages, bénéfices ou valeurs ajoutées.",
            "implementation": "Créez une question technique sur la mise en œuvre ou l'implémentation.",
            "problem_solving": "Formulez une question de résolution de problème ou de choix stratégique.",
            "technical": "Développez une question sur les aspects techniques, l'architecture ou les détails d'implémentation."
        }
        return instructions.get(aspect, "Créez une question pertinente sur le contenu.")


class ChunkBasedQuestionGenerator:
    """
    Generates questions directly from document chunks with proper distribution.
    
    This approach ensures:
    - Each chunk contributes equally to question generation
    - Natural variety from different content
    - Comprehensive coverage of the document
    - Controlled distribution of questions per chunk
    """
    
    def __init__(self):
        """Initialize chunk-based generator."""
        self.llm_manager = get_llm_manager()
        self.prompt_builder = get_question_prompt_builder()
        self.parser = get_question_parser()
        self.selector = get_question_selector()
        self.document_manager = get_document_manager()
        self.variety_validator = ChunkQuestionVarietyValidator()
        
    async def generate_questions_from_document(
        self,
        document_id: int,
        total_questions: int,
        config: GenerationConfig,
        session_id: str = "chunk_based"
    ) -> Dict:
        """
        Generate questions from a document using chunk-based approach.
        
        Args:
            document_id: ID of the document
            total_questions: Total number of questions to generate
            config: Generation configuration
            session_id: Session ID for tracking
            
        Returns:
            Generation results with chunk distribution
        """
        logger.info(f"Starting chunk-based generation: {total_questions} questions from document {document_id}")
        
        try:
            # Get all chunks for the document
            chunks = await self._get_document_chunks(document_id)
            
            if not chunks:
                raise ChunkBasedGenerationError(f"No chunks found for document {document_id}")
            
            # Calculate distribution
            distribution = self._calculate_chunk_distribution(chunks, total_questions)
            
            logger.info(f"Distributing {total_questions} questions across {len(chunks)} chunks")
            logger.debug(f"Distribution: {distribution}")
            
            # Generate questions chunk by chunk
            all_questions = []
            generation_stats = {
                "chunks_processed": 0,
                "questions_generated": 0,
                "chunk_distributions": {},
                "errors": []
            }
            
            for i, (chunk, num_questions) in enumerate(distribution):
                if num_questions == 0:
                    continue
                    
                try:
                    logger.debug(f"Processing chunk {i+1}/{len(distribution)}: {num_questions} questions")
                    
                    # Update progress - starting chunk processing
                    update_progress(
                        session_id=session_id,
                        current_step=f"Traitement chunk {i+1}/{len(distribution)} ({num_questions} questions)",
                        metadata={"current_chunk": chunk.chunk_id, "chunk_index": i+1, "total_chunks": len(distribution)}
                    )
                    
                    chunk_questions = await self.generate_questions_from_chunk(
                        chunk=chunk,
                        num_questions=num_questions,
                        config=config,
                        document_id=document_id,
                        session_id=session_id
                    )
                    
                    all_questions.extend(chunk_questions)
                    generation_stats["questions_generated"] += len(chunk_questions)
                    generation_stats["chunk_distributions"][chunk.chunk_id] = {
                        "requested": num_questions,
                        "generated": len(chunk_questions),
                        "chunk_index": chunk.chunk_index
                    }
                    
                    # Update progress - chunk completed
                    update_progress(
                        session_id=session_id,
                        processed_questions=len(all_questions),
                        current_step=f"Chunk {i+1}/{len(distribution)} terminé ({len(chunk_questions)} questions générées)",
                        metadata={"chunk_completed": chunk.chunk_id, "questions_from_chunk": len(chunk_questions)}
                    )
                    
                except Exception as e:
                    error_msg = f"Failed to generate questions from chunk {chunk.chunk_id}: {e}"
                    logger.warning(error_msg)
                    generation_stats["errors"].append(error_msg)
                    
                    # Update progress - chunk failed
                    update_progress(
                        session_id=session_id,
                        current_step=f"Erreur chunk {i+1}/{len(distribution)}: {str(e)[:50]}...",
                        metadata={"chunk_error": chunk.chunk_id, "error": str(e)}
                    )
                    continue
                    
                generation_stats["chunks_processed"] += 1
            
            logger.info(f"Chunk-based generation completed: {len(all_questions)} questions from {generation_stats['chunks_processed']} chunks")
            
            return {
                "questions": all_questions,
                "generation_stats": generation_stats,
                "chunk_distribution": distribution
            }
            
        except Exception as e:
            logger.error(f"Chunk-based generation failed: {e}")
            raise ChunkBasedGenerationError(f"Generation failed: {e}")
    
    async def generate_questions_from_chunk(
        self,
        chunk: DocumentChunk,
        num_questions: int,
        config: GenerationConfig,
        document_id: int,
        session_id: str
    ) -> List[QuestionCreate]:
        """
        Generate multiple varied questions from a single chunk.
        
        Args:
            chunk: Source chunk
            num_questions: Number of questions to generate
            config: Generation configuration
            document_id: Document ID
            session_id: Session ID
            
        Returns:
            List of generated questions
        """
        logger.debug(f"Generating {num_questions} questions from chunk {chunk.chunk_id}")
        
        # Create variety plan
        question_plans = self.variety_validator.generate_varied_questions_plan(
            chunk, num_questions, config
        )
        
        questions = []
        
        for i, plan in enumerate(question_plans):
            try:
                # Update progress for individual question generation
                increment_progress(
                    session_id=session_id,
                    current_step=f"Génération question {i+1}/{len(question_plans)} ({plan['aspect']})",
                    metadata={"aspect": plan['aspect'], "question_in_chunk": i+1, "total_in_chunk": len(question_plans)}
                )
                
                question = await self._generate_single_question_from_chunk(
                    chunk=chunk,
                    plan=plan,
                    config=config,
                    document_id=document_id,
                    session_id=session_id
                )
                questions.append(question)
                
            except Exception as e:
                logger.warning(f"Failed to generate question with plan {plan['aspect']}: {e}")
                continue
        
        logger.debug(f"Generated {len(questions)} questions from chunk {chunk.chunk_id}")
        return questions
    
    async def _generate_single_question_from_chunk(
        self,
        chunk: DocumentChunk,
        plan: Dict,
        config: GenerationConfig,
        document_id: int,
        session_id: str
    ) -> QuestionCreate:
        """Generate a single question from chunk using the plan."""
        # Create context from chunk
        context = self._create_context_from_chunk(chunk, plan)
        
        # Build prompt with variety instruction
        base_prompt = self.prompt_builder.build_generation_prompt(
            context=context,
            config=config,
            question_type=plan["question_type"],
            difficulty=plan["difficulty"],
            language=config.language
        )
        
        # Add variety-specific instruction
        variety_prompt = f"{base_prompt}\n\n{plan['instruction']}\nFocus: {plan['focus']}"
        
        # Generate using LLM
        system_prompt = self.prompt_builder.build_system_prompt(config.language)
        response = await self.llm_manager.generate_response(
            prompt=variety_prompt,
            system_prompt=system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Parse response
        question = self.parser.parse_llm_response(
            response=response,
            config=config,
            document_id=document_id,
            session_id=session_id,
            topic=f"Chunk {chunk.chunk_id} - {plan['aspect']}",
            prompt=variety_prompt,
            source_chunks=[chunk.chunk_id],
            context_confidence=1.0  # High confidence since we're using the chunk directly
        )
        
        return question
    
    def _create_context_from_chunk(self, chunk: DocumentChunk, plan: Dict) -> QuestionContext:
        """Create question context from chunk."""
        return QuestionContext(
            topic=f"Chunk {chunk.chunk_index} - {plan['aspect']}",
            context_text=chunk.content,
            source_chunks=[chunk.chunk_id],
            themes=chunk.themes or [],
            confidence_score=1.0,
            metadata={
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "aspect": plan["aspect"],
                "focus": plan["focus"],
                "source": "chunk_direct"
            }
        )
    
    async def _get_document_chunks(self, document_id: int) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        try:
            # Use document manager to get chunks
            chunks = self.document_manager.get_document_chunks(document_id)
            
            # Convert to DocumentChunk schemas if needed
            if chunks and hasattr(chunks[0], 'chunk_text'):
                # Convert from database model to schema
                schema_chunks = []
                for chunk in chunks:
                    schema_chunk = DocumentChunk(
                        chunk_id=str(chunk.id),
                        document_id=str(chunk.document_id),
                        content=chunk.chunk_text,
                        chunk_index=chunk.chunk_order,
                        start_char=chunk.start_char or 0,
                        end_char=chunk.end_char or len(chunk.chunk_text),
                        metadata=chunk.metadata or {},
                        themes=chunk.themes or []
                    )
                    schema_chunks.append(schema_chunk)
                return schema_chunks
            
            return chunks or []
            
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []
    
    def _calculate_chunk_distribution(
        self, 
        chunks: List[DocumentChunk], 
        total_questions: int
    ) -> List[Tuple[DocumentChunk, int]]:
        """
        Calculate how many questions to generate from each chunk.
        
        Args:
            chunks: Available chunks
            total_questions: Total questions to distribute
            
        Returns:
            List of (chunk, num_questions) tuples
        """
        if not chunks:
            return []
            
        # Base distribution: divide equally
        base_per_chunk = total_questions // len(chunks)
        remainder = total_questions % len(chunks)
        
        # Assign base amount to each chunk
        distribution = [(chunk, base_per_chunk) for chunk in chunks]
        
        # Distribute remainder to chunks with more content or better themes
        if remainder > 0:
            # Sort chunks by content length and theme richness
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                score = len(chunk.content) + len(chunk.themes) * 100
                scored_chunks.append((score, i, chunk))
            
            # Sort by score (descending)
            scored_chunks.sort(reverse=True)
            
            # Give extra questions to top chunks
            for i in range(remainder):
                _, chunk_idx, _ = scored_chunks[i]
                chunk, count = distribution[chunk_idx]
                distribution[chunk_idx] = (chunk, count + 1)
        
        return distribution


# Global instance
_chunk_based_generator: ChunkBasedQuestionGenerator | None = None


def get_chunk_based_generator() -> ChunkBasedQuestionGenerator:
    """Get the global chunk-based generator instance."""
    global _chunk_based_generator
    if _chunk_based_generator is None:
        _chunk_based_generator = ChunkBasedQuestionGenerator()
    return _chunk_based_generator


# Convenience functions
async def generate_questions_from_document_chunks(
    document_id: int,
    total_questions: int,
    config: GenerationConfig,
    session_id: str = "chunk_based"
) -> Dict:
    """Generate questions from document using chunk-based approach."""
    generator = get_chunk_based_generator()
    return await generator.generate_questions_from_document(
        document_id, total_questions, config, session_id
    )


async def generate_questions_from_single_chunk(
    chunk: DocumentChunk,
    num_questions: int,
    config: GenerationConfig,
    document_id: int,
    session_id: str = "chunk_based"
) -> List[QuestionCreate]:
    """Generate questions from a single chunk."""
    generator = get_chunk_based_generator()
    return await generator.generate_questions_from_chunk(
        chunk, num_questions, config, document_id, session_id
    )