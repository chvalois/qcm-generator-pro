"""
QCM Generator Pro - QCM Generation Service

Orchestrates the QCM generation process using specialized services.
Follows SRP by delegating specific tasks to dedicated services.
"""

import logging
import random
import uuid
from typing import Any, Callable, Dict, List, Optional

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import GenerationConfig, QuestionCreate, QuestionContext
from src.services.llm.llm_manager import get_llm_manager
from src.services.infrastructure.rag_engine import get_rag_engine
from src.services.generation.question_prompt_builder import get_question_prompt_builder
from src.services.generation.question_parser import get_question_parser
from src.services.generation.question_selection import get_question_selector
from src.services.generation.progressive_workflow import get_progressive_workflow_manager
from src.services.infrastructure.progress_tracker import (
    start_progress_session, update_progress, increment_progress, 
    complete_progress_session, fail_progress_session
)
from src.services.quality.question_deduplicator import get_question_deduplicator
from src.services.quality.question_diversity_enhancer import get_diversity_enhancer
from src.services.llm.langsmith_tracker import get_langsmith_tracker

# LangSmith imports
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    # Fallback decorator
    def traceable(name: str = None, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)


class QCMGenerationError(Exception):
    """Exception raised when QCM generation fails."""
    pass


class QCMGenerator:
    """
    Orchestrator service for QCM generation.
    
    This class delegates specific responsibilities to specialized services:
    - QuestionPromptBuilder: Prompt creation
    - QuestionParser: Response parsing  
    - QuestionSelector: Type/difficulty selection
    - ProgressiveWorkflowManager: Workflow orchestration
    """
    
    def __init__(self):
        """Initialize QCM generator with specialized services."""
        self.llm_manager = get_llm_manager()
        self.rag_engine = get_rag_engine()
        self.prompt_builder = get_question_prompt_builder()
        self.parser = get_question_parser()
        self.selector = get_question_selector()
        self.workflow_manager = get_progressive_workflow_manager()
        self.deduplicator = get_question_deduplicator()
        self.diversity_enhancer = get_diversity_enhancer()
        self.langsmith_tracker = get_langsmith_tracker()
        
        # Track generation diversity
        self.used_contexts: List[str] = []
        self.used_topics_count: Dict[str, int] = {}
        
    def create_question_generation_prompt(
        self, 
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty,
        language: Language = Language.FR,
        examples_file: Optional[str] = None,
        max_examples: int = 3
    ) -> str:
        """
        Create prompt for question generation.
        
        Delegates to QuestionPromptBuilder service.
        """
        # Use Few-Shot Examples if provided
        if examples_file:
            return self.prompt_builder.build_generation_prompt_with_examples(
                context=context,
                config=config,
                question_type=question_type,
                difficulty=difficulty,
                language=language,
                examples_file=examples_file,
                max_examples=max_examples
            )
        else:
            return self.prompt_builder.build_generation_prompt(
                context=context,
                config=config,
                question_type=question_type,
                difficulty=difficulty,
                language=language
            )
        
    async def generate_single_question(
        self,
        topic: str,
        config: GenerationConfig,
        document_ids: Optional[List[str]] = None,
        themes_filter: Optional[List[str]] = None,
        document_id: int = 1,
        session_id: str = "default",
        examples_file: Optional[str] = None,
        max_examples: int = 3
    ) -> QuestionCreate:
        """
        Generate a single QCM question.
        
        Orchestrates the generation process using specialized services.
        
        Args:
            topic: Question topic
            config: Generation configuration
            document_ids: Filter by document IDs
            themes_filter: Filter by themes
            document_id: Document ID for the question
            session_id: Session ID for the question
            
        Returns:
            Generated question
            
        Raises:
            QCMGenerationError: If generation fails
        """
        try:
            import time
            generation_start_time = time.time()
            logger.debug(f"Generating single question for topic: {topic}")
            
            # Start LangSmith tracing
            with self.langsmith_tracker.trace_qcm_generation(
                session_id=session_id,
                topic=topic,
                question_type="auto-select",  # Will be determined
                difficulty="auto-select",    # Will be determined
                examples_file=examples_file,
                metadata={
                    "document_ids": document_ids,
                    "themes_filter": themes_filter
                }
            ) as langsmith_run_id:
                
                # Get context from RAG engine with diversity check
                context = self._get_or_create_context(topic, document_ids, themes_filter, avoid_similar=True)
                
                # Select question parameters
                question_type = self.selector.select_question_type(config)
                difficulty = self.selector.select_difficulty(config)
                
                # Create generation prompt with diversity enhancement
                base_prompt = self.create_question_generation_prompt(
                    context=context,
                    config=config,
                    question_type=question_type,
                    difficulty=difficulty,
                    language=config.language,
                    examples_file=examples_file,
                    max_examples=max_examples
                )
                
                # Apply diversity enhancement based on topic usage
                diversity_level = min(self.used_topics_count.get(topic, 0), 3)
                prompt = self.diversity_enhancer.enhance_prompt_diversity(
                    base_prompt=base_prompt,
                    context=context,
                    config=config,
                    question_type=question_type,
                    difficulty=difficulty,
                    diversity_level=diversity_level
                )
            
                # Enrich prompt with Few-Shot context for LangSmith visibility
                if examples_file:
                    # Add Few-Shot context at the beginning of the prompt
                    try:
                        from .simple_examples_loader import get_examples_loader
                        loader = get_examples_loader()
                        examples = loader.get_examples_for_context(examples_file, max_examples=max_examples or 3)
                        
                        if examples:
                            fewshot_header = f"\n=== FEW-SHOT EXAMPLES (File: {examples_file}, Count: {len(examples)}) ===\n\n"
                            
                            for i, ex in enumerate(examples, 1):
                                fewshot_header += f"EXAMPLE {i}:\n"
                                fewshot_header += f"Thème: {ex.get('theme', 'N/A')}\n"
                                fewshot_header += f"Difficulté: {ex.get('difficulty', 'N/A')}\n"
                                fewshot_header += f"Question: {ex.get('question', 'N/A')}\n"
                                fewshot_header += f"Options: {ex.get('options', [])}\n"
                                fewshot_header += f"Correct: {ex.get('correct', [])}\n"
                                fewshot_header += f"Explication: {ex.get('explanation', 'N/A')}\n\n"
                            
                            fewshot_header += "=== END EXAMPLES ===\n\n"
                            fewshot_header += "Utilisez ces exemples comme guide pour générer une question similaire:\n\n"
                            
                            # Prepend to the existing prompt
                            prompt = fewshot_header + prompt
                            
                    except Exception as e:
                        logger.warning(f"Could not load Few-Shot examples: {e}")

                # Generate question using LLM
                try:
                    system_prompt = self.prompt_builder.build_system_prompt(config.language)
                    response = await self.llm_manager.generate_response(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        langsmith_run_id=langsmith_run_id,
                        # LangSmith metadata
                        examples_file=examples_file,
                        max_examples=max_examples,
                        topic=topic,
                        question_type=question_type.value if question_type else None,
                        difficulty=difficulty.value if difficulty else None
                    )
                except Exception as e:
                    logger.error(f"LLM generation failed for topic '{topic}': {e}")
                    raise QCMGenerationError(f"LLM generation failed: {e}")
                
                # Parse response into QuestionCreate object
                question = self.parser.parse_llm_response(
                    response=response,
                    config=config,
                    document_id=document_id,
                    session_id=session_id,
                    topic=topic,
                    prompt=prompt,
                    source_chunks=context.source_chunks,
                    context_confidence=context.confidence_score
                )
                
                # Check for duplicates and regenerate if needed
                if self.deduplicator.is_duplicate(question):
                    logger.warning(f"Duplicate question detected for topic: {topic}, regenerating...")
                    # Try once more with enhanced diversity
                    alternative_context = self._get_or_create_context(
                        self._generate_alternative_topic(topic), 
                        document_ids, 
                        themes_filter, 
                        avoid_similar=True
                    )
                    
                    alternative_prompt = self.create_question_generation_prompt(
                        context=alternative_context,
                        config=config,
                        question_type=question_type,
                        difficulty=difficulty,
                        language=config.language,
                        examples_file=examples_file,
                        max_examples=max_examples
                    )
                    
                    try:
                        system_prompt = self.prompt_builder.build_system_prompt(config.language)
                        alternative_response = await self.llm_manager.generate_response(
                            prompt=alternative_prompt,
                            system_prompt=system_prompt,
                            temperature=min(config.temperature + 0.3, 1.0),  # Increase creativity
                            max_tokens=config.max_tokens,
                            # LangSmith metadata
                            examples_file=examples_file,
                            max_examples=max_examples,
                            topic=f"{topic} (alternative)",
                            question_type=question_type.value if question_type else None,
                            difficulty=difficulty.value if difficulty else None
                        )
                        
                        question = self.parser.parse_llm_response(
                            response=alternative_response,
                            config=config,
                            document_id=document_id,
                            session_id=session_id,
                            topic=f"{topic} (alternative)",
                            prompt=alternative_prompt,
                            source_chunks=alternative_context.source_chunks,
                            context_confidence=alternative_context.confidence_score
                        )
                    except Exception as e:
                        logger.warning(f"Alternative generation failed: {e}, using original question")
                
                # Add question to deduplicator tracking
                self.deduplicator.add_question(question)
                
                # Log generation success (LangSmith tracing is handled by @traceable decorator)
                generation_time = time.time() - generation_start_time
                logger.info(f"Question generated successfully in {generation_time:.2f}s for topic: {topic}")
                
                return question
            
        except Exception as e:
            logger.error(f"Failed to generate question for topic '{topic}': {e}")
            raise QCMGenerationError(f"Question generation failed: {e}")

    
    def _get_or_create_context(
        self, 
        topic: str, 
        document_ids: Optional[List[str]], 
        themes_filter: Optional[List[str]],
        avoid_similar: bool = True
    ) -> QuestionContext:
        """Get context from RAG or create fallback context with diversity."""
        # Try to get diverse context by using different search terms
        enhanced_topic = self._enhance_topic_for_diversity(topic)
        
        context = self.rag_engine.get_question_context(
            topic=enhanced_topic,
            document_ids=document_ids,
            themes_filter=themes_filter
        )
        
        # If context is too similar to previous ones, try alternative search
        if avoid_similar and self._is_context_too_similar(context.context_text):
            logger.debug(f"Context too similar for topic: {topic}, trying alternative search")
            alternative_topic = self._generate_alternative_topic(topic)
            context = self.rag_engine.get_question_context(
                topic=alternative_topic,
                document_ids=document_ids,
                themes_filter=themes_filter
            )
        
        if not context.context_text.strip():
            logger.warning(f"No relevant context found for topic: {topic}")
            context = self._create_diverse_fallback_context(topic)
            logger.info(f"Using diverse fallback context for topic: {topic}")
        
        # Track used contexts
        self.used_contexts.append(context.context_text[:200])  # Store first 200 chars
        
        return context
    
    def _enhance_topic_for_diversity(self, topic: str) -> str:
        """Enhance topic search to promote diversity based on usage count."""
        usage_count = self.used_topics_count.get(topic, 0)
        self.used_topics_count[topic] = usage_count + 1
        
        if usage_count == 0:
            return topic  # First use, keep original
        elif usage_count == 1:
            return f"{topic} applications pratiques"  # Second use, focus on applications
        elif usage_count == 2:
            return f"{topic} concepts avancés"  # Third use, focus on advanced concepts
        else:
            # For subsequent uses, add variety modifiers
            modifiers = ["exemples concrets", "cas d'usage", "implémentation", "techniques", "méthodes"]
            modifier = modifiers[usage_count % len(modifiers)]
            return f"{topic} {modifier}"
    
    def _generate_alternative_topic(self, topic: str) -> str:
        """Generate alternative search terms for the same topic."""
        alternatives = {
            "Microsoft Fabric": ["plateforme analytique", "solution données", "architecture données"],
            "données": ["information", "analyse", "traitement"],
            "analytique": ["analyse", "business intelligence", "décisionnel"],
            "entrepôt": ["warehouse", "stockage", "centralisation"],
            "intégration": ["fusion", "consolidation", "assemblage"]
        }
        
        for key, alts in alternatives.items():
            if key.lower() in topic.lower():
                return random.choice(alts)
        
        # Generic fallback: add descriptive terms
        descriptors = ["fonctionnalités", "caractéristiques", "aspects", "éléments"]
        return f"{topic} {random.choice(descriptors)}"
    
    def _is_context_too_similar(self, new_context: str) -> bool:
        """Check if new context is too similar to previously used contexts."""
        if not self.used_contexts:
            return False
        
        new_words = set(new_context.lower().split())
        
        for used_context in self.used_contexts[-5:]:  # Check last 5 contexts
            used_words = set(used_context.lower().split())
            
            if len(new_words) == 0 or len(used_words) == 0:
                continue
                
            # Calculate similarity ratio
            intersection = len(new_words.intersection(used_words))
            union = len(new_words.union(used_words))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.7:  # 70% similarity threshold
                return True
        
        return False
    
    def _create_diverse_fallback_context(self, topic: str) -> QuestionContext:
        """Create diverse fallback contexts based on topic usage."""
        usage_count = self.used_topics_count.get(topic, 0)
        
        base_contexts = [
            f"Contenu éducatif sur {topic}. Ce domaine englobe les principes fondamentaux et leur mise en application.",
            f"Formation technique sur {topic}. Focus sur les méthodes et outils pratiques utilisés en entreprise.",
            f"Guide professionnel {topic}. Exploration des cas d'usage et des meilleures pratiques.",
            f"Documentation {topic}. Présentation des concepts clés et de leur implémentation.",
            f"Manuel {topic}. Analyse des fonctionnalités et de leur utilisation optimale."
        ]
        
        context_text = base_contexts[usage_count % len(base_contexts)]
        
        return QuestionContext(
            topic=topic,
            context_text=context_text,
            source_chunks=[],
            themes=[topic],
            confidence_score=0.5,
            metadata={"fallback": True, "demo_mode": True, "diversity_level": usage_count}
        )
            

        

        
    async def generate_questions_batch(
        self,
        topics: List[str],
        config: GenerationConfig,
        document_ids: Optional[List[str]] = None,
        themes_filter: Optional[List[str]] = None,
        batch_size: int = 5,
        session_id: str = "default",
        progress_session_id: Optional[str] = None,
        examples_file: Optional[str] = None,
        max_examples: int = 3
    ) -> List[QuestionCreate]:
        """
        Generate a batch of questions.
        
        Simplified orchestration for batch generation.
        
        Args:
            topics: List of topics
            config: Generation configuration
            document_ids: Filter by document IDs
            themes_filter: Filter by themes
            batch_size: Number of questions to generate
            session_id: Session ID for the questions
            
        Returns:
            List of generated questions
        """
        questions = []
        
        # Convert document_ids to integer if needed
        document_id = self._parse_document_id(document_ids)
        
        # Create diverse topic variations instead of cycling
        diverse_topics = self._create_diverse_topic_list(topics, batch_size)
        
        for i, topic in enumerate(diverse_topics[:batch_size]):
            try:
                # Update progress if tracking session exists
                if progress_session_id:
                    update_progress(
                        progress_session_id, 
                        current_step=f"Génération question {i+1}/{batch_size}: {topic}"
                    )
                
                question = await self.generate_single_question(
                    topic=topic,
                    config=config,
                    document_ids=document_ids,
                    themes_filter=themes_filter,
                    document_id=document_id,
                    session_id=session_id,
                    examples_file=examples_file,
                    max_examples=max_examples
                )
                questions.append(question)
                
                # Increment progress after successful generation
                if progress_session_id:
                    increment_progress(
                        progress_session_id,
                        current_step=f"Question générée: {topic}"
                    )
                
            except QCMGenerationError as e:
                logger.warning(f"Failed to generate question for topic '{topic}': {e}")
                continue
                
        return questions

    
    def _create_diverse_topic_list(self, topics: List[str], target_size: int) -> List[str]:
        """Create a diverse list of topics with variations to avoid repetition."""
        if not topics:
            return []
            
        diverse_topics = []
        
        # First, add original topics
        for topic in topics:
            diverse_topics.append(topic)
            
        # Use diversity enhancer to create variations
        for base_topic in topics:
            if len(diverse_topics) >= target_size:
                break
                
            # Generate variations using the diversity enhancer
            variations = self.diversity_enhancer.create_topic_variations(
                base_topic, 
                count=min(5, target_size - len(diverse_topics) + 1)
            )
            
            # Add variations that aren't already in the list
            for variation in variations[1:]:  # Skip first (original)
                if len(diverse_topics) >= target_size:
                    break
                if variation not in diverse_topics:
                    diverse_topics.append(variation)
        
        return diverse_topics[:target_size]
    
    
    def reset_diversity_tracking(self):
        """Reset diversity tracking for new generation sessions."""
        self.deduplicator.reset()
        self.used_contexts.clear()
        self.used_topics_count.clear()
        logger.info("Diversity tracking reset")
    
    def _parse_document_id(self, document_ids: Optional[List[str]]) -> int:
        """Parse document IDs to get a consistent integer ID."""
        if not document_ids:
            return 1
            
        doc_id_str = str(document_ids[0])
        try:
            return int(doc_id_str)
        except ValueError:
            # Use hash for non-numeric IDs
            return abs(hash(doc_id_str)) % (10**8)
    

        
    async def generate_progressive_qcm(
        self,
        topics: List[str],
        config: GenerationConfig,
        document_ids: Optional[List[str]] = None,
        themes_filter: Optional[List[str]] = None,
        validation_callback: Optional[Callable] = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Generate QCM using progressive validation workflow.
        
        Delegates to ProgressiveWorkflowManager service.
        
        Args:
            topics: List of topics for questions
            config: Generation configuration
            document_ids: Filter by document IDs
            themes_filter: Filter by themes
            validation_callback: Optional callback for validation steps
            session_id: Session ID for the questions
            
        Returns:
            Generation results with validation steps
        """
        logger.info(f"Starting progressive QCM generation for {len(topics)} topics")
        
        # Initialize progress tracking session
        progress_session_id = session_id or f"qcm_generation_{uuid.uuid4().hex[:8]}"
        start_progress_session(
            session_id=progress_session_id,
            total_questions=config.num_questions,
            initial_step="Initialisation de la génération progressive"
        )
        
        try:
            # Create generation callback that uses this instance with progress tracking
            async def generation_callback(batch_size: int) -> List[QuestionCreate]:
                return await self.generate_questions_batch(
                    topics=topics,
                    config=config,
                    document_ids=document_ids,
                    themes_filter=themes_filter,
                    batch_size=batch_size,
                    session_id=session_id or "default",
                    progress_session_id=progress_session_id
                )
            
            # Delegate to workflow manager
            result = await self.workflow_manager.execute_progressive_workflow(
                total_questions=config.num_questions,
                generation_callback=generation_callback,
                validation_callback=validation_callback,
                session_id=session_id
            )
            
            # Complete progress session
            complete_progress_session(
                progress_session_id, 
                final_step="Génération progressive terminée"
            )
            
            # Add progress session ID to result
            result['progress_session_id'] = progress_session_id
            
            return result
            
        except Exception as e:
            # Fail progress session on error
            fail_progress_session(
                progress_session_id,
                error_message=str(e),
                error_step="Erreur lors de la génération"
            )
            raise


# Global QCM generator instance
_qcm_generator: QCMGenerator | None = None


def get_qcm_generator() -> QCMGenerator:
    """Get the global QCM generator instance."""
    global _qcm_generator
    if _qcm_generator is None:
        _qcm_generator = QCMGenerator()
    return _qcm_generator


# Convenience functions
async def generate_qcm_question(
    topic: str,
    config: GenerationConfig,
    document_ids: Optional[List[str]] = None,
    themes_filter: Optional[List[str]] = None,
    document_id: int = 1,
    session_id: str = "default",
    examples_file: Optional[str] = None,
    max_examples: int = 3
) -> QuestionCreate:
    """Generate a single QCM question."""
    generator = get_qcm_generator()
    return await generator.generate_single_question(
        topic, config, document_ids, themes_filter, document_id, session_id,
        examples_file, max_examples
    )


async def generate_progressive_qcm(
    topics: List[str],
    config: GenerationConfig,
    document_ids: Optional[List[str]] = None,
    themes_filter: Optional[List[str]] = None,
    validation_callback: Optional[Callable] = None,
    session_id: str = None
) -> Dict[str, Any]:
    """Generate QCM using progressive workflow."""
    generator = get_qcm_generator()
    return await generator.generate_progressive_qcm(
        topics, config, document_ids, themes_filter, validation_callback, session_id
    )


def generate_progressive_qcm_sync(
    topics: List[str],
    config: GenerationConfig,
    document_ids: Optional[List[str]] = None,
    themes_filter: Optional[List[str]] = None,
    validation_callback: Optional[Callable] = None,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for generate_progressive_qcm.
    
    Simplified version without complex event loop management.
    """
    import asyncio
    
    try:
        # Use asyncio.run for clean execution
        return asyncio.run(
            generate_progressive_qcm(
                topics=topics,
                config=config,
                document_ids=document_ids,
                themes_filter=themes_filter,
                validation_callback=validation_callback,
                session_id=session_id
            )
        )
    except Exception as e:
        logger.error(f"Error in sync QCM generation: {e}")
        return {
            "final_questions": [],
            "generation_stats": {
                "total_generated": 0,
                "total_requested": config.num_questions,
                "difficulty_distribution": {},
                "type_distribution": {},
                "error": str(e)
            }
        }