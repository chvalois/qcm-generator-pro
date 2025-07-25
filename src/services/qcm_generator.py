"""
QCM Generator Pro - QCM Generation Service

Orchestrates the QCM generation process using specialized services.
Follows SRP by delegating specific tasks to dedicated services.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import GenerationConfig, QuestionCreate, QuestionContext
from src.services.llm_manager import get_llm_manager
from src.services.rag_engine import get_rag_engine
from src.services.question_prompt_builder import get_question_prompt_builder
from src.services.question_parser import get_question_parser
from src.services.question_selection import get_question_selector
from src.services.progressive_workflow import get_progressive_workflow_manager

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
        
    def create_question_generation_prompt(
        self, 
        context: QuestionContext,
        config: GenerationConfig,
        question_type: QuestionType,
        difficulty: Difficulty,
        language: Language = Language.FR
    ) -> str:
        """
        Create prompt for question generation.
        
        Delegates to QuestionPromptBuilder service.
        """
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
        session_id: str = "default"
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
            logger.debug(f"Generating single question for topic: {topic}")
            
            # Get context from RAG engine
            context = self._get_or_create_context(topic, document_ids, themes_filter)
            
            # Select question parameters
            question_type = self.selector.select_question_type(config)
            difficulty = self.selector.select_difficulty(config)
            
            # Create generation prompt
            prompt = self.create_question_generation_prompt(
                context=context,
                config=config,
                question_type=question_type,
                difficulty=difficulty,
                language=config.language
            )
            
            # Generate question using LLM
            try:
                system_prompt = self.prompt_builder.build_system_prompt(config.language)
                response = await self.llm_manager.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
            except Exception as e:
                logger.error(f"LLM generation failed for topic '{topic}': {e}")
                raise QCMGenerationError(f"LLM generation failed: {e}")
            
            # Parse response into QuestionCreate object
            return self.parser.parse_llm_response(
                response=response,
                config=config,
                document_id=document_id,
                session_id=session_id,
                topic=topic,
                prompt=prompt,
                source_chunks=context.source_chunks,
                context_confidence=context.confidence_score
            )
            
        except Exception as e:
            logger.error(f"Failed to generate question for topic '{topic}': {e}")
            raise QCMGenerationError(f"Question generation failed: {e}")

    
    def _get_or_create_context(
        self, 
        topic: str, 
        document_ids: Optional[List[str]], 
        themes_filter: Optional[List[str]]
    ) -> QuestionContext:
        """Get context from RAG or create fallback context."""
        context = self.rag_engine.get_question_context(
            topic=topic,
            document_ids=document_ids,
            themes_filter=themes_filter
        )
        
        if not context.context_text.strip():
            logger.warning(f"No relevant context found for topic: {topic}")
            context = QuestionContext(
                topic=topic,
                context_text=(
                    f"Contenu éducatif sur le thème: {topic}. "
                    f"Ce thème couvre les concepts fondamentaux et les applications pratiques."
                ),
                source_chunks=[],
                themes=[topic],
                confidence_score=0.5,
                metadata={"fallback": True, "demo_mode": True}
            )
            logger.info(f"Using fallback context for topic: {topic}")
        
        return context
            

        

        
    async def generate_questions_batch(
        self,
        topics: List[str],
        config: GenerationConfig,
        document_ids: Optional[List[str]] = None,
        themes_filter: Optional[List[str]] = None,
        batch_size: int = 5,
        session_id: str = "default"
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
        
        for i in range(min(batch_size, len(topics))):
            topic = topics[i % len(topics)]  # Cycle through topics if needed
            
            try:
                question = await self.generate_single_question(
                    topic=topic,
                    config=config,
                    document_ids=document_ids,
                    themes_filter=themes_filter,
                    document_id=document_id,
                    session_id=session_id
                )
                questions.append(question)
                
            except QCMGenerationError as e:
                logger.warning(f"Failed to generate question for topic '{topic}': {e}")
                continue
                
        return questions

    
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
        
        # Create generation callback that uses this instance
        async def generation_callback(batch_size: int) -> List[QuestionCreate]:
            return await self.generate_questions_batch(
                topics=topics,
                config=config,
                document_ids=document_ids,
                themes_filter=themes_filter,
                batch_size=batch_size,
                session_id=session_id or "default"
            )
        
        # Delegate to workflow manager
        return await self.workflow_manager.execute_progressive_workflow(
            total_questions=config.num_questions,
            generation_callback=generation_callback,
            validation_callback=validation_callback,
            session_id=session_id
        )


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
    session_id: str = "default"
) -> QuestionCreate:
    """Generate a single QCM question."""
    generator = get_qcm_generator()
    return await generator.generate_single_question(
        topic, config, document_ids, themes_filter, document_id, session_id
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