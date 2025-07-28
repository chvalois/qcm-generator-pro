"""
Enhanced QCM Generator

Combines the original QCM generator with the new chunk-based approach.
Provides both topic-based and chunk-based generation methods.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import GenerationConfig, QuestionCreate
from src.services.qcm_generator import get_qcm_generator
from src.services.chunk_based_generator import get_chunk_based_generator
from src.services.chunk_variety_validator import get_chunk_variety_validator
from src.services.progress_tracker import (
    get_progress_tracker, start_progress_session, update_progress, 
    complete_progress_session, fail_progress_session
)

logger = logging.getLogger(__name__)


class GenerationMode(str, Enum):
    """Generation modes available."""
    TOPIC_BASED = "topic_based"      # Original approach using topics
    CHUNK_BASED = "chunk_based"      # New approach using document chunks
    HYBRID = "hybrid"                # Combination of both approaches


class EnhancedQCMGenerator:
    """
    Enhanced QCM generator that supports multiple generation strategies.
    
    Features:
    - Chunk-based generation for better content coverage
    - Topic-based generation for flexibility  
    - Hybrid approach combining both methods
    - Automatic variety validation
    - Smart mode selection based on available data
    """
    
    def __init__(self):
        """Initialize enhanced generator."""
        self.topic_generator = get_qcm_generator()
        self.chunk_generator = get_chunk_based_generator()
        self.variety_validator = get_chunk_variety_validator()
        
    async def generate_questions(
        self,
        config: GenerationConfig,
        document_ids: Optional[List[str]] = None,
        mode: GenerationMode = GenerationMode.CHUNK_BASED,
        session_id: str = "enhanced",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate questions using the specified mode.
        
        Args:
            config: Generation configuration
            document_ids: Document IDs to use (for chunk-based)
            mode: Generation mode to use
            session_id: Session ID for tracking
            **kwargs: Additional parameters
            
        Returns:
            Generation results with statistics and validation
        """
        logger.info(f"Starting enhanced QCM generation: {config.num_questions} questions using {mode.value} mode")
        
        # Initialize progress tracking
        progress_session = start_progress_session(
            session_id=session_id,
            total_questions=config.num_questions,
            initial_step=f"Démarrage génération {mode.value}"
        )
        
        try:
            if mode == GenerationMode.CHUNK_BASED:
                return await self._generate_chunk_based(config, document_ids, session_id, **kwargs)
            elif mode == GenerationMode.TOPIC_BASED:
                return await self._generate_topic_based(config, document_ids, session_id, **kwargs)
            elif mode == GenerationMode.HYBRID:
                return await self._generate_hybrid(config, document_ids, session_id, **kwargs)
            else:
                raise ValueError(f"Unknown generation mode: {mode}")
                
        except Exception as e:
            logger.error(f"Enhanced generation failed: {e}")
            fail_progress_session(session_id, str(e), "Erreur de génération")
            return {
                "questions": [],
                "generation_stats": {
                    "total_generated": 0,
                    "total_requested": config.num_questions,
                    "mode": mode.value,
                    "error": str(e)
                },
                "variety_validation": None
            }
    
    async def _generate_chunk_based(
        self,
        config: GenerationConfig,
        document_ids: Optional[List[str]],
        session_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using chunk-based approach."""
        if not document_ids:
            raise ValueError("Document IDs required for chunk-based generation")
        
        # For now, use the first document ID (can be enhanced to handle multiple documents)
        document_id = int(document_ids[0]) if document_ids else 1
        
        # Generate using chunk-based generator
        chunk_result = await self.chunk_generator.generate_questions_from_document(
            document_id=document_id,
            total_questions=config.num_questions,
            config=config,
            session_id=session_id
        )
        
        # Validate variety if we have multiple questions
        variety_validation = None
        if len(chunk_result["questions"]) > 1:
            # Group questions by chunk for variety validation
            chunk_questions = self._group_questions_by_chunk(chunk_result["questions"])
            variety_validation = self.variety_validator.validate_multiple_chunks(chunk_questions)
        
        return {
            "questions": chunk_result["questions"],
            "generation_stats": {
                **chunk_result["generation_stats"],
                "mode": GenerationMode.CHUNK_BASED.value,
                "total_generated": len(chunk_result["questions"]),
                "total_requested": config.num_questions
            },
            "chunk_distribution": chunk_result.get("chunk_distribution"),
            "variety_validation": variety_validation
        }
    
    async def _generate_topic_based(
        self,
        config: GenerationConfig,
        document_ids: Optional[List[str]],
        session_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using original topic-based approach."""
        topics = kwargs.get("topics", ["General topic"])
        themes_filter = kwargs.get("themes_filter")
        
        # Use original generator
        topic_result = await self.topic_generator.generate_progressive_qcm(
            topics=topics,
            config=config,
            document_ids=document_ids,
            themes_filter=themes_filter,
            session_id=session_id
        )
        
        questions = topic_result.get("final_questions", [])
        
        return {
            "questions": questions,
            "generation_stats": {
                **topic_result.get("generation_stats", {}),
                "mode": GenerationMode.TOPIC_BASED.value,
                "total_generated": len(questions),
                "total_requested": config.num_questions
            },
            "variety_validation": None  # Not applicable for topic-based
        }
    
    async def _generate_hybrid(
        self,
        config: GenerationConfig,
        document_ids: Optional[List[str]],
        session_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using hybrid approach (chunk + topic)."""
        # Split questions between chunk-based and topic-based
        chunk_questions = int(config.num_questions * 0.7)  # 70% chunk-based
        topic_questions = config.num_questions - chunk_questions  # 30% topic-based
        
        all_questions = []
        all_stats = {
            "mode": GenerationMode.HYBRID.value,
            "chunk_based_requested": chunk_questions,
            "topic_based_requested": topic_questions,
            "chunk_based_generated": 0,
            "topic_based_generated": 0
        }
        
        # Generate chunk-based questions
        if chunk_questions > 0 and document_ids:
            chunk_config = config.model_copy()
            chunk_config.num_questions = chunk_questions
            
            chunk_result = await self._generate_chunk_based(
                chunk_config, document_ids, f"{session_id}_chunk", **kwargs
            )
            
            all_questions.extend(chunk_result["questions"])
            all_stats["chunk_based_generated"] = len(chunk_result["questions"])
            all_stats["chunk_distribution"] = chunk_result.get("chunk_distribution")
        
        # Generate topic-based questions
        if topic_questions > 0:
            topic_config = config.model_copy()
            topic_config.num_questions = topic_questions
            
            topic_result = await self._generate_topic_based(
                topic_config, document_ids, f"{session_id}_topic", **kwargs
            )
            
            all_questions.extend(topic_result["questions"])
            all_stats["topic_based_generated"] = len(topic_result["questions"])
        
        return {
            "questions": all_questions,
            "generation_stats": {
                **all_stats,
                "total_generated": len(all_questions),
                "total_requested": config.num_questions
            },
            "variety_validation": None
        }
    
    def _group_questions_by_chunk(self, questions: List[QuestionCreate]) -> Dict[str, List[QuestionCreate]]:
        """Group questions by their source chunk."""
        chunk_questions = {}
        
        for question in questions:
            # Extract chunk ID from source_chunks or metadata
            chunk_id = "unknown"
            
            if hasattr(question, 'source_chunks') and question.source_chunks:
                chunk_id = question.source_chunks[0]
            elif hasattr(question, 'metadata') and question.metadata:
                chunk_id = question.metadata.get('chunk_id', 'unknown')
            
            if chunk_id not in chunk_questions:
                chunk_questions[chunk_id] = []
            chunk_questions[chunk_id].append(question)
        
        return chunk_questions
    
    def recommend_generation_mode(
        self,
        document_ids: Optional[List[str]] = None,
        total_questions: int = 10,
        available_chunks: int = 0
    ) -> GenerationMode:
        """
        Recommend the best generation mode based on available data.
        
        Args:
            document_ids: Available document IDs
            total_questions: Number of questions to generate
            available_chunks: Number of available chunks
            
        Returns:
            Recommended generation mode
        """
        # If no documents, use topic-based
        if not document_ids or available_chunks == 0:
            logger.info("Recommending TOPIC_BASED: No documents or chunks available")
            return GenerationMode.TOPIC_BASED
        
        # If very few chunks relative to questions, use hybrid
        if available_chunks < total_questions * 0.5:
            logger.info(f"Recommending HYBRID: Few chunks ({available_chunks}) for {total_questions} questions")
            return GenerationMode.HYBRID
        
        # If good chunk coverage, use chunk-based
        logger.info(f"Recommending CHUNK_BASED: Good chunk coverage ({available_chunks} chunks)")
        return GenerationMode.CHUNK_BASED
    
    async def generate_with_auto_mode(
        self,
        config: GenerationConfig,
        document_ids: Optional[List[str]] = None,
        session_id: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate questions with automatic mode selection.
        
        Args:
            config: Generation configuration
            document_ids: Document IDs (if available)
            session_id: Session ID
            **kwargs: Additional parameters
            
        Returns:
            Generation results
        """
        # Get available chunks count (simplified - would need actual implementation)
        available_chunks = 0
        if document_ids:
            try:
                # This would need to query the actual document to get chunk count
                available_chunks = 10  # Placeholder
            except Exception:
                available_chunks = 0
        
        # Recommend mode
        recommended_mode = self.recommend_generation_mode(
            document_ids, config.num_questions, available_chunks
        )
        
        logger.info(f"Auto-selected generation mode: {recommended_mode.value}")
        
        # Generate with recommended mode
        return await self.generate_questions(
            config=config,
            document_ids=document_ids,
            mode=recommended_mode,
            session_id=session_id,
            **kwargs
        )


# Global instance
_enhanced_generator: EnhancedQCMGenerator | None = None


def get_enhanced_qcm_generator() -> EnhancedQCMGenerator:
    """Get the global enhanced QCM generator instance."""
    global _enhanced_generator
    if _enhanced_generator is None:
        _enhanced_generator = EnhancedQCMGenerator()
    return _enhanced_generator


# Convenience functions
async def generate_questions_enhanced(
    config: GenerationConfig,
    document_ids: Optional[List[str]] = None,
    mode: GenerationMode = GenerationMode.CHUNK_BASED,
    session_id: str = "enhanced",
    **kwargs
) -> Dict[str, Any]:
    """Generate questions using enhanced generator."""
    generator = get_enhanced_qcm_generator()
    return await generator.generate_questions(
        config, document_ids, mode, session_id, **kwargs
    )


async def generate_questions_auto_mode(
    config: GenerationConfig,
    document_ids: Optional[List[str]] = None,
    session_id: str = "auto",
    **kwargs
) -> Dict[str, Any]:
    """Generate questions with automatic mode selection."""
    generator = get_enhanced_qcm_generator()
    return await generator.generate_with_auto_mode(
        config, document_ids, session_id, **kwargs
    )