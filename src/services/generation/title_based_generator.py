"""
QCM Generator Pro - Title-Based Question Generation Service

This service generates QCM questions based on document title hierarchy (H1-H4).
It allows users to select specific titles and generate questions from chunks
associated with those titles.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.models.enums import Difficulty, Language, QuestionType
from src.models.schemas import GenerationConfig, QuestionCreate
from src.services.document.document_manager import get_document_manager
from .qcm_generator import QCMGenerator
from src.services.infrastructure.rag_engine import get_rag_engine
from src.services.infrastructure.progress_tracker import update_progress, increment_progress

logger = logging.getLogger(__name__)


@dataclass
class TitleSelectionCriteria:
    """Criteria for selecting titles for question generation."""
    document_id: str
    h1_title: Optional[str] = None
    h2_title: Optional[str] = None
    h3_title: Optional[str] = None
    h4_title: Optional[str] = None
    
    def matches_chunk(self, chunk_hierarchy: Dict[str, Any]) -> bool:
        """Check if a chunk's hierarchy matches this criteria."""
        if not chunk_hierarchy:
            return False
            
        # Check each level - if specified, it must match
        if self.h1_title and chunk_hierarchy.get('h1_title') != self.h1_title:
            return False
        if self.h2_title and chunk_hierarchy.get('h2_title') != self.h2_title:
            return False
        if self.h3_title and chunk_hierarchy.get('h3_title') != self.h3_title:
            return False
        if self.h4_title and chunk_hierarchy.get('h4_title') != self.h4_title:
            return False
            
        return True
    
    def get_title_path(self) -> str:
        """Get a human-readable title path."""
        parts = []
        if self.h1_title:
            parts.append(f"H1: {self.h1_title}")
        if self.h2_title:
            parts.append(f"H2: {self.h2_title}")
        if self.h3_title:
            parts.append(f"H3: {self.h3_title}")
        if self.h4_title:
            parts.append(f"H4: {self.h4_title}")
        return " > ".join(parts) if parts else "Document entier"


class TitleBasedGeneratorError(Exception):
    """Exception raised when title-based generation fails."""
    pass


class TitleBasedQCMGenerator:
    """
    Service for generating QCM questions based on document title hierarchy.
    
    This service allows users to:
    1. Browse available titles in a document
    2. Select specific title levels (H1, H2, H3, H4)
    3. Generate questions from chunks associated with those titles
    """
    
    def __init__(self):
        """Initialize the title-based generator."""
        self.doc_manager = get_document_manager()
        self.qcm_generator = QCMGenerator()
        self.rag_engine = get_rag_engine()
        
    def get_document_title_structure(self, document_id: str) -> Dict[str, Any]:
        """
        Get the complete title structure for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Hierarchical structure of titles with statistics
        """
        try:
            chunks = self.doc_manager.get_document_chunks(document_id, include_titles=True)
            
            if not chunks:
                return {"error": "No chunks found for document"}
            
            # Build hierarchical structure
            structure = {
                "document_id": document_id,
                "total_chunks": len(chunks),
                "h1_titles": {},
                "statistics": {
                    "chunks_with_titles": 0,
                    "chunks_without_titles": 0
                }
            }
            
            for chunk in chunks:
                hierarchy = chunk.get('title_hierarchy', {})
                
                if not hierarchy or not any(hierarchy.values()):
                    structure["statistics"]["chunks_without_titles"] += 1
                    continue
                    
                structure["statistics"]["chunks_with_titles"] += 1
                
                h1 = hierarchy.get('h1_title')
                h2 = hierarchy.get('h2_title')
                h3 = hierarchy.get('h3_title')
                h4 = hierarchy.get('h4_title')
                
                # Build nested structure
                if h1:
                    if h1 not in structure["h1_titles"]:
                        structure["h1_titles"][h1] = {
                            "chunk_count": 0,
                            "h2_titles": {}
                        }
                    structure["h1_titles"][h1]["chunk_count"] += 1
                    
                    if h2:
                        if h2 not in structure["h1_titles"][h1]["h2_titles"]:
                            structure["h1_titles"][h1]["h2_titles"][h2] = {
                                "chunk_count": 0,
                                "h3_titles": {}
                            }
                        structure["h1_titles"][h1]["h2_titles"][h2]["chunk_count"] += 1
                        
                        if h3:
                            if h3 not in structure["h1_titles"][h1]["h2_titles"][h2]["h3_titles"]:
                                structure["h1_titles"][h1]["h2_titles"][h2]["h3_titles"][h3] = {
                                    "chunk_count": 0,
                                    "h4_titles": {}
                                }
                            structure["h1_titles"][h1]["h2_titles"][h2]["h3_titles"][h3]["chunk_count"] += 1
                            
                            if h4:
                                if h4 not in structure["h1_titles"][h1]["h2_titles"][h2]["h3_titles"][h3]["h4_titles"]:
                                    structure["h1_titles"][h1]["h2_titles"][h2]["h3_titles"][h3]["h4_titles"][h4] = {
                                        "chunk_count": 0
                                    }
                                structure["h1_titles"][h1]["h2_titles"][h2]["h3_titles"][h3]["h4_titles"][h4]["chunk_count"] += 1
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to get title structure for document {document_id}: {e}")
            return {"error": str(e)}
    
    def get_chunks_for_title_selection(
        self, 
        criteria: TitleSelectionCriteria
    ) -> List[Dict[str, Any]]:
        """
        Get chunks that match the specified title criteria.
        
        Args:
            criteria: Title selection criteria
            
        Returns:
            List of matching chunks with their content
        """
        try:
            all_chunks = self.doc_manager.get_document_chunks(criteria.document_id, include_titles=True)
            
            matching_chunks = []
            for chunk in all_chunks:
                hierarchy = chunk.get('title_hierarchy', {})
                if criteria.matches_chunk(hierarchy):
                    matching_chunks.append(chunk)
            
            logger.info(f"Found {len(matching_chunks)} chunks matching criteria: {criteria.get_title_path()}")
            return matching_chunks
            
        except Exception as e:
            logger.error(f"Failed to get chunks for title selection: {e}")
            return []
    
    async def generate_questions_from_title(
        self,
        criteria: TitleSelectionCriteria,
        config: GenerationConfig,
        session_id: Optional[str] = None,
        progress_session_id: Optional[str] = None,
        examples_file: Optional[str] = None,
        max_examples: int = 3
    ) -> List[QuestionCreate]:
        """
        Generate QCM questions from chunks matching the title criteria.
        
        Args:
            criteria: Title selection criteria
            config: Generation configuration
            session_id: Optional session ID
            progress_session_id: Optional progress tracking session ID
            
        Returns:
            List of generated questions
        """
        try:
            # Get matching chunks
            matching_chunks = self.get_chunks_for_title_selection(criteria)
            
            if not matching_chunks:
                raise TitleBasedGeneratorError(f"No chunks found for title criteria: {criteria.get_title_path()}")
            
            logger.info(f"Generating {config.num_questions} questions from {len(matching_chunks)} chunks")
            
            # Combine chunk texts for context
            combined_text = "\n\n".join(chunk['chunk_text'] for chunk in matching_chunks)
            
            # Add document to RAG engine with title-specific context
            # Simplify metadata to avoid Pydantic validation issues
            title_context = {
                "title_path": criteria.get_title_path(),
                "chunk_count": str(len(matching_chunks)),
                "source": "title_based_generation",
                "h1_title": criteria.h1_title or "",
                "h2_title": criteria.h2_title or "",
                "h3_title": criteria.h3_title or "",
                "h4_title": criteria.h4_title or ""
            }
            
            # Create a temporary document ID for this title selection
            temp_doc_id = f"{criteria.document_id}_title_{abs(hash(criteria.get_title_path())) % 10000}"
            
            # Add to RAG engine
            self.rag_engine.add_document(
                document_id=temp_doc_id,
                text=combined_text,
                metadata=title_context,
                themes=[criteria.get_title_path()]
            )
            
            # Generate questions using the standard generator
            topic = criteria.get_title_path() or "Contenu sélectionné"
            
            questions = []
            for i in range(config.num_questions):
                try:
                    # Update progress if tracking session exists
                    if progress_session_id:
                        update_progress(
                            progress_session_id,
                            current_step=f"Génération question {i+1}/{config.num_questions} depuis: {topic}"
                        )
                    
                    question = await self.qcm_generator.generate_single_question(
                        topic=topic,
                        config=config,
                        document_ids=[temp_doc_id],
                        session_id=session_id or f"title_gen_{abs(hash(topic)) % 10000}",
                        examples_file=examples_file,
                        max_examples=max_examples
                    )
                    
                    # Add title-specific metadata
                    if hasattr(question, 'generation_params'):
                        question.generation_params['title_selection'] = criteria.get_title_path()
                        question.generation_params['chunk_count'] = len(matching_chunks)
                    
                    questions.append(question)
                    
                    # Increment progress after successful generation
                    if progress_session_id:
                        increment_progress(
                            progress_session_id,
                            current_step=f"Question générée depuis: {topic}"
                        )
                    
                except Exception as e:
                    logger.warning(f"Failed to generate question {i+1} for title '{topic}': {e}")
                    continue
            
            logger.info(f"Successfully generated {len(questions)} questions from title: {criteria.get_title_path()}")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate questions from title: {e}")
            raise TitleBasedGeneratorError(f"Title-based generation failed: {e}")
    
    def get_title_suggestions(self, document_id: str, min_chunks: int = 3) -> List[Dict[str, Any]]:
        """
        Get suggestions for good title selections based on chunk distribution.
        
        Args:
            document_id: Document identifier
            min_chunks: Minimum chunks required for a suggestion
            
        Returns:
            List of suggested title selections with metadata
        """
        try:
            structure = self.get_document_title_structure(document_id)
            suggestions = []
            
            if "error" in structure:
                return suggestions
            
            # Suggest H1 titles with enough content
            for h1_title, h1_data in structure["h1_titles"].items():
                if h1_data["chunk_count"] >= min_chunks:
                    suggestions.append({
                        "level": "H1",
                        "title": h1_title,
                        "chunk_count": h1_data["chunk_count"],
                        "criteria": TitleSelectionCriteria(
                            document_id=document_id,
                            h1_title=h1_title
                        ),
                        "description": f"Tout le contenu sous '{h1_title}'"
                    })
                
                # Suggest H2 titles
                for h2_title, h2_data in h1_data["h2_titles"].items():
                    if h2_data["chunk_count"] >= min_chunks:
                        suggestions.append({
                            "level": "H2",
                            "title": f"{h1_title} > {h2_title}",
                            "chunk_count": h2_data["chunk_count"],
                            "criteria": TitleSelectionCriteria(
                                document_id=document_id,
                                h1_title=h1_title,
                                h2_title=h2_title
                            ),
                            "description": f"Section '{h2_title}' sous '{h1_title}'"
                        })
                    
                    # Suggest H3 titles
                    for h3_title, h3_data in h2_data["h3_titles"].items():
                        if h3_data["chunk_count"] >= min_chunks:
                            suggestions.append({
                                "level": "H3",
                                "title": f"{h1_title} > {h2_title} > {h3_title}",
                                "chunk_count": h3_data["chunk_count"],
                                "criteria": TitleSelectionCriteria(
                                    document_id=document_id,
                                    h1_title=h1_title,
                                    h2_title=h2_title,
                                    h3_title=h3_title
                                ),
                                "description": f"Sous-section '{h3_title}'"
                            })
            
            # Sort by chunk count (descending)
            suggestions.sort(key=lambda x: x["chunk_count"], reverse=True)
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get title suggestions: {e}")
            return []


# Global instance
_title_based_generator: Optional[TitleBasedQCMGenerator] = None


def get_title_based_generator() -> TitleBasedQCMGenerator:
    """Get the global title-based generator instance."""
    global _title_based_generator
    if _title_based_generator is None:
        _title_based_generator = TitleBasedQCMGenerator()
    return _title_based_generator


# Convenience functions
async def generate_questions_from_title(
    document_id: str,
    h1_title: Optional[str] = None,
    h2_title: Optional[str] = None,
    h3_title: Optional[str] = None,
    h4_title: Optional[str] = None,
    config: Optional[GenerationConfig] = None,
    session_id: Optional[str] = None
) -> List[QuestionCreate]:
    """
    Convenience function to generate questions from a title selection.
    
    Args:
        document_id: Document identifier
        h1_title: H1 title to filter by
        h2_title: H2 title to filter by
        h3_title: H3 title to filter by
        h4_title: H4 title to filter by
        config: Generation configuration
        session_id: Optional session ID
        
    Returns:
        List of generated questions
    """
    criteria = TitleSelectionCriteria(
        document_id=document_id,
        h1_title=h1_title,
        h2_title=h2_title,
        h3_title=h3_title,
        h4_title=h4_title
    )
    
    if config is None:
        config = GenerationConfig()
    
    generator = get_title_based_generator()
    return await generator.generate_questions_from_title(criteria, config, session_id)


def get_document_title_structure(document_id: str) -> Dict[str, Any]:
    """Get the title structure for a document."""
    generator = get_title_based_generator()
    return generator.get_document_title_structure(document_id)


def get_title_suggestions(document_id: str, min_chunks: int = 3) -> List[Dict[str, Any]]:
    """Get title suggestions for a document."""
    generator = get_title_based_generator()
    return generator.get_title_suggestions(document_id, min_chunks)