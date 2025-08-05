"""Question generation services.

This module contains services responsible for QCM generation,
including main generation logic, workflow management, and prompt building.
"""

from .qcm_generator import get_qcm_generator, generate_progressive_qcm
from .chunk_based_generator import get_chunk_based_generator
from .title_based_generator import get_title_based_generator
from .enhanced_qcm_generator import EnhancedQCMGenerator, GenerationMode
from .progressive_workflow import get_progressive_workflow_manager
from .question_prompt_builder import get_question_prompt_builder
from .question_parser import get_question_parser
from .question_selection import get_question_selector

__all__ = [
    # Main Generation
    "get_qcm_generator",
    "generate_progressive_qcm",
    # Specialized Generators
    "get_chunk_based_generator",
    "get_title_based_generator",
    "EnhancedQCMGenerator",
    "GenerationMode",
    # Workflow Management
    "get_progressive_workflow_manager",
    # Supporting Services
    "get_question_prompt_builder",
    "get_question_parser",
    "get_question_selector",
]