"""
Streamlit Pages

This module contains the main page components for different application sections.
"""

from .document_upload import DocumentUploadPage
from .document_manager import DocumentManagerPage
from .qcm_generation import QCMGenerationPage
from .title_generation import TitleGenerationPage
from .export_page import ExportPage
from .system_page import SystemPage

__all__ = [
    "DocumentUploadPage",
    "DocumentManagerPage", 
    "QCMGenerationPage",
    "TitleGenerationPage",
    "ExportPage",
    "SystemPage",
]