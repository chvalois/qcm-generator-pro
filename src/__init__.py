"""
QCM Generator Pro - Local multilingual QCM generation from PDF documents.

A comprehensive system for generating Multiple Choice Questions (QCM) from PDF
documents using local LLM models with RAG-based intelligent question generation.
"""

import warnings

# Suppress cryptography deprecation warnings from pypdf
warnings.filterwarnings("ignore", message=".*ARC4.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*cryptography.*", category=DeprecationWarning)

# Suppress other common deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pypdf")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cryptography")

__version__ = "0.1.0"
__author__ = "QCM Generator Team"
__email__ = "dev@qcmgenerator.local"
