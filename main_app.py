#!/usr/bin/env python3
"""
Main Streamlit Application Entry Point

This is the single entry point for the QCM Generator Pro Streamlit application.
It ensures that Streamlit doesn't auto-detect multiple pages.
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run the main application
from ui.streamlit_app import launch_streamlit_app

if __name__ == "__main__":
    launch_streamlit_app()