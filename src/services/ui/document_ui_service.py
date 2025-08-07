"""Document UI Service for Streamlit operations."""

import logging
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional, Any, Dict

logger = logging.getLogger(__name__)


class DocumentUIService:
    """Service for document-related UI operations."""
    
    def __init__(self):
        """Initialize document UI service."""
        pass
    
    def upload_and_process_document(self, file, config: Optional[Any] = None) -> Tuple[str, str, str, Optional[Dict]]:
        """
        Handle document upload and processing.
        
        Args:
            file: Uploaded file from Streamlit
            config: Optional processing configuration
            
        Returns:
            Tuple of (status_message, file_info, themes_info)
        """
        try:
            from src.services.document.document_manager import get_document_manager
            from src.core.config import settings
            import asyncio
            
            if file is None:
                return "❌ Aucun fichier sélectionné", "", "", None

            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Validation du fichier...")
            progress_bar.progress(0.1)

            # Create upload directory
            upload_dir = settings.data_dir / "pdfs"
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            file_path = self._get_unique_filename(upload_dir, file.name)

            # Write file content
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())

            # Validate PDF
            try:
                from src.services.document.pdf_processor import validate_pdf_file
                if not validate_pdf_file(file_path):
                    return "❌ Fichier PDF invalide ou corrompu", "", "", None
            except ImportError:
                if file_path.stat().st_size == 0:
                    return "❌ Fichier vide", "", "", None

            status_text.text("Traitement du document...")
            progress_bar.progress(0.3)

            # Process with DocumentManager
            doc_manager = get_document_manager()
            document = asyncio.run(doc_manager.process_and_store_document(
                file_path,
                config=config,
                store_in_rag=True
            ))

            progress_bar.progress(0.8)
            status_text.text("Finalisation...")

            # Get themes
            themes = doc_manager.get_document_themes(str(document.id))
            theme_names = [theme.theme_name for theme in themes]

            progress_bar.progress(1.0)
            status_text.text("✅ Traitement terminé!")

            # Store metadata for later display
            doc_metadata = getattr(document, 'doc_metadata', None) if hasattr(document, 'doc_metadata') else None

            return (
                f"✅ Document traité avec succès! ID: {document.id}",
                f"Fichier: {document.filename} ({document.total_pages} pages)",
                f"Thèmes détectés: {', '.join(theme_names) if theme_names else 'Aucun'}",
                doc_metadata  # Return metadata as 4th element
            )

        except Exception as e:
            logger.error(f"Document upload error: {e}")
            return f"❌ Erreur lors du traitement: {str(e)}", "", "", None
    
    def _get_unique_filename(self, directory: Path, original_filename: str) -> Path:
        """
        Generate unique filename to avoid conflicts.
        
        Args:
            directory: Target directory
            original_filename: Original filename
            
        Returns:
            Unique file path
        """
        base_path = directory / original_filename
        if not base_path.exists():
            return base_path
        
        # Add suffix if file exists
        name_parts = original_filename.rsplit('.', 1)
        base_name = name_parts[0]
        extension = name_parts[1] if len(name_parts) > 1 else ''
        
        counter = 1
        while True:
            new_filename = f"{base_name}_{counter}.{extension}" if extension else f"{base_name}_{counter}"
            new_path = directory / new_filename
            if not new_path.exists():
                return new_path
            counter += 1
    
    def _display_title_structure(self, metadata: Dict[str, Any]) -> None:
        """Display detected title structure in a user-friendly format."""
        title_structure = metadata.get("title_structure", {})
        
        if not title_structure or title_structure.get("total_titles", 0) == 0:
            st.info("ℹ️ Aucune structure de titres détectée dans ce document.")
            return
        
        st.subheader("📋 Structure des titres détectée")
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Titres détectés", title_structure.get("total_titles", 0))
        with col2:
            st.metric("Pages avec titres", title_structure.get("pages_with_titles", 0))
        with col3:
            level_counts = title_structure.get("level_counts", {})
            max_level = max(len(level_counts.keys()), 1) if level_counts else 1
            st.metric("Niveaux hiérarchiques", max_level)
        
        # Detailed structure
        levels = title_structure.get("levels", {})
        if levels:
            st.markdown("**Hiérarchie des titres :**")
            
            # Display each level
            for level in sorted(levels.keys()):
                titles = levels[level]
                level_num = level.replace('H', '') if 'H' in level else '1'
                try:
                    indent = "  " * (int(level_num) - 1)
                except:
                    indent = ""
                
                with st.expander(f"{indent}📊 {level} - {len(titles)} titre(s)", expanded=int(level_num) <= 2):
                    for title_info in titles:
                        confidence = title_info.get("confidence", 0)
                        confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🟠"
                        
                        col1, col2, col3 = st.columns([6, 1, 1])
                        with col1:
                            st.write(f"{indent}• {title_info.get('text', 'Titre sans nom')}")
                        with col2:
                            st.write(f"Page {title_info.get('page', 'N/A')}")
                        with col3:
                            st.write(f"{confidence_color} {confidence:.2f}")
        
        # Information message about hierarchy
        st.info("💡 Cette structure sera utilisée pour organiser les questions générées selon la hiérarchie du document.")
        
        # Educational hierarchy information
        educational_levels = []
        if levels:
            for level, titles in levels.items():
                for title_info in titles:
                    title_text = title_info.get('text', '').lower()
                    if any(keyword in title_text for keyword in ['parcours', 'module', 'unité']):
                        educational_levels.append((level, title_info.get('text', '')))
        
        if educational_levels:
            st.success(f"🎓 Structure éducative détectée : {len(educational_levels)} éléments pédagogiques (Parcours/Module/Unité)")