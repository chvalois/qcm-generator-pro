"""Streamlit UI Helper Service."""

import logging
from typing import List

logger = logging.getLogger(__name__)


class StreamlitHelpersService:
    """Service providing Streamlit-specific UI helpers."""
    
    def __init__(self):
        """Initialize helpers service."""
        pass
    
    def get_available_example_files(self) -> List[str]:
        """Get list of available few-shot example files."""
        try:
            from src.services.llm.simple_examples_loader import get_examples_loader
            loader = get_examples_loader()
            return loader.list_available_projects()
        except Exception as e:
            logger.error(f"Failed to get example files: {e}")
            return []
    
    def get_available_themes(self) -> List[str]:
        """Get available themes from processed documents."""
        try:
            from src.services.generation.qcm_generator import get_available_themes
            themes_data = get_available_themes()
            
            # Extract theme names from dictionary structure
            if themes_data and isinstance(themes_data[0], dict):
                return [theme.get('name', str(theme)) for theme in themes_data]
            else:
                return themes_data
        except Exception as e:
            logger.error(f"Failed to get themes: {e}")
            return []
    
    def test_llm_connection(self) -> str:
        """Test LLM connection status and format for UI display."""
        try:
            from src.services.llm.llm_manager import test_llm_connection_sync
            
            result = test_llm_connection_sync()
            
            if isinstance(result, dict):
                working_providers = []
                failed_providers = []
                
                for provider, details in result.items():
                    if details.get("status") == "success":
                        model = details.get("model", "")
                        working_providers.append(f"{provider} ({model})")
                    else:
                        error = details.get("error", "Erreur inconnue")
                        failed_providers.append(f"{provider}: {error}")
                
                status_parts = []
                
                if working_providers:
                    status_parts.append(f"✅ **Fonctionnels:** {', '.join(working_providers)}")
                
                if failed_providers:
                    status_parts.append(f"❌ **En erreur:** {'; '.join(failed_providers)}")
                
                if status_parts:
                    return "\\n\\n".join(status_parts)
                else:
                    return "❓ Aucun résultat de test disponible"
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return f"❌ Erreur lors du test de connexion: {str(e)}"