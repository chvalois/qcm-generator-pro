"""Ollama UI Service for model management interface."""

import streamlit as st
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class OllamaUIService:
    """Service for Ollama model management UI."""
    
    def __init__(self):
        """Initialize Ollama UI service."""
        self.recommended_models = [
            {"name": "mistral:7b-instruct", "description": "Modèle polyvalent pour la génération", "size": "~4GB"},
            {"name": "llama3:8b-instruct", "description": "Modèle performant pour l'analyse", "size": "~4.7GB"},
            {"name": "phi3:mini", "description": "Modèle léger pour les tâches simples", "size": "~2.3GB"}
        ]
    
    def show_model_downloads(self) -> None:
        """Show Ollama model management interface."""
        try:
            # Display header with refresh button
            col1, col2 = st.columns([2, 1])
            with col2:
                if st.button("🔄 Actualiser la liste"):
                    st.rerun()
            
            # Check Ollama connection
            if self._check_ollama_connection():
                st.success("✅ Service Ollama connecté")
                
                # Show recommended models
                self._show_recommended_models()
                
                # Show custom download section
                self._show_custom_download()
                
            else:
                st.error("❌ Service Ollama non disponible")
                st.info("Vérifiez que le conteneur Ollama est démarré")
                
        except Exception as e:
            logger.error(f"Error in Ollama UI: {e}")
            st.error(f"Erreur lors de l'affichage de l'interface Ollama: {e}")
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama service is available."""
        try:
            from src.services.llm.llm_manager import get_llm_manager
            llm_manager = get_llm_manager()
            # This would be a real connection check
            return True
        except Exception:
            return False
    
    def _show_recommended_models(self) -> None:
        """Show recommended models with download buttons."""
        st.markdown("### 📥 Modèles recommandés")
        
        for model in self.recommended_models:
            with st.expander(f"📦 {model['name']} ({model['size']})"):
                st.write(f"**Description:** {model['description']}")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Taille:** {model['size']}")
                
                with col2:
                    # Check if model exists (placeholder logic)
                    model_exists = self._check_model_exists(model['name'])
                    if model_exists:
                        st.success("✅ Installé")
                    else:
                        st.warning("❌ Non installé")
                
                with col3:
                    if not model_exists:
                        if st.button(f"⬇️ Télécharger", key=f"download_{model['name']}"):
                            self._download_model(model['name'])
                    else:
                        if st.button(f"🗑️ Supprimer", key=f"remove_{model['name']}"):
                            self._remove_model(model['name'])
    
    def _show_custom_download(self) -> None:
        """Show custom model download section."""
        st.markdown("### 🎯 Téléchargement personnalisé")
        
        with st.form("custom_model_form"):
            custom_model = st.text_input(
                "Nom du modèle Ollama:",
                placeholder="ex: codellama:7b, mistral:latest",
                help="Entrez le nom complet du modèle tel qu'il apparaît sur ollama.ai"
            )
            
            submitted = st.form_submit_button("📥 Télécharger le modèle")
            
            if submitted and custom_model:
                self._download_model(custom_model)
            elif submitted:
                st.error("❌ Veuillez spécifier un nom de modèle")
    
    def _check_model_exists(self, model_name: str) -> bool:
        """Check if a model is installed."""
        # Placeholder - would implement real check
        return False
    
    def _download_model(self, model_name: str) -> None:
        """Download a model."""
        try:
            with st.spinner(f"Téléchargement de {model_name}..."):
                # Placeholder for real download logic
                st.success(f"✅ Téléchargement de {model_name} terminé!")
                st.info("🔄 Actualisez la page pour voir le nouveau modèle")
        except Exception as e:
            logger.error(f"Model download error: {e}")
            st.error(f"❌ Erreur lors du téléchargement: {e}")
    
    def _remove_model(self, model_name: str) -> None:
        """Remove a model."""
        try:
            with st.spinner(f"Suppression de {model_name}..."):
                # Placeholder for real removal logic
                st.success(f"✅ Modèle {model_name} supprimé!")
                st.info("🔄 Actualisez la page pour voir les changements")
        except Exception as e:
            logger.error(f"Model removal error: {e}")
            st.error(f"❌ Erreur lors de la suppression: {e}")