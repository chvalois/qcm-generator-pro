#!/usr/bin/env python3
"""
QCM Generator Pro - Docker Setup Script

This script handles initial setup tasks for Docker deployment including:
- Database initialization
- Ollama model downloads
- Environment validation
- Health checks
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import httpx

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.core.config import settings
from src.models.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DockerSetupError(Exception):
    """Exception raised during Docker setup."""
    pass


class DockerSetupManager:
    """Manages Docker deployment setup tasks."""
    
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.recommended_models = [
            "mistral:7b"
            # "llama3:8b",  # Skip large models in Docker setup
            # "phi3:mini"   # Can be added later if needed
        ]
        
    async def wait_for_ollama(self, timeout: int = 300) -> bool:
        """Wait for Ollama service to be ready."""
        logger.info(f"Waiting for Ollama service at {self.ollama_url}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.ollama_url}/api/tags", timeout=10.0)
                    if response.status_code == 200:
                        logger.info("Ollama service is ready")
                        return True
            except Exception as e:
                logger.debug(f"Ollama not ready yet: {e}")
                
            await asyncio.sleep(5)
            
        logger.error(f"Ollama service not ready after {timeout} seconds")
        return False
        
    async def list_ollama_models(self) -> List[Dict]:
        """List available Ollama models."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags", timeout=30.0)
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
            
    async def pull_ollama_model(self, model_name: str) -> bool:
        """Pull a model from Ollama."""
        logger.info(f"Pulling Ollama model: {model_name}")
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:  # 30 min timeout
                response = await client.post(
                    f"{self.ollama_url}/api/pull",
                    json={"name": model_name},
                    timeout=1800.0
                )
                
                if response.status_code == 200:
                    # Stream the response to show progress
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "status" in data:
                                    logger.info(f"Model {model_name}: {data['status']}")
                                if data.get("status") == "success":
                                    logger.info(f"Successfully pulled model: {model_name}")
                                    return True
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"Failed to pull model {model_name}: HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
            
        return False
        
    async def setup_ollama_models(self) -> bool:
        """Setup recommended Ollama models."""
        logger.info("Setting up Ollama models")
        
        # Wait for Ollama to be ready
        if not await self.wait_for_ollama():
            return False
            
        # Check existing models
        existing_models = await self.list_ollama_models()
        existing_names = [model.get("name", "") for model in existing_models]
        
        logger.info(f"Found {len(existing_models)} existing models: {existing_names}")
        
        # Pull missing recommended models
        success_count = 0
        for model_name in self.recommended_models:
            # Check if model already exists
            if any(model_name in existing for existing in existing_names):
                logger.info(f"Model {model_name} already exists, skipping")
                success_count += 1
                continue
                
            # Pull the model
            if await self.pull_ollama_model(model_name):
                success_count += 1
            else:
                logger.warning(f"Failed to pull model: {model_name}")
                
        logger.info(f"Successfully set up {success_count}/{len(self.recommended_models)} models")
        return success_count > 0
        
    async def initialize_database(self) -> bool:
        """Initialize the application database."""
        logger.info("Initializing database")
        
        try:
            # Create data directories
            data_dir = Path("/app/data")
            for subdir in ["database", "pdfs", "vectorstore", "exports", "cache"]:
                (data_dir / subdir).mkdir(parents=True, exist_ok=True)
                
            # Initialize database
            init_db()
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
            
    async def validate_environment(self) -> bool:
        """Validate Docker environment configuration."""
        logger.info("Validating environment configuration")
        
        required_dirs = [
            "/app/data",
            "/app/logs", 
            "/app/uploads"
        ]
        
        # Check required directories
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                logger.error(f"Required directory missing: {dir_path}")
                return False
            if not os.access(path, os.W_OK):
                logger.error(f"Directory not writable: {dir_path}")
                return False
                
        # Check environment variables
        required_env_vars = ["PYTHONPATH", "DATABASE_URL"]
        for var in required_env_vars:
            if not os.getenv(var):
                logger.warning(f"Environment variable not set: {var}")
                
        # Test database connection
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(settings.database.url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
            
        logger.info("Environment validation successful")
        return True
        
    async def health_check(self) -> Dict[str, bool]:
        """Perform comprehensive health check."""
        logger.info("Performing health check")
        
        results = {
            "environment": False,
            "database": False,
            "ollama": False,
            "models": False
        }
        
        try:
            # Environment check
            results["environment"] = await self.validate_environment()
            
            # Database check
            results["database"] = await self.initialize_database()
            
            # Ollama service check
            results["ollama"] = await self.wait_for_ollama(timeout=60)
            
            # Models check
            if results["ollama"]:
                models = await self.list_ollama_models()
                results["models"] = len(models) > 0
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            
        # Log results
        for component, status in results.items():
            status_str = "✓" if status else "✗"
            logger.info(f"{component.upper()}: {status_str}")
            
        return results
        
    async def full_setup(self) -> bool:
        """Perform full Docker setup."""
        logger.info("Starting full Docker setup")
        
        try:
            # Validate environment
            if not await self.validate_environment():
                raise DockerSetupError("Environment validation failed")
                
            # Initialize database
            if not await self.initialize_database():
                raise DockerSetupError("Database initialization failed")
                
            # Setup Ollama models (optional - can fail)
            models_setup = await self.setup_ollama_models()
            if not models_setup:
                logger.warning("Ollama models setup failed - application will run with fallback")
                
            # Final health check
            health_results = await self.health_check()
            critical_components = ["environment", "database"]
            
            if all(health_results.get(comp, False) for comp in critical_components):
                logger.info("Docker setup completed successfully")
                return True
            else:
                failed_components = [comp for comp in critical_components 
                                   if not health_results.get(comp, False)]
                raise DockerSetupError(f"Critical components failed: {failed_components}")
                
        except Exception as e:
            logger.error(f"Docker setup failed: {e}")
            return False


async def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QCM Generator Pro Docker Setup")
    parser.add_argument("--models-only", action="store_true", 
                       help="Only setup Ollama models")
    parser.add_argument("--db-only", action="store_true", 
                       help="Only initialize database")
    parser.add_argument("--health-check", action="store_true",
                       help="Only perform health check")
    parser.add_argument("--wait-ollama", action="store_true",
                       help="Wait for Ollama service to be ready")
    
    args = parser.parse_args()
    
    setup_manager = DockerSetupManager()
    
    try:
        if args.models_only:
            success = await setup_manager.setup_ollama_models()
        elif args.db_only:
            success = await setup_manager.initialize_database()
        elif args.health_check:
            results = await setup_manager.health_check()
            success = all(results.values())
        elif args.wait_ollama:
            success = await setup_manager.wait_for_ollama()
        else:
            success = await setup_manager.full_setup()
            
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())