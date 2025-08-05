"""
Simple Examples Loader Service

Loads few-shot examples from JSON files for question generation guidance.
Follows KISS principle - no complex similarity matching, just direct loading.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SimpleExamplesLoader:
    """
    Simple service to load few-shot examples from JSON files.
    
    This service focuses on simplicity - it loads examples from JSON
    and provides them to the prompt builder without complex matching logic.
    """
    
    def __init__(self, examples_dir: Path = Path("data/few_shot_examples")):
        """Initialize with examples directory."""
        self.examples_dir = examples_dir
        self._cache: Dict[str, Dict] = {}
        
    def load_project_examples(self, project_file: str) -> Optional[Dict]:
        """
        Load examples from a project JSON file.
        
        Args:
            project_file: Name of the JSON file (e.g., "python_advanced.json")
            
        Returns:
            Dictionary with project examples or None if not found
        """
        if project_file in self._cache:
            return self._cache[project_file]
            
        file_path = self.examples_dir / project_file
        
        if not file_path.exists():
            logger.warning(f"Examples file not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate basic structure
            if not self._validate_structure(data):
                logger.error(f"Invalid structure in {project_file}")
                return None
                
            self._cache[project_file] = data
            logger.info(f"Loaded {len(data.get('examples', []))} examples from {project_file}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading examples from {project_file}: {e}")
            return None
    
    def get_examples_for_context(
        self, 
        project_file: str,
        question_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        theme: Optional[str] = None,
        max_examples: int = 3
    ) -> List[Dict]:
        """
        Get filtered examples for a specific context.
        
        Args:
            project_file: JSON file containing examples
            question_type: Filter by question type (optional)
            difficulty: Filter by difficulty (optional) 
            theme: Filter by theme (optional)
            max_examples: Maximum number of examples to return
            
        Returns:
            List of example dictionaries
        """
        project_data = self.load_project_examples(project_file)
        if not project_data:
            return []
            
        examples = project_data.get('examples', [])
        
        # Apply filters
        filtered_examples = []
        for example in examples:
            if question_type and example.get('type') != question_type:
                continue
            if difficulty and example.get('difficulty') != difficulty:
                continue
            if theme and theme.lower() not in example.get('theme', '').lower():
                continue
                
            filtered_examples.append(example)
            
        # Return up to max_examples
        return filtered_examples[:max_examples]
    
    def list_available_projects(self) -> List[str]:
        """
        List all available example files.
        
        Returns:
            List of JSON file names
        """
        if not self.examples_dir.exists():
            return []
            
        return [f.name for f in self.examples_dir.glob("*.json")]
    
    def _validate_structure(self, data: Dict) -> bool:
        """
        Validate the basic structure of examples data.
        
        Args:
            data: Loaded JSON data
            
        Returns:
            True if structure is valid
        """
        required_fields = ['project_name', 'examples']
        
        if not all(field in data for field in required_fields):
            return False
            
        examples = data.get('examples', [])
        if not isinstance(examples, list):
            return False
            
        # Validate each example has required fields
        for example in examples:
            required_example_fields = ['question', 'options', 'correct', 'explanation']
            if not all(field in example for field in required_example_fields):
                return False
                
        return True


# Global instance
_examples_loader: Optional[SimpleExamplesLoader] = None


def get_examples_loader() -> SimpleExamplesLoader:
    """Get the global examples loader instance."""
    global _examples_loader
    if _examples_loader is None:
        _examples_loader = SimpleExamplesLoader()
    return _examples_loader