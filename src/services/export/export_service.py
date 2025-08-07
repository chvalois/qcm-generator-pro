"""Export Service for QCM Generator Pro."""

import csv
import json
import datetime
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

from src.core.config import settings

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting questions to various formats."""
    
    def __init__(self):
        """Initialize export service."""
        self.export_dir = settings.data_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_questions(self, export_format: str) -> Tuple[str, str]:
        """
        Export questions from session state to specified format.
        
        Args:
            export_format: Export format ("CSV (Udemy)" or "JSON")
        
        Returns:
            Tuple of (status_message, download_info)
        """
        try:
            from src.ui.core.session_state import SessionStateManager
            
            questions = SessionStateManager.get_generated_questions()
            
            if not questions:
                return ("❌ Aucune question à exporter", "")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format == "CSV (Udemy)":
                return self._export_csv_udemy(questions, timestamp)
            else:  # JSON
                return self._export_json(questions, timestamp)
                
        except Exception as e:
            logger.error(f"Export error: {e}")
            return (f"❌ Erreur lors de l'export: {e}", "")
    
    def _export_csv_udemy(self, questions: List[Dict[str, Any]], timestamp: str) -> Tuple[str, str]:
        """Export questions in Udemy CSV format."""
        filename = f"qcm_export_{timestamp}.csv"
        export_path = self.export_dir / filename
        
        with open(export_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Question", "Question Type",
                "Answer Option 1", "Explanation 1",
                "Answer Option 2", "Explanation 2", 
                "Answer Option 3", "Explanation 3",
                "Answer Option 4", "Explanation 4",
                "Answer Option 5", "Explanation 5",
                "Answer Option 6", "Explanation 6",
                "Correct Answers", "Overall Explanation", "Domain"
            ])
            
            for question in questions:
                # Handle both dict and object structures
                question_text, options, correct_answers_indices, explanation, theme = self._extract_question_data(question)
                
                # Convert options to text
                option_texts = [self._extract_option_text(opt) for opt in options]
                
                # Convert correct answer indices to 1-based strings
                correct_answers = [str(i + 1) for i in correct_answers_indices]
                if not correct_answers:
                    correct_answers = ["1"]  # Default to first option
                
                # Determine question type
                question_type = "multiple-choice" if len(correct_answers) == 1 else "multi-select"
                
                # Prepare row data (pad options to 6 items)
                row_data = [question_text, question_type]
                
                # Add up to 6 options with empty explanations
                for i in range(6):
                    if i < len(option_texts):
                        row_data.extend([option_texts[i], ""])
                    else:
                        row_data.extend(["", ""])
                
                # Add correct answers, overall explanation, and domain
                row_data.extend([",".join(correct_answers), explanation, theme])
                writer.writerow(row_data)
        
        return self._create_success_response(questions, filename)
    
    def _export_json(self, questions: List[Dict[str, Any]], timestamp: str) -> Tuple[str, str]:
        """Export questions in JSON format."""
        filename = f"qcm_export_{timestamp}.json"
        export_path = self.export_dir / filename
        
        export_data = {
            "export_info": {
                "timestamp": timestamp,
                "questions_count": len(questions),
                "export_format": "JSON"
            },
            "questions": []
        }
        
        for i, question in enumerate(questions):
            if isinstance(question, dict):
                question_data = self._process_dict_question(question)
            else:
                question_data = self._process_object_question(question, i + 1)
            
            export_data["questions"].append(question_data)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return self._create_success_response(questions, filename)
    
    def _extract_question_data(self, question) -> Tuple[str, List, List, str, str]:
        """Extract common question data regardless of format."""
        if isinstance(question, dict):
            return (
                question.get('question_text', ''),
                question.get('options', []),
                question.get('correct_answers', []),
                question.get('explanation', ''),
                question.get('theme', 'General')
            )
        else:
            return (
                getattr(question, 'question_text', ''),
                getattr(question, 'options', []),
                getattr(question, 'correct_answers', []),
                getattr(question, 'explanation', ''),
                getattr(question, 'theme', 'General')
            )
    
    def _extract_option_text(self, opt) -> str:
        """Extract text from option regardless of format."""
        if isinstance(opt, dict):
            return opt.get('text', str(opt))
        else:
            return getattr(opt, 'text', str(opt))
    
    def _process_dict_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Process question that's already a dictionary."""
        question_data = question.copy()
        
        # Convert options if they're objects
        if 'options' in question_data:
            options = []
            for opt in question_data['options']:
                if isinstance(opt, dict):
                    options.append(opt)
                else:
                    options.append({
                        'text': getattr(opt, 'text', str(opt)),
                        'is_correct': getattr(opt, 'is_correct', False)
                    })
            question_data['options'] = options
        
        return question_data
    
    def _process_object_question(self, question, question_id: int) -> Dict[str, Any]:
        """Process question that's an object."""
        raw_options = getattr(question, 'options', [])
        options = []
        for opt in raw_options:
            if isinstance(opt, dict):
                options.append(opt)
            else:
                options.append({
                    'text': getattr(opt, 'text', str(opt)),
                    'is_correct': getattr(opt, 'is_correct', False)
                })
        
        # Convert enums to strings if needed
        question_type = getattr(question, 'question_type', 'multiple-choice')
        if hasattr(question_type, 'value'):
            question_type = question_type.value
        
        difficulty = getattr(question, 'difficulty', 'medium')
        if hasattr(difficulty, 'value'):
            difficulty = difficulty.value
        
        return {
            "id": question_id,
            "question_text": getattr(question, 'question_text', ''),
            "options": options,
            "correct_answers": getattr(question, 'correct_answers', []),
            "explanation": getattr(question, 'explanation', ''),
            "difficulty": difficulty,
            "theme": getattr(question, 'theme', 'General'),
            "question_type": question_type
        }
    
    def _create_success_response(self, questions: List, filename: str) -> Tuple[str, str]:
        """Create standardized success response."""
        return (
            f"✅ Export réussi ! ({len(questions)} questions)",
            f"📁 Fichier créé : {filename}"
        )