"""
QCM Generator Pro - Export API Routes

This module provides FastAPI routes for exporting QCM questions
in various formats, including CSV for Udemy.
"""

import csv
import io
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from ...core.config import settings
from ...models.database import GenerationSession as GenerationSessionModel
from ...models.database import Question as QuestionModel
from ...models.enums import ExportFormat, ValidationStatus
from ...models.schemas import ExportRequest, SuccessResponse
from ..dependencies import get_db_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export"])


class ExportService:
    """Service for handling question exports."""
    
    @staticmethod
    def format_question_for_udemy(question: QuestionModel) -> Dict[str, Any]:
        """
        Format a question for Udemy CSV export (v2 format).
        
        Args:
            question: Question model to format
            
        Returns:
            Formatted question data for Udemy v2 template
        """
        # Udemy v2 format with up to 6 options and individual explanations
        formatted = {
            "Question": question.question_text,
            "Question Type": "multiple-choice" if question.question_type.value == "multiple-choice" else "multi-select",
            "Overall Explanation": question.explanation or "",
            "Domain": question.theme or "General"
        }
        
        # Add options and explanations (up to 6 options)
        options = question.options or []
        for i in range(6):  # Udemy v2 supports up to 6 options
            option_num = i + 1
            if i < len(options):
                # Extract option text - handle both string and dict formats
                if isinstance(options[i], dict):
                    option_text = options[i].get("text", "")
                    option_explanation = options[i].get("explanation", "")
                else:
                    option_text = str(options[i])
                    option_explanation = ""
                    
                formatted[f"Answer Option {option_num}"] = option_text
                formatted[f"Explanation {option_num}"] = option_explanation
            else:
                formatted[f"Answer Option {option_num}"] = ""
                formatted[f"Explanation {option_num}"] = ""
                
        # Format correct answers (Udemy uses 1-based indexing)
        if question.correct_answers:
            correct_indices = [str(idx + 1) for idx in question.correct_answers]
            formatted["Correct Answers"] = ",".join(correct_indices)
        else:
            formatted["Correct Answers"] = "1"  # Default fallback
            
        return formatted
        
    @staticmethod
    def create_udemy_csv(questions: list[QuestionModel]) -> str:
        """
        Create CSV content in Udemy v2 format.
        
        Args:
            questions: List of questions to export
            
        Returns:
            CSV content as string
        """
        output = io.StringIO()
        fieldnames = [
            "Question", "Question Type",
            "Answer Option 1", "Explanation 1",
            "Answer Option 2", "Explanation 2", 
            "Answer Option 3", "Explanation 3",
            "Answer Option 4", "Explanation 4",
            "Answer Option 5", "Explanation 5",
            "Answer Option 6", "Explanation 6",
            "Correct Answers", "Overall Explanation", "Domain"
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        for question in questions:
            formatted_question = ExportService.format_question_for_udemy(question)
            writer.writerow(formatted_question)
            
        return output.getvalue()
        
    @staticmethod
    def create_json_export(questions: list[QuestionModel]) -> Dict[str, Any]:
        """
        Create JSON export of questions.
        
        Args:
            questions: List of questions to export
            
        Returns:
            JSON-serializable dictionary
        """
        export_data = {
            "export_info": {
                "format": "json",
                "question_count": len(questions),
                "exported_at": "timestamp_placeholder"
            },
            "questions": []
        }
        
        for question in questions:
            question_data = {
                "id": question.id,
                "question_text": question.question_text,
                "question_type": question.question_type.value,
                "language": question.language.value,
                "difficulty": question.difficulty.value,
                "theme": question.theme,
                "options": question.options,
                "correct_answers": question.correct_answers,
                "explanation": question.explanation,
                "validation_status": question.validation_status.value,
                "metadata": question.question_metadata
            }
            export_data["questions"].append(question_data)
            
        return export_data


@router.post("/{session_id}", response_model=Dict[str, Any])
async def export_session(
    session_id: str,
    export_request: ExportRequest,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Export questions from a generation session.
    
    Args:
        session_id: Generation session ID
        export_request: Export configuration
        db: Database session
        
    Returns:
        Export information with download link
        
    Raises:
        HTTPException: If session not found or export fails
    """
    logger.info(f"Exporting session {session_id} in {export_request.format.value} format")
    
    try:
        # Verify session exists
        session = db.query(GenerationSessionModel).filter(
            GenerationSessionModel.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Generation session {session_id} not found"
            )
            
        # Get questions with optional filtering
        query = db.query(QuestionModel).filter(QuestionModel.session_id == session_id)
        
        # Apply validation status filter
        if export_request.validation_status_filter:
            query = query.filter(
                QuestionModel.validation_status.in_(export_request.validation_status_filter)
            )
            
        # Apply theme filter
        if export_request.themes_filter:
            query = query.filter(QuestionModel.theme.in_(export_request.themes_filter))
            
        questions = query.all()
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No questions found matching the export criteria"
            )
            
        # Create export directory
        export_dir = settings.data_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate export based on format
        export_filename = f"{session_id}_{export_request.format.value}"
        
        if export_request.format == ExportFormat.UDEMY_CSV:
            # Create Udemy CSV export
            csv_content = ExportService.create_udemy_csv(questions)
            export_filename += ".csv"
            export_path = export_dir / export_filename
            
            with open(export_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_content)
                
        elif export_request.format == ExportFormat.JSON:
            # Create JSON export
            import json
            json_content = ExportService.create_json_export(questions)
            export_filename += ".json"
            export_path = export_dir / export_filename
            
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False)
                
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Export format {export_request.format.value} not supported"
            )
            
        # Prepare response
        response = {
            "session_id": session_id,
            "export_format": export_request.format.value,
            "questions_exported": len(questions),
            "export_filename": export_filename,
            "export_path": str(export_path),
            "download_url": f"/export/download/{export_filename}",
            "export_stats": {
                "total_questions": len(questions),
                "by_validation_status": {
                    status.value: len([q for q in questions if q.validation_status == status])
                    for status in ValidationStatus
                },
                "by_difficulty": {
                    "easy": len([q for q in questions if q.difficulty.value == "easy"]),
                    "medium": len([q for q in questions if q.difficulty.value == "medium"]),
                    "hard": len([q for q in questions if q.difficulty.value == "hard"])
                },
                "themes": list(set(q.theme for q in questions if q.theme))
            }
        }
        
        logger.info(f"Export completed: {export_filename} ({len(questions)} questions)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_export(filename: str) -> Response:
    """
    Download an exported file.
    
    Args:
        filename: Name of the exported file
        
    Returns:
        File download response
        
    Raises:
        HTTPException: If file not found
    """
    logger.debug(f"Downloading export file: {filename}")
    
    try:
        export_path = settings.data_dir / "exports" / filename
        
        if not export_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Export file {filename} not found"
            )
            
        # Read file content
        with open(export_path, "rb") as f:
            content = f.read()
            
        # Determine content type
        if filename.endswith(".csv"):
            media_type = "text/csv"
        elif filename.endswith(".json"):
            media_type = "application/json"
        else:
            media_type = "application/octet-stream"
            
        # Create response with appropriate headers
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(content))
            }
        )
        
        logger.debug(f"File download served: {filename} ({len(content)} bytes)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed for file {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File download failed"
        )


@router.get("/formats", response_model=Dict[str, Any])
async def get_export_formats() -> Dict[str, Any]:
    """
    Get available export formats and their descriptions.
    
    Returns:
        Available export formats
    """
    formats = {
        "udemy_csv": {
            "name": "Udemy CSV v2",
            "description": "CSV format compatible with Udemy Practice Test bulk upload template v2",
            "file_extension": ".csv",
            "features": [
                "Multiple choice questions",
                "Multi-select questions", 
                "Up to 6 answer options per question",
                "Individual explanations per option",
                "Overall explanations included",
                "Domain/theme classification",
                "Ready for Udemy Practice Test upload"
            ]
        },
        "json": {
            "name": "JSON Export",
            "description": "Complete question data in JSON format",
            "file_extension": ".json",
            "features": [
                "Full question metadata",
                "Validation information",
                "Theme classification",
                "Machine-readable format"
            ]
        }
    }
    
    return {
        "available_formats": formats,
        "supported_extensions": [".csv", ".json"],
        "notes": [
            "Udemy CSV format is optimized for direct upload to Udemy courses",
            "JSON format includes complete metadata for further processing",
            "Export filtering available by validation status and themes"
        ]
    }


@router.delete("/cleanup", response_model=SuccessResponse)
async def cleanup_old_exports(
    days_old: int = 7
) -> SuccessResponse:
    """
    Clean up old export files.
    
    Args:
        days_old: Delete files older than this many days
        
    Returns:
        Cleanup results
    """
    logger.info(f"Cleaning up export files older than {days_old} days")
    
    try:
        from datetime import datetime, timedelta
        
        export_dir = settings.data_dir / "exports"
        if not export_dir.exists():
            return SuccessResponse(
                message="No export directory found",
                details={"files_deleted": 0}
            )
            
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for file_path in export_dir.iterdir():
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old export file: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path.name}: {e}")
                        
        return SuccessResponse(
            message=f"Cleanup completed: {deleted_count} files deleted",
            details={
                "files_deleted": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "days_old": days_old
            }
        )
        
    except Exception as e:
        logger.error(f"Export cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cleanup operation failed"
        )