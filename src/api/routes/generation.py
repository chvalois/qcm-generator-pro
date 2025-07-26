"""
QCM Generator Pro - QCM Generation API Routes

This module provides FastAPI routes for QCM generation using the
progressive workflow (1 → 5 → all questions).
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...models.database import Document as DocumentModel
from ...models.database import GenerationSession as GenerationSessionModel
from ...models.database import Question as QuestionModel
from ...models.enums import ValidationStatus
from ...models.schemas import (
    GenerationConfig,
    GenerationSessionCreate,
    QuestionCreate,
    QuestionResponse,
    SuccessResponse,
)
from ...services.qcm_generator import QCMGenerator
from ...services.rag_engine import SimpleRAGEngine
from ...services.validator import QuestionValidator
from ...services.title_based_generator import get_title_based_generator, TitleSelectionCriteria
from ..dependencies import (
    get_db_session,
    get_qcm_generator_service,
    get_rag_engine_service,
    get_validator_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generation", tags=["generation"])


@router.post("/start", response_model=Dict[str, Any])
async def start_generation_session(
    request: GenerationSessionCreate,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Start a new QCM generation session.
    
    This endpoint initializes a session and returns session info.
    For multi-question requests, use the progressive generation endpoints.
    
    Args:
        request: Generation session configuration
        db: Database session
        
    Returns:
        Session information with next steps
    """
    logger.info(f"Starting QCM generation session for {len(request.document_ids)} documents")
    
    try:
        # Validate documents exist
        documents = db.query(DocumentModel).filter(
            DocumentModel.id.in_(request.document_ids)
        ).all()
        
        if len(documents) != len(request.document_ids):
            found_ids = [doc.id for doc in documents]
            missing_ids = [doc_id for doc_id in request.document_ids if doc_id not in found_ids]
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Documents not found: {missing_ids}"
            )
            
        # Create generation session in database
        session_data = GenerationSessionModel(
            session_id=request.session_id,
            document_id=request.document_ids[0],  # Use first document for now
            total_questions_requested=request.config.num_questions,
            language=request.config.language.value,
            model_used="mistral-local",  # Default model
            generation_config=request.config.model_dump(),
            current_batch=0,
            batch_sizes=[1, 5, -1],  # Progressive batch sizes
            batches_completed=0,
            questions_generated=0,
            status="initialized"
        )
        
        db.add(session_data)
        db.commit()
        db.refresh(session_data)
        
        # Extract topics from documents/themes for generation
        topics = []
        if request.config.themes_filter:
            topics.extend(request.config.themes_filter)
        else:
            # Extract topics from document themes or use document titles
            for doc in documents:
                if hasattr(doc, 'themes') and doc.themes:
                    topics.extend([theme.theme_name for theme in doc.themes])
                else:
                    # Use document filename as fallback topic
                    topic = doc.filename.replace('.pdf', '').replace('_', ' ')
                    topics.append(topic)
                    
        if not topics:
            topics = ["Contenu général"]  # Fallback topic
            
        logger.info(f"Generation topics: {topics[:5]}...")  # Log first 5 topics
        
        # Determine workflow type
        requires_validation = request.config.num_questions > 1
        next_action = "generate_phase_1" if requires_validation else "generate_single"
        
        response = {
            "session_id": request.session_id,
            "status": "initialized",
            "requires_progressive_validation": requires_validation,
            "next_action": next_action,
            "total_questions_requested": request.config.num_questions,
            "topics": topics,
            "session_info": {
                "id": session_data.id,
                "created_at": session_data.created_at,
                "total_questions_requested": session_data.total_questions_requested,
                "current_batch": session_data.current_batch
            }
        }
        
        logger.info(f"QCM generation session initialized: {request.session_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session initialization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session initialization failed: {str(e)}"
        )

@router.post("/sessions/{session_id}/generate-single", response_model=Dict[str, Any])
async def generate_single_question(
    session_id: str,
    db: Session = Depends(get_db_session),
    qcm_generator: QCMGenerator = Depends(get_qcm_generator_service)
) -> Dict[str, Any]:
    """
    Generate a single question (for requests of 1 question only).
    
    Args:
        session_id: Generation session ID
        db: Database session
        qcm_generator: QCM generator service
        
    Returns:
        Generated question
    """
    try:
        # Get session
        session = db.query(GenerationSessionModel).filter(
            GenerationSessionModel.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id} not found")
            
        config_dict = dict(session.generation_config)
        config = GenerationConfig(**config_dict)
        
        if config.num_questions != 1:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "This endpoint is for single questions only")
        
        # Generate single question
        question = await qcm_generator.generate_single_question(
            topic="Contenu général",
            config=config,
            document_ids=[str(doc_id) for doc_id in session.document_ids],
            session_id=session_id
        )
        
        # Store in database
        db_question = QuestionModel(
            session_id=session_id,
            question_text=question.question_text,
            question_type=question.question_type,
            language=question.language,
            difficulty=question.difficulty,
            theme=getattr(question, 'theme', 'Contenu général'),
            options=question.options,
            correct_answers=getattr(question, 'correct_answers', []),
            explanation=question.explanation,
            validation_status=ValidationStatus.APPROVED,
            question_metadata=getattr(question, 'question_metadata', {})
        )
        db.add(db_question)
        
        # Update session
        session.status = "completed"
        session.questions_generated = 1
        db.commit()
        
        return {
            "session_id": session_id,
            "status": "completed",
            "question": {
                "id": db_question.id,
                "question_text": db_question.question_text,
                "question_type": db_question.question_type,
                "options": db_question.options,
                "correct_answers": db_question.correct_answers,
                "explanation": db_question.explanation
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single question generation failed: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@router.post("/sessions/{session_id}/generate-phase/{phase_number}", response_model=Dict[str, Any])
async def generate_phase(
    session_id: str,
    phase_number: int,
    db: Session = Depends(get_db_session),
    qcm_generator: QCMGenerator = Depends(get_qcm_generator_service)
) -> Dict[str, Any]:
    """
    Generate questions for a specific phase of the progressive workflow.
    
    Args:
        session_id: Generation session ID
        phase_number: Phase number (1, 2, or 3)
        db: Database session
        qcm_generator: QCM generator service
        
    Returns:
        Generated questions for this phase
    """
    try:
        # Get session
        session = db.query(GenerationSessionModel).filter(
            GenerationSessionModel.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id} not found")
            
        config_dict = dict(session.generation_config)
        config = GenerationConfig(**config_dict)
        
        # Determine batch size for this phase
        total_requested = config.num_questions
        already_generated = session.questions_generated or 0
        remaining = total_requested - already_generated
        
        if phase_number == 1:
            batch_size = 1
        elif phase_number == 2:
            batch_size = min(5, remaining)
        else:  # phase 3
            batch_size = remaining
            
        if batch_size <= 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No more questions to generate")
        
        # Extract topics
        topics = ["Contenu général"]  # Simplified for now
        
        # Generate questions for this phase
        questions = await qcm_generator.generate_questions_batch(
            topics=topics,
            config=config,
            document_ids=[str(doc_id) for doc_id in session.document_ids],
            batch_size=batch_size,
            session_id=session_id
        )
        
        # Store questions in database
        stored_questions = []
        for question in questions:
            db_question = QuestionModel(
                session_id=session_id,
                question_text=question.question_text,
                question_type=question.question_type,
                language=question.language,
                difficulty=question.difficulty,
                theme=getattr(question, 'theme', 'Contenu général'),
                options=question.options,
                correct_answers=getattr(question, 'correct_answers', []),
                explanation=question.explanation,
                validation_status=ValidationStatus.PENDING,
                question_metadata=getattr(question, 'question_metadata', {})
            )
            db.add(db_question)
            stored_questions.append(db_question)
        
        # Update session
        session.current_batch = phase_number
        session.questions_generated = already_generated + len(questions)
        session.status = f"phase_{phase_number}_completed"
        db.commit()
        
        # Prepare response
        questions_data = []
        for q in stored_questions:
            questions_data.append({
                "id": q.id,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "options": q.options,
                "correct_answers": q.correct_answers,
                "explanation": q.explanation
            })
        
        # Determine next action
        next_phase = None
        is_completed = session.questions_generated >= total_requested
        
        if not is_completed:
            if phase_number == 1:
                next_phase = 2
            elif phase_number == 2 and session.questions_generated < total_requested:
                next_phase = 3
        
        return {
            "session_id": session_id,
            "phase": phase_number,
            "questions_generated": len(questions),
            "total_generated": session.questions_generated,
            "total_requested": total_requested,
            "questions": questions_data,
            "is_completed": is_completed,
            "next_phase": next_phase,
            "status": "awaiting_user_validation" if not is_completed else "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Phase {phase_number} generation failed: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@router.post("/sessions/{session_id}/approve-phase/{phase_number}", response_model=Dict[str, Any])
async def approve_phase(
    session_id: str,
    phase_number: int,
    approved: bool = True,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Approve or reject a phase of generation.
    
    Args:
        session_id: Generation session ID
        phase_number: Phase number that was validated
        approved: Whether the phase is approved
        db: Database session
        
    Returns:
        Approval status and next steps
    """
    try:
        # Get session
        session = db.query(GenerationSessionModel).filter(
            GenerationSessionModel.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id} not found")
        
        if approved:
            # Update question validation status for this phase
            recent_questions = db.query(QuestionModel).filter(
                QuestionModel.session_id == session_id,
                QuestionModel.validation_status == ValidationStatus.PENDING
            ).all()
            
            for question in recent_questions:
                question.validation_status = ValidationStatus.APPROVED
            
            # Determine next action
            config_dict = dict(session.config)
            config = GenerationConfig(**config_dict)
            total_requested = config.num_questions
            
            if session.questions_generated >= total_requested:
                session.status = "completed"
                next_action = "completed"
            else:
                next_phase = phase_number + 1
                next_action = f"generate_phase_{next_phase}"
            
            db.commit()
            
            return {
                "session_id": session_id,
                "phase": phase_number,
                "approved": True,
                "next_action": next_action,
                "questions_approved": len(recent_questions),
                "total_generated": session.questions_generated,
                "total_requested": total_requested
            }
        else:
            # User rejected this phase - stop generation
            session.status = f"stopped_at_phase_{phase_number}"
            
            # Mark pending questions as rejected
            recent_questions = db.query(QuestionModel).filter(
                QuestionModel.session_id == session_id,
                QuestionModel.validation_status == ValidationStatus.PENDING
            ).all()
            
            for question in recent_questions:
                question.validation_status = ValidationStatus.REJECTED
            
            db.commit()
            
            return {
                "session_id": session_id,
                "phase": phase_number,
                "approved": False,
                "next_action": "stopped",
                "questions_rejected": len(recent_questions),
                "final_count": session.questions_generated - len(recent_questions)
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Phase approval failed: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@router.get("/sessions/{session_id}/questions", response_model=List[QuestionResponse])
async def get_session_questions(
    session_id: str,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db_session)
) -> List[QuestionResponse]:
    """
    Get questions from a generation session.
    
    Args:
        session_id: Generation session ID
        skip: Number of questions to skip
        limit: Maximum number of questions to return
        db: Database session
        
    Returns:
        List of questions
        
    Raises:
        HTTPException: If session not found
    """
    logger.debug(f"Getting questions for session: {session_id}")
    
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
            
        # Get questions
        questions = db.query(QuestionModel).filter(
            QuestionModel.session_id == session_id
        ).offset(skip).limit(limit).all()
        
        response_questions = []
        for question in questions:
            response = QuestionResponse(
                id=question.id,
                session_id=question.session_id,
                question_text=question.question_text,
                question_type=question.question_type,
                language=question.language,
                difficulty=question.difficulty,
                theme=question.theme,
                options=question.options,
                correct_answers=question.correct_answers,
                explanation=question.explanation,
                validation_status=question.validation_status,
                question_metadata=question.question_metadata or {},
                created_at=question.created_at
            )
            response_questions.append(response)
            
        logger.debug(f"Retrieved {len(response_questions)} questions for session {session_id}")
        return response_questions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get questions for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve questions"
        )


@router.post("/questions/{question_id}/validate", response_model=Dict[str, Any])
async def validate_question(
    question_id: int,
    db: Session = Depends(get_db_session),
    validator: QuestionValidator = Depends(get_validator_service)
) -> Dict[str, Any]:
    """
    Validate a specific question.
    
    Args:
        question_id: Question ID to validate
        db: Database session
        validator: Question validator service
        
    Returns:
        Validation results
        
    Raises:
        HTTPException: If question not found or validation fails
    """
    logger.debug(f"Validating question: {question_id}")
    
    try:
        # Get question from database
        question = db.query(QuestionModel).filter(QuestionModel.id == question_id).first()
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )
            
        # Convert to QuestionCreate for validation
        question_data = QuestionCreate(
            question_text=question.question_text,
            question_type=question.question_type,
            language=question.language,
            difficulty=question.difficulty,
            theme=question.theme,
            options=question.options,
            correct_answers=question.correct_answers,
            explanation=question.explanation,
            validation_status=question.validation_status,
            question_metadata=question.question_metadata or {}
        )
        
        # Perform validation
        validation_result = validator.validate_question(question_data)
        
        # Update question validation status in database
        if validation_result.is_valid and validation_result.score >= 0.8:
            question.validation_status = ValidationStatus.APPROVED
        elif validation_result.is_valid and validation_result.score >= 0.6:
            question.validation_status = ValidationStatus.PENDING
        else:
            question.validation_status = ValidationStatus.REJECTED
            
        db.commit()
        
        response = {
            "question_id": question_id,
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "score": validation_result.score,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "validation_type": validation_result.validation_type,
                "details": validation_result.details
            },
            "updated_status": question.validation_status.value
        }
        
        logger.debug(f"Question {question_id} validated: score={validation_result.score:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate question {question_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Question validation failed"
        )


@router.post("/questions/batch-validate", response_model=Dict[str, Any])
async def batch_validate_questions(
    session_id: str,
    db: Session = Depends(get_db_session),
    validator: QuestionValidator = Depends(get_validator_service)
) -> Dict[str, Any]:
    """
    Batch validate all questions in a session.
    
    Args:
        session_id: Generation session ID
        db: Database session
        validator: Question validator service
        
    Returns:
        Batch validation results
        
    Raises:
        HTTPException: If session not found or validation fails
    """
    logger.info(f"Batch validating questions for session: {session_id}")
    
    try:
        # Get all questions for the session
        questions = db.query(QuestionModel).filter(
            QuestionModel.session_id == session_id
        ).all()
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No questions found for session {session_id}"
            )
            
        # Convert to QuestionCreate objects
        question_data_list = []
        for question in questions:
            question_data = QuestionCreate(
                question_text=question.question_text,
                question_type=question.question_type,
                language=question.language,
                difficulty=question.difficulty,
                theme=question.theme,
                options=question.options,
                correct_answers=question.correct_answers,
                explanation=question.explanation,
                validation_status=question.validation_status,
                question_metadata=question.question_metadata or {}
            )
            question_data_list.append(question_data)
            
        # Perform batch validation
        validation_results = validator.validate_question_batch(question_data_list)
        
        # Update question statuses in database
        updated_count = 0
        validation_summary = {"validated": 0, "pending": 0, "rejected": 0}
        
        for question, validation_result in zip(questions, validation_results):
            old_status = question.validation_status
            
            if validation_result.is_valid and validation_result.score >= 0.8:
                question.validation_status = ValidationStatus.APPROVED
                validation_summary["validated"] += 1
            elif validation_result.is_valid and validation_result.score >= 0.6:
                question.validation_status = ValidationStatus.PENDING
                validation_summary["pending"] += 1
            else:
                question.validation_status = ValidationStatus.REJECTED
                validation_summary["rejected"] += 1
                
            if question.validation_status != old_status:
                updated_count += 1
                
        db.commit()
        
        # Calculate overall statistics
        total_score = sum(result.score for result in validation_results)
        avg_score = total_score / len(validation_results) if validation_results else 0
        valid_count = sum(1 for result in validation_results if result.is_valid)
        
        response = {
            "session_id": session_id,
            "questions_validated": len(questions),
            "questions_updated": updated_count,
            "validation_summary": validation_summary,
            "overall_stats": {
                "average_score": round(avg_score, 3),
                "valid_questions": valid_count,
                "valid_percentage": round((valid_count / len(questions)) * 100, 1) if questions else 0
            },
            "detailed_results": [
                {
                    "question_id": question.id,
                    "score": result.score,
                    "is_valid": result.is_valid,
                    "status": question.validation_status.value,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
                for question, result in zip(questions, validation_results)
            ]
        }
        
        logger.info(f"Batch validation completed: {len(questions)} questions, avg score: {avg_score:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch validation failed for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch validation failed"
        )


@router.get("/sessions/{session_id}/status", response_model=Dict[str, Any])
async def get_generation_status(
    session_id: str,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get the status of a generation session.
    
    Args:
        session_id: Generation session ID
        db: Database session
        
    Returns:
        Session status information
        
    Raises:
        HTTPException: If session not found
    """
    logger.debug(f"Getting status for session: {session_id}")
    
    try:
        session = db.query(GenerationSessionModel).filter(
            GenerationSessionModel.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Generation session {session_id} not found"
            )
            
        # Count questions by validation status
        question_counts = db.query(QuestionModel).filter(
            QuestionModel.session_id == session_id
        ).all()
        
        validation_counts = {
            "validated": len([q for q in question_counts if q.validation_status == ValidationStatus.APPROVED]),
            "pending": len([q for q in question_counts if q.validation_status == ValidationStatus.PENDING]),
            "rejected": len([q for q in question_counts if q.validation_status == ValidationStatus.REJECTED])
        }
        
        response = {
            "session_id": session_id,
            "status": session.status,
            "created_at": session.created_at,
            "total_questions_requested": session.total_questions_requested,
            "questions_generated": session.questions_generated,
            "validation_counts": validation_counts,
            "config": session.config,
            "document_ids": session.document_ids,
            "error_message": session.error_message
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session status"
        )


# Title-based generation endpoints

@router.get("/documents/{document_id}/title-structure")
async def get_document_title_structure(
    document_id: str,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get the title hierarchy structure for a document.
    
    Args:
        document_id: Document identifier
        db: Database session
        
    Returns:
        Document title structure with statistics
    """
    logger.debug(f"Getting title structure for document: {document_id}")
    
    try:
        # Verify document exists
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Get title structure
        title_generator = get_title_based_generator()
        structure = title_generator.get_document_title_structure(document_id)
        
        if "error" in structure:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to analyze document structure: {structure['error']}"
            )
        
        return structure
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get title structure for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze document title structure"
        )


@router.get("/documents/{document_id}/title-suggestions")
async def get_title_suggestions(
    document_id: str,
    min_chunks: int = 3,
    db: Session = Depends(get_db_session)
) -> List[Dict[str, Any]]:
    """
    Get suggested title selections for question generation.
    
    Args:
        document_id: Document identifier
        min_chunks: Minimum chunks required for a suggestion
        db: Database session
        
    Returns:
        List of suggested title selections
    """
    logger.debug(f"Getting title suggestions for document: {document_id}")
    
    try:
        # Verify document exists
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Get suggestions
        title_generator = get_title_based_generator()
        suggestions = title_generator.get_title_suggestions(document_id, min_chunks)
        
        # Convert TitleSelectionCriteria objects to dict for JSON serialization
        for suggestion in suggestions:
            if 'criteria' in suggestion:
                criteria = suggestion['criteria']
                suggestion['criteria'] = {
                    'document_id': criteria.document_id,
                    'h1_title': criteria.h1_title,
                    'h2_title': criteria.h2_title,
                    'h3_title': criteria.h3_title,
                    'h4_title': criteria.h4_title
                }
        
        return suggestions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get title suggestions for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get title suggestions"
        )


@router.post("/documents/{document_id}/generate-from-title")
async def generate_questions_from_title(
    document_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Generate QCM questions from a specific title selection.
    
    Args:
        document_id: Document identifier
        request: Generation request with title selection and config
        db: Database session
        
    Returns:
        Generated questions and metadata
        
    Request body should contain:
    {
        "title_selection": {
            "h1_title": "optional H1 title",
            "h2_title": "optional H2 title", 
            "h3_title": "optional H3 title",
            "h4_title": "optional H4 title"
        },
        "config": {
            "num_questions": 5,
            "language": "fr",
            "difficulty_distribution": {...},
            "question_types": {...}
        },
        "session_id": "optional session id"
    }
    """
    logger.debug(f"Generating questions from title for document: {document_id}")
    
    try:
        # Verify document exists
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Extract request data
        title_selection = request.get('title_selection', {})
        config_data = request.get('config', {})
        session_id = request.get('session_id')
        
        # Create title selection criteria
        criteria = TitleSelectionCriteria(
            document_id=document_id,
            h1_title=title_selection.get('h1_title'),
            h2_title=title_selection.get('h2_title'),
            h3_title=title_selection.get('h3_title'),
            h4_title=title_selection.get('h4_title')
        )
        
        # Create generation config
        config = GenerationConfig(**config_data)
        
        # Generate questions
        title_generator = get_title_based_generator()
        questions = await title_generator.generate_questions_from_title(
            criteria, config, session_id
        )
        
        # Store questions in database if session provided
        stored_questions = []
        if session_id:
            for question in questions:
                db_question = QuestionModel(
                    session_id=session_id,
                    question_text=question.question_text,
                    question_type=question.question_type.value,
                    language=question.language.value,
                    difficulty=question.difficulty.value,
                    theme=criteria.get_title_path(),
                    options=[{"text": opt.text, "is_correct": opt.is_correct} for opt in question.options],
                    correct_answers=[i for i, opt in enumerate(question.options) if opt.is_correct],
                    explanation=question.explanation,
                    validation_status=ValidationStatus.PENDING,
                    metadata={
                        "generation_source": "title_based",
                        "title_path": criteria.get_title_path(),
                        "h1_title": title_selection.get('h1_title', ''),
                        "h2_title": title_selection.get('h2_title', ''),
                        "h3_title": title_selection.get('h3_title', ''),
                        "h4_title": title_selection.get('h4_title', '')
                    }
                )
                db.add(db_question)
                stored_questions.append(db_question)
            
            db.commit()
            for q in stored_questions:
                db.refresh(q)
        
        # Convert to response format
        response_questions = []
        for question in questions:
            response_questions.append({
                "question_text": question.question_text,
                "question_type": question.question_type.value,
                "language": question.language.value,
                "difficulty": question.difficulty.value,
                "theme": criteria.get_title_path(),
                "options": [{"text": opt.text, "is_correct": opt.is_correct} for opt in question.options],
                "explanation": question.explanation,
                "metadata": getattr(question, 'generation_params', {})
            })
        
        return {
            "questions": response_questions,
            "generation_info": {
                "document_id": document_id,
                "title_path": criteria.get_title_path(),
                "questions_generated": len(questions),
                "session_id": session_id,
                "chunks_used": len(title_generator.get_chunks_for_title_selection(criteria))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate questions from title for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate questions from title: {str(e)}"
        )


@router.get("/documents/{document_id}/title-chunks")
async def get_chunks_for_title(
    document_id: str,
    h1_title: Optional[str] = None,
    h2_title: Optional[str] = None,
    h3_title: Optional[str] = None,
    h4_title: Optional[str] = None,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get chunks that match the specified title criteria.
    
    Args:
        document_id: Document identifier
        h1_title: Optional H1 title filter
        h2_title: Optional H2 title filter
        h3_title: Optional H3 title filter
        h4_title: Optional H4 title filter
        db: Database session
        
    Returns:
        Matching chunks with metadata
    """
    logger.debug(f"Getting chunks for title selection in document: {document_id}")
    
    try:
        # Verify document exists
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Create title selection criteria
        criteria = TitleSelectionCriteria(
            document_id=document_id,
            h1_title=h1_title,
            h2_title=h2_title,
            h3_title=h3_title,
            h4_title=h4_title
        )
        
        # Get matching chunks
        title_generator = get_title_based_generator()
        chunks = title_generator.get_chunks_for_title_selection(criteria)
        
        return {
            "title_path": criteria.get_title_path(),
            "matching_chunks": len(chunks),
            "chunks": chunks,
            "selection_criteria": {
                "h1_title": h1_title,
                "h2_title": h2_title,
                "h3_title": h3_title,
                "h4_title": h4_title
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunks for title in document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chunks for title selection"
        )