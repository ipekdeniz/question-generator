"""
FastAPI endpoints for the question generation service.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse

from src.api.schemas import (
    QuestionResponse, GenerateQuestionsRequest, DocumentsResponse
)
from src.models.engine import LLMEngine

from src.api.services.question_service import QuestionGenerationService
from src.api.services.document_service import DocumentService
from src.api.services.index_service import IndexService


router = APIRouter()
engine = LLMEngine()


def get_engine():
    """Dependency to get the LLM engine."""
    return engine


@router.post("/questions/generate", response_model=QuestionResponse)
async def generate_questions(
    request: GenerateQuestionsRequest,
    engine: LLMEngine = Depends(get_engine)
):
    """
    Generate multiple-choice questions from documents.
    """
    service = QuestionGenerationService(engine)
    result = service.generate_questions(num_questions=request.num_questions)
    if result is None:
        raise HTTPException(status_code=404, detail="No documents found")
    return result


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    service = DocumentService()
    return service.list_documents()


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(None)
):
    service = DocumentService()
    try:
        result = await service.upload_document(file, description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(status_code=201, content=result)


@router.post("/index/rebuild")
async def rebuild_index(
    engine: LLMEngine = Depends(get_engine)
):
    service = IndexService(engine)
    result, status = service.rebuild_index()
    return JSONResponse(status_code=status, content=result)


@router.delete("/index")
async def clear_index(
    engine: LLMEngine = Depends(get_engine)
):
    service = IndexService(engine)
    result, status = service.clear_index()
    return JSONResponse(status_code=status, content=result) 