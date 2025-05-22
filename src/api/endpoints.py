"""
FastAPI endpoints for the question generation service.
"""
import os
import uuid
import time
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse

from src.api.schemas import (
    Question, QuestionResponse, QueryRequest, QueryResponse,
    GenerateQuestionsRequest, DocumentMetadata, DocumentsResponse
)
from src.models.engine import LLMEngine
from src.models.parser import parse_questions_from_text
from src.utils.document_loader import load_documents, split_documents_into_nodes, get_document_text
from src.config import settings


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
    
    Args:
        request: Question generation parameters
        engine: LLM engine dependency
    
    Returns:
        Generated questions
    """
    # Load documents
    documents = load_documents()
    
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")
    
    # Try to load existing index first
    if not engine.index:
        engine.load_index()
        
    # If no index exists, create a new one
    if not engine.index:
        # Process documents for RAG
        nodes = split_documents_into_nodes(documents)
        engine.create_index(nodes)
    
    # Generate questions using RAG approach
    raw_questions = engine.generate_questions(
        num_questions=request.num_questions,
        similarity_top_k=10  # Retrieve 10 most relevant chunks
    )
    
    # Parse questions
    parsed_questions = parse_questions_from_text(raw_questions, request.num_questions)
    
    # Add IDs to questions
    questions = []
    for q in parsed_questions:
        # Convert options format to match schema
        options = []
        for opt in q['options']:
            label = opt[0]
            text = opt[3:].strip()
            options.append({"label": label, "text": text})
        
        questions.append(Question(
            id=str(uuid.uuid4()),
            question=q['question'],
            options=options,
            correct_answer=q['correct_answer'],
            explanation=q['explanation']
        ))
    
    return QuestionResponse(questions=questions, count=len(questions))


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    engine: LLMEngine = Depends(get_engine)
):
    """
    Query the document collection.
    
    Args:
        request: Query parameters
        engine: LLM engine dependency
    
    Returns:
        Answer with sources
    """
    # Load documents
    documents = load_documents()
    
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")
    
    # Try to load existing index first
    if not engine.index:
        engine.load_index()
        
    # If no index exists, create a new one
    if not engine.index:
        # Process documents
        nodes = split_documents_into_nodes(documents)
        engine.create_index(nodes)
    
    # Create query engine
    query_engine = engine.get_query_engine(similarity_top_k=request.top_k)
    
    if not query_engine:
        raise HTTPException(status_code=500, detail="Failed to create query engine")
    
    # Execute query
    response = query_engine.query(request.query)
    
    # Extract sources
    sources = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            source = node.node.metadata.get('file_name', 'Unknown Source')
            if source not in sources:
                sources.append(source)
    
    return QueryResponse(
        answer=response.response,
        sources=sources
    )


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    """
    List all documents in the collection.
    
    Returns:
        List of document metadata
    """
    # Get document list from documents directory
    document_list = []
    
    if os.path.exists(settings.DOCUMENTS_DIR):
        for filename in os.listdir(settings.DOCUMENTS_DIR):
            # Skip hidden files (files starting with a dot)
            if filename.startswith('.'):
                continue
                
            file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
            if os.path.isfile(file_path):
                file_stat = os.stat(file_path)
                document_list.append(DocumentMetadata(
                    id=str(uuid.uuid4()),
                    filename=filename,
                    size=file_stat.st_size,
                    created_at=time.ctime(file_stat.st_ctime)
                ))
    
    return DocumentsResponse(
        documents=document_list,
        count=len(document_list)
    )


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(None)
):
    """
    Upload a document to the collection.
    
    Args:
        file: File to upload
        description: Optional description
    
    Returns:
        Upload status
    """
    # Create documents directory if it doesn't exist
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    
    return JSONResponse(
        status_code=201,
        content={
            "filename": file.filename,
            "size": len(content),
            "description": description,
            "status": "uploaded"
        }
    )


@router.post("/index/rebuild")
async def rebuild_index(
    engine: LLMEngine = Depends(get_engine)
):
    """
    Clear and rebuild the document index.
    
    Args:
        engine: LLM engine dependency
    
    Returns:
        Status of the rebuild operation
    """
    # First, clear existing index
    engine.clear_index()
    
    # Load documents
    documents = load_documents()
    
    if not documents:
        return JSONResponse(
            status_code=404, 
            content={"status": "error", "message": "No documents found for indexing"}
        )
    
    # Process documents
    nodes = split_documents_into_nodes(documents)
    
    # Create new index
    if engine.create_index(nodes):
        return JSONResponse(
            status_code=200,
            content={
                "status": "success", 
                "message": f"Index rebuilt with {len(documents)} documents",
                "document_count": len(documents),
                "node_count": len(nodes)
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Failed to rebuild index"}
        )


@router.delete("/index")
async def clear_index(
    engine: LLMEngine = Depends(get_engine)
):
    """
    Clear the document index.
    
    Args:
        engine: LLM engine dependency
    
    Returns:
        Status of the clear operation
    """
    if engine.clear_index():
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "Index cleared successfully"}
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No index found or error clearing index"}
        ) 