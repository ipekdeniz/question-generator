"""
Pydantic schemas for API data validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Option(BaseModel):
    """Schema for a question option."""
    text: str = Field(..., description="Text of the option")
    label: str = Field(..., description="Label (A, B, C, D) of the option")


class QuestionBase(BaseModel):
    """Base schema for a multiple choice question."""
    question: str = Field(..., description="Question text")
    options: List[Option] = Field(..., description="Available options")
    correct_answer: str = Field(..., description="Correct option letter (A, B, C, D)")
    explanation: Optional[str] = Field(None, description="Explanation for the correct answer")


class QuestionCreate(BaseModel):
    """Schema for creating a question."""
    question: str = Field(..., description="Question text")
    options: List[str] = Field(..., description="List of option texts")
    correct_answer: str = Field(..., description="Correct option letter (A, B, C, D)")
    explanation: Optional[str] = Field(None, description="Explanation for the correct answer")


class Question(QuestionBase):
    """Full question schema."""
    id: str = Field(..., description="Question ID")


class QuestionResponse(BaseModel):
    """Response schema for returning questions."""
    questions: List[Question] = Field(..., description="List of questions")
    count: int = Field(..., description="Number of questions")


class GenerateQuestionsRequest(BaseModel):
    """Schema for question generation request."""
    num_questions: int = Field(5, description="Number of questions to generate", ge=1, le=20)
    use_all_documents: bool = Field(True, description="Whether to use all documents in the collection")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to use")


class QueryRequest(BaseModel):
    """Schema for querying the documents."""
    query: str = Field(..., description="Query text")
    top_k: int = Field(3, description="Number of results to return", ge=1, le=10)


class QueryResponse(BaseModel):
    """Schema for query response."""
    answer: str = Field(..., description="Answer to the query")
    sources: List[str] = Field(..., description="Source documents for the answer")


class DocumentMetadata(BaseModel):
    """Schema for document metadata."""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="Document size in bytes")
    created_at: str = Field(..., description="Creation timestamp")


class DocumentsResponse(BaseModel):
    """Schema for listing documents."""
    documents: List[DocumentMetadata] = Field(..., description="List of documents")
    count: int = Field(..., description="Number of documents") 