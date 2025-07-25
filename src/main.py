"""
Main application entry point.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline

from src.api.endpoints import router
from src.config import settings

# Create FastAPI application with offline docs
app = FastAPIOffline(
    title="Document-Based Question Generator",
    description="API for generating multiple-choice questions from documents using LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Document-Based Question Generator",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs"
    }