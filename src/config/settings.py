"""
Configuration settings for the application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "mistral:7b-instruct")
MODEL_TIMEOUT = float(os.getenv("MODEL_TIMEOUT", "60.0"))

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

# Vector DB settings
PERSIST_DIR = os.getenv("PERSIST_DIR", os.path.join(STORAGE_DIR, "vectordb"))

# Document processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_DOCUMENT_SIZE = int(os.getenv("MAX_DOCUMENT_SIZE", "20000"))
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "storage/embedding_cache/bge-small-en-v1.5-sbert")

# Server settings
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "4"))

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True) 