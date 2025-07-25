# Document-Based Question Generator

A professional application for generating multiple-choice questions from documents using local Large Language Models.

## Features

- **Document Processing**: Upload and manage PDF, TXT, and other text documents
- **Question Generation**: Create high-quality multiple-choice questions from your documents
- **Vector Search**: Perform semantic search in your document collection
- **Local LLM**: Uses Ollama for local LLM processing (no data leaves your system)
- **API Interface**: Complete REST API with Swagger documentation
- **Persistence**: Vector database for fast and efficient document retrieval
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Architecture

The application follows a modular architecture:

- **API Layer**: FastAPI endpoints for interacting with the system
- **Data Models**: Pydantic schemas for validation and documentation
- **Core Engine**: LlamaIndex integration with Ollama
- **Storage**: ChromaDB vector database for document embeddings
- **Utils**: Document processing and parsing utilities

## Prerequisites

- Python 3.8+ or Docker
- Ollama with Mistral model installed (`ollama pull mistral:7b-instruct`)
- 8GB+ RAM (16GB+ recommended)

## Installation

### Using Python (Development)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd document-question-generator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file:
   ```bash
   cp env_example .env
   ```

5. Run the application:
   ```bash
   python run.py
   ```

### Using Docker (Production)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd document-question-generator
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Access the API at http://localhost:8000

## API Usage

The API is available at `http://localhost:8000` with the following endpoints:

- `GET /api/documents`: List all documents
- `POST /api/documents/upload`: Upload a new document
- `POST /api/questions/generate`: Generate questions from documents
- `POST /api/query`: Query the document collection

Complete API documentation is available at `http://localhost:8000/docs`.

## Example Usage

### Generate Questions

```bash
curl -X POST "http://localhost:8000/api/questions/generate" \
  -H "Content-Type: application/json" \
  -d '{"num_questions": 5, "use_all_documents": true}'
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 3}'
```

## Customization

- Change the LLM model in `.env` by modifying `MODEL_NAME`
- Configure Ollama connection in `.env`:
  - `OLLAMA_HOST`: Ollama server host (default: localhost)
  - `OLLAMA_PORT`: Ollama server port (default: 11434)
  - `OLLAMA_BASE_URL`: Full Ollama URL (optional override)
- Adjust chunking parameters in `.env` with `CHUNK_SIZE` and `CHUNK_OVERLAP`

## Roadmap

- Web UI for easier interaction
- Multiple question formats (not just multiple-choice)
- Question difficulty levels
- Export to various formats (PDF, DOCX, etc.)
- User authentication and management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 