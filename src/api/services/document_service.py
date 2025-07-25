import os
import time
import uuid
from src.api.schemas import DocumentMetadata, DocumentsResponse
from src.config import settings

class DocumentService:
    def list_documents(self) -> DocumentsResponse:
        document_list = []
        if os.path.exists(settings.DOCUMENTS_DIR):
            for filename in os.listdir(settings.DOCUMENTS_DIR):
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
        return DocumentsResponse(documents=document_list, count=len(document_list))

    async def upload_document(self, file, description=None):
        allowed_extensions = {"pdf", "doc", "docx", "txt"}
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in allowed_extensions:
            raise Exception(f"Unsupported file type: .{ext}. Only PDF, DOC, DOCX, and TXT files are allowed.")
        os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
        file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")
        return {
            "filename": file.filename,
            "size": len(content),
            "description": description,
            "status": "uploaded"
        } 