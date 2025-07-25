from src.models.engine import LLMEngine
from src.utils.document_loader import load_documents, split_documents_into_nodes

class IndexService:
    def __init__(self, engine: LLMEngine = None):
        self.engine = engine or LLMEngine()

    def rebuild_index(self):
        self.engine.clear_index()
        documents = load_documents()
        if not documents:
            return {"status": "error", "message": "No documents found for indexing"}, 404
        nodes = split_documents_into_nodes(documents)
        if self.engine.create_index(nodes):
            return {
                "status": "success",
                "message": f"Index rebuilt with {len(documents)} documents",
                "document_count": len(documents),
                "node_count": len(nodes)
            }, 200
        else:
            return {"status": "error", "message": "Failed to rebuild index"}, 500

    def clear_index(self):
        if self.engine.clear_index():
            return {"status": "success", "message": "Index cleared successfully"}, 200
        else:
            return {"status": "error", "message": "No index found or error clearing index"}, 404 