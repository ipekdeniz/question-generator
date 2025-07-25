import os
from llama_index.core import VectorStoreIndex, StorageContext
from src.config import settings

class IndexManager:
    def __init__(self, engine):
        self.engine = engine
        self.chroma_client = engine.chroma_client
        self.chroma_collection = engine.chroma_collection
        self.vector_store = engine.vector_store
        self.storage_context = engine.storage_context
        self.index = None

    def create_index(self, nodes):
        if not nodes:
            print("No nodes to index.")
            return None
        print("Creating vector index...")
        self.index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context
        )
        self.engine.index = self.index
        self.persist_index()
        print("Vector index created and persisted.")
        return self.index

    def persist_index(self):
        if self.index and self.storage_context:
            index_persist_dir = os.path.join(settings.PERSIST_DIR, "index")
            os.makedirs(index_persist_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=index_persist_dir)
            print(f"Index persisted to {index_persist_dir}")
            return True
        return False

    def load_index(self):
        index_persist_dir = os.path.join(settings.PERSIST_DIR, "index")
        if os.path.exists(index_persist_dir):
            try:
                print(f"Loading index from {index_persist_dir}...")
                existing_nodes = self.chroma_collection.get()
                if not existing_nodes['ids']:
                    print("No existing nodes found in vector store.")
                    return None
                print(f"Found {len(existing_nodes['ids'])} existing nodes in vector store.")
                storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context
                )
                self.engine.index = self.index
                print("Index loaded successfully.")
                return self.index
            except Exception as e:
                print(f"Error loading index: {e}")
                return None
        else:
            print("No persisted index found.")
            return None

    def clear_index(self):
        index_persist_dir = os.path.join(settings.PERSIST_DIR, "index")
        try:
            if os.path.exists(index_persist_dir):
                import shutil
                shutil.rmtree(index_persist_dir)
                print(f"Index at {index_persist_dir} has been cleared.")
                if self.chroma_collection:
                    self.chroma_collection.delete(where={})
                    print("ChromaDB collection cleared.")
                self.index = None
                self.engine.index = None
                return True
            return False
        except Exception as e:
            print(f"Error clearing index: {e}")
            return False 