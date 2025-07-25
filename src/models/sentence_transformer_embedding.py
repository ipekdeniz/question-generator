from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_path):
        super().__init__()
        object.__setattr__(self, 'model', SentenceTransformer(model_path))
    
    def _get_query_embedding(self, query: str):
        return self.model.encode(query).tolist()
    
    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str):
        return self._get_query_embedding(text)
    
    def _get_text_embeddings(self, texts: list):
        return self.model.encode(texts).tolist() 