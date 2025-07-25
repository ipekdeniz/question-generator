"""
Core engine for LLM and vector store functionality.
"""
from llama_index.core import StorageContext, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from src.config import settings
from src.models.index_manager import IndexManager
from src.models.sentence_transformer_embedding import SentenceTransformerEmbedding


# Template for multiple choice question generation
QUESTION_GEN_TEMPLATE = PromptTemplate(
    """
    Create {num_questions} high-quality multiple-choice question from this text:

    {context}

    Requirements:
    - Create meaningful questions that test understanding
    - Make all 4 options (A, B, C, D) similar in length and plausibility
    - Only one option should be correct
    - Provide a brief explanation for the correct answer

    Format exactly as follows:
    1. Question: [Clear, specific question]
    A) [Option A - similar length to others]
    B) [Option B - similar length to others]
    C) [Option C - similar length to others]
    D) [Option D - similar length to others]
    Correct Answer: [Letter only: A, B, C, or D]
    Explanation: [Brief explanation why this is correct]

    Create exactly {num_questions} question(s) in this format.
    """
)


class LLMEngine:
    """Main engine for LLM interactions and vector indexing."""
    
    def __init__(self):
        """Initialize the LLM engine with models and vector store."""
        # Set up LLM
        self.llm = Ollama(
            model=settings.MODEL_NAME,
            request_timeout=settings.MODEL_TIMEOUT,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # Set up embedding model
        try:
            self.embed_model = SentenceTransformerEmbedding(settings.EMBEDDING_MODEL_PATH)
            print(f"✅ SentenceTransformer embedding model loaded from {settings.EMBEDDING_MODEL_PATH}")
            
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=settings.PERSIST_DIR)
        self.chroma_collection = self.chroma_client.get_or_create_collection("document_collection")
        
        # Set up vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Configure settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = settings.CHUNK_SIZE
        Settings.chunk_overlap = settings.CHUNK_OVERLAP
        
        self.index_manager = IndexManager(self)
        self.index = None
    
    def create_index(self, nodes):
        """
        Create a vector index from document nodes.
        
        Args:
            nodes: List of document nodes to index
            
        Returns:
            The created index
        """
        self.index = self.index_manager.create_index(nodes)
        return self.index
    
    def persist_index(self):
        """
        Persist the index to disk for later use.
        """
        return self.index_manager.persist_index()
    
    def load_index(self):
        """
        Load a previously persisted index from disk.
        
        Returns:
            Loaded index or None if not found
        """
        self.index = self.index_manager.load_index()
        return self.index
    
    def clear_index(self):
        """
        Clear the persisted index.
        
        Returns:
            True if successful, False otherwise
        """
        result = self.index_manager.clear_index()
        if result:
            self.index = None
        return result
    
    def _generate_with_llm(self, context, num_questions=5):
        """
        Internal helper to generate questions using LLM.
        
        Args:
            context: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            Raw LLM response text with questions
        """
        if not context:
            print("No content provided for question generation.")
            return None
        
        print(f"Generating {num_questions} questions...")
        
        # Try direct Ollama API first
        try:
            import requests
            import json
            
            prompt = QUESTION_GEN_TEMPLATE.format(
                context=context,
                num_questions=num_questions
            )
            
            print(f"Using direct Ollama API...")
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Direct Ollama API success: {len(result.get('response', ''))} chars")
                return result.get('response', '')
            else:
                print(f"Direct Ollama API failed: {response.status_code}")
                
        except Exception as e:
            print(f"Direct Ollama API error: {e}")
        
        # Fallback to LlamaIndex wrapper
        print(f"Falling back to LlamaIndex wrapper...")
        response = self.llm.complete(
            QUESTION_GEN_TEMPLATE.format(
                context=context,
                num_questions=num_questions
            )
        )
        
        return response.text
    
    def generate_questions(self, index=None, num_questions=5, similarity_top_k=1):
        """
        Generate multiple-choice questions using RAG approach.
        
        Args:
            index: Vector index to use (uses self.index if None)
            num_questions: Number of questions to generate
            similarity_top_k: Number of top relevant chunks to use
            
        Returns:
            Raw LLM response text with questions
        """
        if not index and not self.index:
            print("No index available. Please create an index first.")
            return None
        
        active_index = index if index else self.index
        
        # Create a query for getting diverse chunks from the documents
        # Use a generic query to get a good coverage of the document content
        retriever = active_index.as_retriever(similarity_top_k=similarity_top_k)
        nodes = retriever.retrieve("Summarize the main topics and key information in these documents")
        
        print(f"Retrieved {len(nodes)} nodes from index")
        
        # Debug: Check node content
        for i, node in enumerate(nodes):
            # Always get text, fallback to get_content
            content = getattr(node.node, 'text', None) or node.node.get_content()
            print(f"Node {i}: {len(content)} chars - {content[:100]}...")
        
        # Use minimal context for faster processing
        estimated_max_chars = 800  # Increased for better quality
        prompt_overhead = 200  # Approximate chars for prompt template
        
        # Process nodes to fit within context window
        selected_nodes = []
        current_length = 0
        
        # First, sort nodes by relevance score (if available)
        if hasattr(nodes[0], 'score'):
            nodes = sorted(nodes, key=lambda n: n.score if hasattr(n, 'score') else 0, reverse=True)
        
        # Then select nodes up to max length
        for node in nodes:
            content = getattr(node.node, 'text', None) or node.node.get_content()
            # If this single node exceeds our limit, we might need to truncate it
            if len(content) > estimated_max_chars - prompt_overhead and len(selected_nodes) == 0:
                print(f"Single node too large, truncating to fit context window")
                selected_nodes.append(content[:estimated_max_chars - prompt_overhead])
                break
            
            if current_length + len(content) <= estimated_max_chars - prompt_overhead:
                selected_nodes.append(content)
                current_length += len(content)
            else:
                # We've reached our limit
                break
        
        print(f"Using {len(selected_nodes)} of {len(nodes)} relevant chunks (total {current_length} chars)")
        
        # Combine the selected chunks into context for question generation
        context = "\n\n".join(selected_nodes)
        
        print(f"Final context length: {len(context)} chars")
        print(f"Context preview: {context[:200]}...")
        
        if not context.strip():
            print("ERROR: Empty context! Cannot generate questions.")
            return "Error: No content available for question generation."
        
        print(f"Generating {num_questions} questions from selected chunks...")
        
        # Call the LLM to generate questions
        return self._generate_with_llm(context, num_questions) 