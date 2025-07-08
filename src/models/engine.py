"""
Core engine for LLM and vector store functionality.
"""
import os
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

from src.config import settings


# Template for multiple choice question generation
QUESTION_GEN_TEMPLATE = PromptTemplate(
    """
    I want you to create {num_questions} multiple-choice questions based on the document text provided below. 
    Please create exactly {num_questions} questions.

    Document text:
    \"\"\"
    {context}
    \"\"\"

    For each question:
    1. Create a meaningful question related to the text content
    2. Write 4 options (A, B, C, D) - only one should be the correct answer
    3. Indicate which option is the correct answer
    4. Write a brief explanation for the correct answer

    Answer in the following format:

    1. Question: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Correct option letter]
    Explanation: [Explanation for the correct answer]

    2. Question: [Question text]
    A) [Option A]
    ...

    (Please create exactly {num_questions} questions in this format. Number each question clearly.)
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
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL
        )
        
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
        
        self.index = None
    
    def create_index(self, nodes):
        """
        Create a vector index from document nodes.
        
        Args:
            nodes: List of document nodes to index
            
        Returns:
            The created index
        """
        if not nodes:
            print("No nodes to index.")
            return None
        
        print("Creating vector index...")
        self.index = VectorStoreIndex(
            nodes, 
            storage_context=self.storage_context
        )
        
        # Persist the index to disk
        self.persist_index()
        
        print("Vector index created and persisted.")
        return self.index
    
    def persist_index(self):
        """
        Persist the index to disk for later use.
        """
        if self.index and self.storage_context:
            index_persist_dir = os.path.join(settings.PERSIST_DIR, "index")
            os.makedirs(index_persist_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=index_persist_dir)
            print(f"Index persisted to {index_persist_dir}")
            return True
        return False
    
    def load_index(self):
        """
        Load a previously persisted index from disk.
        
        Returns:
            Loaded index or None if not found
        """
        index_persist_dir = os.path.join(settings.PERSIST_DIR, "index")
        if os.path.exists(index_persist_dir):
            try:
                print(f"Loading index from {index_persist_dir}...")
                storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store,
                    persist_dir=index_persist_dir
                )
                self.index = VectorStoreIndex.from_storage(storage_context)
                print("Index loaded successfully.")
                return self.index
            except Exception as e:
                print(f"Error loading index: {e}")
                return None
        else:
            print("No persisted index found.")
            return None
    
    def clear_index(self):
        """
        Clear the persisted index.
        
        Returns:
            True if successful, False otherwise
        """
        index_persist_dir = os.path.join(settings.PERSIST_DIR, "index")
        try:
            if os.path.exists(index_persist_dir):
                import shutil
                shutil.rmtree(index_persist_dir)
                print(f"Index at {index_persist_dir} has been cleared.")
                
                # Clear ChromaDB collection
                if self.chroma_collection:
                    self.chroma_collection.delete(where={})
                    print("ChromaDB collection cleared.")
                
                # Reset the index reference
                self.index = None
                return True
            return False
        except Exception as e:
            print(f"Error clearing index: {e}")
            return False
    
    def get_query_engine(self, similarity_top_k=3, response_mode="compact"):
        """
        Create a query engine for asking questions.
        
        Args:
            similarity_top_k: Number of top similar chunks to retrieve
            response_mode: How to synthesize responses ("compact", "refine", etc.)
            
        Returns:
            Query engine object
        """
        if not self.index:
            print("Index not created yet. Please create an index first.")
            return None
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode
        )
        
        # Create query engine
        query_engine = self.index.as_query_engine(
            response_synthesizer=response_synthesizer,
            similarity_top_k=similarity_top_k
        )
        
        return query_engine
    
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
        
        response = self.llm.complete(
            QUESTION_GEN_TEMPLATE.format(
                context=context,
                num_questions=num_questions
            )
        )
        
        return response.text     
    
    def generate_questions(self, index=None, num_questions=5, similarity_top_k=10):
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
        
        # Estimate max tokens we can use for context (leaving room for prompt and response)
        # Mistral context depends on model version, typically 8k-32k tokens
        # A safe approach is to use ~4000 tokens for context
        estimated_max_chars = 16000  # ~4000 tokens, rough estimate
        prompt_overhead = 500  # Approximate chars for prompt template
        
        # Process nodes to fit within context window
        selected_nodes = []
        current_length = 0
        
        # First, sort nodes by relevance score (if available)
        if hasattr(nodes[0], 'score'):
            nodes = sorted(nodes, key=lambda n: n.score if hasattr(n, 'score') else 0, reverse=True)
        
        # Then select nodes up to max length
        for node in nodes:
            content = node.node.get_content()
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
        
        print(f"Generating {num_questions} questions from selected chunks...")
        
        # Call the LLM to generate questions
        return self._generate_with_llm(context, num_questions) 