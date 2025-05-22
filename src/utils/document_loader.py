"""
Document loading and processing utilities.
"""
import os
from typing import List, Optional

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser

from src.config import settings


def load_documents(docs_dir: str = settings.DOCUMENTS_DIR) -> List[Document]:
    """
    Load documents from the specified directory.
    
    Args:
        docs_dir: Directory containing documents to load
        
    Returns:
        List of loaded documents
    """
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Created '{docs_dir}' directory. Please add your documents there.")
        return []
    
    documents = SimpleDirectoryReader(docs_dir).load_data()
    if not documents:
        print(f"No documents found in '{docs_dir}' directory.")
        return []
    
    print(f"Successfully loaded {len(documents)} documents.")
    return documents


def get_document_text(documents: List[Document]) -> str:
    """
    Combine document texts, limited to the maximum allowed size.
    
    Args:
        documents: List of documents to combine
        
    Returns:
        Combined document text
    """
    all_text = "\n\n".join([doc.text for doc in documents])
    
    if len(all_text) > settings.MAX_DOCUMENT_SIZE:
        print(f"Text too long, using only first {settings.MAX_DOCUMENT_SIZE//1000}K characters.")
        all_text = all_text[:settings.MAX_DOCUMENT_SIZE]
    
    return all_text


def split_documents_into_nodes(documents: List[Document]):
    """
    Split documents into smaller chunks (nodes) for better processing.
    
    Args:
        documents: List of documents to split
        
    Returns:
        List of nodes
    """
    if not documents:
        return []
    
    print("Processing documents into chunks...")
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=settings.CHUNK_SIZE, 
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"Documents split into {len(nodes)} chunks.")
    
    return nodes 