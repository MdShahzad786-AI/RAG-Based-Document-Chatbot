import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config
from utils.logger import get_logger

logger = get_logger(__name__)

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the HuggingFace embedding model.
    """
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

def get_vector_store() -> Chroma:
    """
    Returns the Chroma vector store, initializing it if necessary.
    """
    embeddings = get_embeddings()
    logger.info(f"Connecting to Chroma vector store at {config.VECTORSTORE_DIR}")
    vector_store = Chroma(
        collection_name="document_collection",
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR)
    )
    return vector_store

def add_documents_to_store(chunks: List[Document]) -> Chroma:
    """
    Adds document chunks to the vector store.
    """
    if not chunks:
        logger.warning("No chunks to add to the vector store.")
        return get_vector_store()
        
    vector_store = get_vector_store()
    logger.info(f"Adding {len(chunks)} chunks to the vector store.")
    vector_store.add_documents(chunks)
    logger.info("Successfully added documents to the vector store.")
    return vector_store

def clear_vector_store():
    """
    Clears the existing vector store by re-creating an empty one or removing the directory.
    """
    try:
        vector_store = get_vector_store()
        vector_store.delete_collection()
        logger.info("Cleared the vector store collection.")
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
