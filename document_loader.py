import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from utils.logger import get_logger

logger = get_logger(__name__)

def process_documents(uploaded_files) -> List[Document]:
    """
    Saves uploaded files to disk, loads them, and splits them into chunks.
    """
    all_chunks = []
    
    # Save uploaded files to the DATA_DIR
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = config.DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
        logger.info(f"Saved uploaded file: {file_path}")

    # Load and split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True
    )

    for file_path in saved_paths:
        try:
            logger.info(f"Loading document: {file_path}")
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            
            logger.info(f"Splitting document: {file_path}")
            chunks = text_splitter.split_documents(docs)
            
            # Ensure source metadata is clean (just filename, not full path)
            for chunk in chunks:
                chunk.metadata["source"] = Path(chunk.metadata.get("source", "")).name
                
            all_chunks.extend(chunks)
            logger.info(f"Processed {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")

    return all_chunks
