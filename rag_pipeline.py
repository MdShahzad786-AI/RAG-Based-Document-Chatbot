from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Free LLM Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

import config
from vector_store import get_vector_store
from utils.logger import get_logger

logger = get_logger(__name__)

# Prompt template designed to return answers with source citations
QA_PROMPT_TEMPLATE = """You are a helpful and professional enterprise AI assistant.
Use the following pieces of retrieved context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always include the source document and page number for your answer in a "Sources:" section at the end of your response.

Context:
{context}

Question:
{question}

Answer:
"""

def get_llm(provider: str = "Gemini", model_name: str = ""):
    """
    Returns the LLM instance based on the chosen provider.
    """
    logger.info(f"Initializing LLM provider: {provider}")
    
    if provider == "Gemini":
        # Uses GEMINI_API_KEY from environment
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    elif provider == "Ollama":
        # Requires Ollama running locally (e.g., llama3 or mistral)
        model = model_name if model_name else "llama3"
        return ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL, temperature=0.3)
    
    elif provider == "HuggingFace":
        # Requires HUGGINGFACEHUB_API_TOKEN
        repo_id = model_name if model_name else "HuggingFaceH4/zephyr-7b-beta"
        hf_llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            temperature=0.3,
            max_new_tokens=512
        )
        return ChatHuggingFace(llm=hf_llm)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def format_docs_with_sources(docs: List[Any]) -> str:
    """
    Formats the retrieved documents into a string containing the text and metadata.
    """
    formatted_context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        # In PyPDFLoader, 'page' is 0-indexed, let's make it 1-indexed for humans
        if isinstance(page, int):
            page += 1
            
        formatted_context += f"--- Chunk {i+1} ---\n"
        formatted_context += f"Source: {source} | Page: {page}\n"
        formatted_context += f"Content: {doc.page_content}\n\n"
        
    return formatted_context

def create_rag_chain(provider: str = "Gemini", model_name: str = ""):
    """
    Creates and returns the RAG pipeline chain.
    """
    # 1. Setup Retriever
    vector_store = get_vector_store()
    # Search for top 5 chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 2. Setup LLM
    llm = get_llm(provider, model_name)
    
    # 3. Setup Prompt
    prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
    
    # 4. Build Chain
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain successfully created.")
    return rag_chain

def answer_question(question: str, provider: str = "Gemini", model_name: str = "") -> str:
    """
    Invokes the RAG chain with the user's question.
    """
    try:
        rag_chain = create_rag_chain(provider, model_name)
        logger.info(f"Answering question: {question}")
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"An error occurred while generating the answer: {e}"
