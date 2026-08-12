import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# API Keys and Endpoints
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# UI Settings
APP_TITLE = "Enterprise RAG Chatbot"
APP_ICON = "🤖"
