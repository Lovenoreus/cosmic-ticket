"""
Configuration for cosmic database and Qdrant vector database tools
"""
import os
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COSMIC_DATABASE_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cosmic_documents-text-embedding-3-large")

# Embedding model configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-3-large")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_MODEL_NAME = os.getenv("AGENT_MODEL_NAME", "gpt-4o-mini")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Qdrant search configuration
QDRANT_RESULT_LIMIT = int(os.getenv("QDRANT_RESULT_LIMIT", "5"))

