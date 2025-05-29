"""Configuration settings for JSON RAG System."""

import os
from typing import Dict, Any

class Config:
    """Central configuration class for the JSON RAG system."""
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "local")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "documents")
    
    # FAISS Configuration
    FAISS_INDEX_PATH = "./data/faiss_index.bin"
    EMBEDDINGS_CACHE_PATH = "./data/embeddings_cache.pkl"
    VECTOR_DIMENSION = 384  # For sentence-transformers/all-MiniLM-L6-v2
    
    # Sentence Transformer Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_SEQUENCE_LENGTH = 512
    
    # Search Configuration  
    TOP_K_RESULTS = 5  # Increased for more comprehensive results
    FUZZY_THRESHOLD = 55  # Lowered threshold for more matches
    SEMANTIC_WEIGHT = 0.7  # Weight for semantic search (0.3 for fuzzy)
    MIN_COMBINED_SCORE = 0.4  # Lowered for more inclusive results
    
    # Text Processing Configuration
    IGNORE_VALUES = {"unknown", "null", "none", "", "0", "zero", "n/a", "na"}
    MIN_TEXT_LENGTH = 3
    MAX_TEXT_LENGTH = 2000
    
    # Session Configuration
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    MAX_SESSION_HISTORY = 10
    
    # Response Generation
    MAX_RESPONSE_LENGTH = 100000  # Increased limit for comprehensive responses
    ENABLE_RESPONSE_TRUNCATION = False  # Disabled by default for full JSON responses
    INCLUDE_SOURCE_REFERENCES = True
    SHOW_FULL_JSON = True  # Show complete source JSON in responses
    
    # Gradio Configuration
    GRADIO_PORT = 7860
    GRADIO_HOST = "0.0.0.0"
    GRADIO_THEME = "default"
    
    # Data Paths
    DATA_DIR = "./data"
    PROCESSED_TEXT_FILE = "./data/processed_texts.jsonl"
    LOGS_DIR = "./logs"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        import os
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
    
    @classmethod
    def get_mongodb_config(cls) -> Dict[str, Any]:
        """Get MongoDB configuration as dictionary."""
        return {
            "uri": cls.MONGODB_URI,
            "database": cls.MONGODB_DATABASE,
            "collection": cls.MONGODB_COLLECTION
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration as dictionary."""
        return {
            "top_k": cls.TOP_K_RESULTS,
            "fuzzy_threshold": cls.FUZZY_THRESHOLD,
            "semantic_weight": cls.SEMANTIC_WEIGHT,
            "embedding_model": cls.EMBEDDING_MODEL
        }