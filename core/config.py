# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the Orthomolecular Medicine RAG system."""

    # Database Configuration
    MONGODB_URI     = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME   = "books"
    COLLECTION_NAME = "chunks"
    BATCH_SIZE      = 4    # Reduce to prevent CUDA memory issues

    # Model Configuration
    MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DIMENSION = 384

    # Search Configuration
    TOP_K = 3

    # Generation Configuration
    MAX_LENGTH = 768      # More balanced length
    MIN_LENGTH = 100

    # Chunk Configuration
    CHUNK_SIZE    = 1024  # More balanced chunk size
    CHUNK_OVERLAP = 128   # Reduced proportionally

