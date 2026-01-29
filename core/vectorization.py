# vectorization.py
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from loguru import logger
from core.config import Config
from core.database import Database

#-------------------------------------------------------------------------------------------#
class VectorizationPipeline:
    def __init__(self):
        """Initialize the vectorization pipeline with the specified model"""
        self.model = SentenceTransformer(Config.MODEL_NAME)
        self.db    = Database()
        logger.info(f"Initialized vectorization pipeline with model: {Config.MODEL_NAME}")
    
    #-----------------------------------------------------------------------#
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text chunks"""
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    #-----------------------------------------------------------------------#
    def process_chunks(self, chunks: List[Dict[str, Any]]):
        """Process chunks and store with embeddings"""
        try:
            # Extract text content from chunks
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Store chunks with embeddings
            self.db.store_chunks(chunks, embeddings)
            
            logger.info(f"Successfully processed {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            raise

#-------------------------------------------------------------------------------------------#
