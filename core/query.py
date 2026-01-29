# query.py
from typing import List, Dict, Optional
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from loguru import logger
from core.config import Config
from core.database import Database
from core.vectorization import VectorizationPipeline


class QueryEngine:
    def __init__(self):
        """Initialize the query engine with necessary components"""
        self.db = Database()
        self.vectorization = VectorizationPipeline()
        
        # Set up device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize BART model and tokenizer explicitly
        # This avoids pipeline task name compatibility issues
        model_name = "facebook/bart-large-cnn"
        logger.info(f"Loading summarization model: {model_name}")
        
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
        
        self.model.eval()  # Set to evaluation mode
        
        logger.info("Initialized orthomolecular query engine")

    async def search(self, query: str, top_k: int = Config.TOP_K) -> List[dict]:
        """Perform vector similarity search on orthomolecular chunks"""
        try:
            # Generate query embedding
            query_embedding = self.vectorization.generate_embeddings([query])[0]
            logger.info(f"Searching for: {query}")
            
            # Get similar chunks
            similar_chunks = self.db.get_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            logger.info(f"Found {len(similar_chunks)} relevant chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    async def generate_response(self, query: str, chunks: List[dict]) -> str:
        """Generate a response based on the query and retrieved chunks"""
        # Clear GPU memory before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            if not chunks:
                return "No relevant information found in the orthomolecular medicine text."
                
            # Combine chunk contents
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                score = chunk.get('score', 0.0)
                content = chunk.get('content', '')
                context_parts.append(
                    f"Chunk {i} (Relevance: {score:.3f}):\n{content}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Tokenize input with truncation (BART max length is 1024)
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=Config.MAX_LENGTH,
                    min_length=Config.MIN_LENGTH,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the generated summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Format final response
            response = (
                f"Based on the orthomolecular medicine text:\n\n"
                f"{summary}"
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return "I apologize, but I encountered an error generating a response."

    def close(self):
        """Cleanup resources"""
        self.db.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
