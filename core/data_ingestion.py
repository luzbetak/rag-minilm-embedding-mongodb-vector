# core/data_ingestion.py
from typing import List, Dict, Any
import json
import re
from loguru import logger
from core.database import Database

class DataIngestionPipeline:
    def __init__(self):
        self.db = Database()
        logger.info("Initialized data ingestion pipeline")
    
    def load_text(self, file_path: str) -> str:
        """Load content from a text file"""
        logger.info(f"Loading text from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize line endings
        text = re.sub(r'\n\s*\n', '\n', text)
        text = ' '.join(text.split())
        return text
    
    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Split text into overlapping chunks"""
        logger.info("Creating text chunks")
        chunks = []
        text = self.clean_text(text)
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position for current chunk
            end = start + chunk_size
            
            # Find sentence boundary for chunk end
            if end < text_length:
                window_end = min(end + overlap, text_length)
                next_period = text.rfind('.', end - overlap, window_end)
                if next_period != -1:
                    end = next_period + 1

            # Extract chunk
            chunk_text = text[start:end].strip()
            
            # Create chunk object
            chunk_obj = {
                "chunk_id": len(chunks),
                "content": chunk_text,
                "start_char": start,
                "end_char": end,
                "length": len(chunk_text)
            }
            chunks.append(chunk_obj)

            # Move to next chunk
            if end == text_length:
                break
            
            # Start from last sentence boundary
            last_period = chunk_text.rfind('.')
            if last_period != -1:
                start = start + last_period + 1
            else:
                start = end - overlap

        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def save_chunks(self, chunks: List[Dict], output_file: str):
        """Save chunks to JSON file"""
        logger.info(f"Saving chunks to {output_file}")
        try:
            output_data = {
                "document_chunks": chunks,
                "metadata": {
                    "total_chunks": len(chunks),
                    "chunk_size": 1000,
                    "overlap": 200
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info("Chunks saved successfully")
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            raise
    
    def load_data(self, file_path: str) -> Dict:
        """Load data from JSON file"""
        logger.info(f"Loading data from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
