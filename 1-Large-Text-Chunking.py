#!/usr/bin/env python3

#----------------------------------------------------------------------------------------#
import os
import sys
from pathlib import Path
from loguru import logger

#----------------------------------------------------------------------------------------#
# Get project root and setup Python path
project_root = Path(__file__).parent.absolute()
from utils import setup_python_path
setup_python_path()

#----------------------------------------------------------------------------------------#
from core.config import Config
from core.data_ingestion import DataIngestionPipeline
from core.vectorization import VectorizationPipeline

#----------------------------------------------------------------------------------------#
# Configure logging
os.makedirs("logs", exist_ok=True)
logger.add("logs/chunking.log", rotation="500 MB")

#----------------------------------------------------------------------------------------#
def process_large_text():
    """Process Large Text into Chunks"""
    try:
        # Initialize pipelines
        data_pipeline = DataIngestionPipeline()
        vector_pipeline = VectorizationPipeline()
        
        # Input file path
        input_file = os.path.join(project_root, "source", "The-Gerson-Therapy-Reduced.txt")
        
        # Read and chunk text
        logger.info(f"Processing file: {input_file}")
        text = data_pipeline.load_text(input_file)
        
        # Create chunks using config settings
        chunks = data_pipeline.create_chunks(
            text,
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP
        )
        
        # Save chunks to JSON for verification
        output_dir = os.path.join(project_root, "data")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "large_text_chunks.json")
        data_pipeline.save_chunks(chunks, output_file)
        
        logger.info(f"Created {len(chunks)} chunks")
        logger.info(f"Saved chunks to {output_file}")
        
        print(f"\n✅ Successfully processed text into {len(chunks)} chunks")
        print(f"📁 Output saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        print(f"\n❌ Error: {str(e)}")

#----------------------------------------------------------------------------------------#
def main():
    print("\nText Chunking")
    print("=====================================")
    
    try:
        process_large_text()
    except KeyboardInterrupt:
        print("\n👋 Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {str(e)}")

#----------------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
