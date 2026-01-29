#!/usr/bin/env python3

#------------------------------------------------------------------#
import os
import sys
from pathlib import Path
from loguru import logger
import time

#------------------------------------------------------------------#
# Get project root and setup Python path
project_root = Path(__file__).parent.absolute()
from utils import setup_python_path
setup_python_path()

#------------------------------------------------------------------#
# Core imports
from core.config import Config
from core.database import Database
from core.vectorization import VectorizationPipeline
from core.data_ingestion import DataIngestionPipeline

#------------------------------------------------------------------#
# Configure logging
logger.remove()
logger.add(sys.stderr,
          format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/indexing.log", rotation="500 MB")

#------------------------------------------------------------------#
class OrthomolecularIndexer:
    def __init__(self, chunks_file='data/large_text_chunks.json'):
        """Initialize indexer with core components"""
        self.chunks_file = chunks_file
        self.db = Database()
        self.vectorizer = VectorizationPipeline()
        self.data_pipeline = DataIngestionPipeline()

    def init_database(self):
        """Initialize database with required indices"""
        try:
            logger.info("Initializing database...")
            self.db.collection.drop()
            self.db.collection.create_index([("chunk_id", 1)], unique=True)
            self.db.collection.create_index([("content", "text")])
            logger.info("Database initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False

    #------------------------------------------------------------------#
    def process_chunks(self):
        """Process and index chunks with embeddings"""
        try:
            # Load chunks
            chunks = self.data_pipeline.load_data(self.chunks_file)
            if not chunks:
                return False

            # Generate embeddings using vectorization pipeline
            self.vectorizer.process_chunks(chunks['document_chunks'])
            logger.info("Chunks processed and stored successfully")
            return True

        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            return False

    #------------------------------------------------------------------#
    def run_all_operations(self):
        """Run all operations in sequence"""
        try:
            logger.info("Starting all operations...")
            
            # Initialize database
            print("\n🔄 Initializing database...")
            if not self.init_database():
                raise Exception("Database initialization failed")
            print("✅ Database initialized successfully")
            
            # Process chunks
            print("\n🔄 Processing chunks and creating embeddings...")
            if not self.process_chunks():
                raise Exception("Chunk processing failed")
            print("✅ Chunks processed successfully")
            
            print("All operations completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in run_all_operations: {e}")
            print(f"\n❌ Error during operations: {str(e)}")
            return False

#------------------------------------------------------------------#
def display_header():
    """Display the application header"""
    print("\n" + "="*50)
    print("🧬 Orthomolecular Medicine Indexing System")
    print("="*50)

#------------------------------------------------------------------#
def main():
    display_header()
    indexer = OrthomolecularIndexer()

    menu_options = {
        "1": ("Initialize database (will delete existing data)", indexer.init_database),
        "2": ("Process chunks and create embeddings", indexer.process_chunks),
        "3": ("Run all operations and exit", lambda: run_all_and_exit(indexer)),
        "4": ("Exit", lambda: sys.exit(0))
    }

    #------------------------------------------------------------------#
    def run_all_and_exit(indexer):
        """Run all operations and exit if successful"""
        if indexer.run_all_operations():
            print("Exiting after successful completion...")
            time.sleep(2)  # Give user time to read the message
            sys.exit(0)

    while True:
        try:
            print("\nOptions:")
            for key, (description, _) in menu_options.items():
                print(f"{key}. {description}")

            choice = input("\nEnter your choice (1-4): ")

            if choice in menu_options:
                start_time = time.time()
                result = menu_options[choice][1]()
                end_time = time.time()
                
                if choice != "4":
                    if result:
                        print(f"\n✅ Operation completed in {end_time - start_time:.2f} seconds")
                    else:
                        print("\n❌ Operation failed")
                    
                    if choice != "3":  # Don't wait for input if running all operations
                        input("\nPress Enter to continue...")
            else:
                print("\n❌ Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {str(e)}")
            input("\nPress Enter to continue...")

#------------------------------------------------------------------#
if __name__ == "__main__":
    main()
