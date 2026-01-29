#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from loguru import logger

# Get project root and setup Python path
project_root = Path(__file__).parent.absolute()
from utils import setup_python_path
setup_python_path()

# Third-party imports
import asyncio
import torch
from prettytable import PrettyTable

# Core imports
from core.config import Config
from core.query import QueryEngine

# Configure logging
logger.remove()
logger.add(sys.stderr, 
          format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/search.log", rotation="500 MB")


class OrthomolecularSearchCLI:
    def __init__(self):
        """Initialize search interface with query engine"""
        self.query_engine = QueryEngine()
        
        # Set consistent widths
        self.HEADER_WIDTH = 70
        self.CONTENT_WIDTH = 100
        
        # Log GPU status
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def create_pretty_table(self, max_width=None):
        """Create a formatted PrettyTable"""
        table = PrettyTable()
        table.field_names = ["Content"]
        table.align = "l"  # Left align content
        if max_width:
            table._max_width = {"Content": max_width}
        table.header = False  # Hide header
        return table

    def create_header_table(self):
        """Create a combined header table with title, device info, and instructions"""
        table = self.create_pretty_table(max_width=self.HEADER_WIDTH)
        
        # Get device info
        device_info = ("Using GPU: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "💻 Running on CPU"
        
        # Create combined header content
        header_content = (
            "Orthomolecular Medicine Search\n"
            f"{'=' * self.HEADER_WIDTH}\n"
            f"{device_info}\n"
            "Enter 'exit' to quit"
        )
        
        table.add_row([header_content])
        return table

    def format_text_block(self, title, content, width=None):
        """Create a formatted text block with title and content"""
        table = self.create_pretty_table(max_width=width)
        if title:
            formatted_content = f"{title}\n{'-' * (width if width else len(title))}\n{content}"
        else:
            formatted_content = content
        table.add_row([formatted_content])
        return table

    def print_results(self, results, query, generated_response):
        """Print formatted search results and response"""
        print("\n📚 Search Results:")
        print("=" * self.CONTENT_WIDTH)
        
        # Print response with same width as chunks
        response_block = self.format_text_block(
            "Generated Response",
            generated_response,
            width=self.CONTENT_WIDTH
        )
        print("\n" + str(response_block))
        
        # Print results
        # print("\nMatching Text Chunks:")
        for i, result in enumerate(results, 1):
            score    = result.get('score', 0.0)
            chunk_id = result.get('chunk_id', 'N/A')
            content  = result.get('content', '')[:500]
            
            chunk_text = (
                f"Chunk {i} (Similarity: {score:.4f})\n"
                f"ID: {chunk_id}\n"
                f"Content:\n{content}..."
            )
            
            chunk_block = self.format_text_block(
                "",
                chunk_text,
                width=self.CONTENT_WIDTH
            )
            print("\n" + str(chunk_block))

    async def search_loop(self):
        """Interactive search loop"""
        # Print combined header
        header_table = self.create_header_table()
        print("\n" + str(header_table))
        
        while True:
            try:
                query = input("\nEnter your question: ").strip()
                if query.lower() == 'exit':
                    break

                if not query:
                    print("Please enter a valid question.")
                    continue

                print("\n🔎 Searching...")
                
                # Perform search
                results = await self.query_engine.search(query, top_k=Config.TOP_K)
                
                # Generate response
                response = await self.query_engine.generate_response(query, results)
                
                # Display results
                self.print_results(results, query, response)
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                error_table = self.create_pretty_table(max_width=self.CONTENT_WIDTH)
                error_table.add_row([f"❌ Error: {str(e)}"])
                print("\n" + str(error_table))

def main():
    try:
        cli = OrthomolecularSearchCLI()
        asyncio.run(cli.search_loop())
    except KeyboardInterrupt:
        goodbye_table             = PrettyTable()
        goodbye_table.field_names = ["Message"]
        goodbye_table.align       = "l"
        goodbye_table.header      = False
        goodbye_table.add_row(["👋 Goodbye!"])
        print("\n" + str(goodbye_table))
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        error_table = PrettyTable()
        error_table.field_names = ["Message"]
        error_table.align = "l"
        error_table.header = False
        error_table.add_row([f"❌ Fatal Error: {str(e)}"])
        print("\n" + str(error_table))
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
