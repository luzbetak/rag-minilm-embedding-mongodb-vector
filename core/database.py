#---------------------------------------------------------------------------------------#
# database.py
#---------------------------------------------------------------------------------------#
from pymongo import MongoClient, ReplaceOne
from loguru import logger
import numpy as np
from core.config import Config

#---------------------------------------------------------------------------------------#
class Database:
    def __init__(self):
        """Initialize MongoDB client for large text database"""
        try:
            self.client     = MongoClient(Config.MONGODB_URI)
            self.db         = self.client[Config.DATABASE_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            self.client.server_info()
            
            # Create indices
            self.collection.create_index([("chunk_id", 1)], unique=True)
            self.collection.create_index([("content", "text")])
            
            logger.info(f"Connected to MongoDB - Database: {self.db.name}")
            count = self.collection.count_documents({})
            logger.info(f"Current chunk count: {count}")
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    #----------------------------------------------------------------------------------#
    def get_similar_chunks(self, query_embedding, top_k=Config.TOP_K):
        """Find similar chunks using vector similarity search"""
        try:
            query_embedding = (query_embedding.tolist() 
                             if isinstance(query_embedding, np.ndarray) 
                             else query_embedding)

            pipeline = [
                {
                    "$set": {
                        "similarity": {
                            "$reduce": {
                                "input": {"$range": [0, {"$size": "$embedding"}]},
                                "initialValue": 0,
                                "in": {
                                    "$add": [
                                        "$$value",
                                        {
                                            "$multiply": [
                                                {"$arrayElemAt": ["$embedding", "$$this"]},
                                                {"$arrayElemAt": [query_embedding, "$$this"]}
                                            ]
                                        }
                                    ]
                                }
                            }
                        }
                    }
                },
                {"$sort": {"similarity": -1}},
                {"$limit": top_k},
                {
                    "$project": {
                        "_id": 0,
                        "chunk_id": 1,
                        "content": 1,
                        "start_char": 1,
                        "end_char": 1,
                        "score": "$similarity"
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} similar chunks")
            return results

        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []

    #----------------------------------------------------------------------------------#
    def store_chunks(self, chunks, embeddings):
        """Store text chunks with their embeddings"""
        if not chunks or not embeddings:
            logger.error("No chunks or embeddings to store")
            return

        operations = []
        for chunk, embedding in zip(chunks, embeddings):
            if not all(key in chunk for key in ["chunk_id", "content"]):
                logger.warning(f"Skipping chunk missing required fields: {chunk}")
                continue

            embedding = (embedding.tolist() 
                        if isinstance(embedding, np.ndarray) 
                        else embedding)

            operations.append(
                ReplaceOne(
                    {"chunk_id": chunk["chunk_id"]},
                    {
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "start_char": chunk["start_char"],
                        "end_char": chunk["end_char"],
                        "length": chunk["length"],
                        "embedding": embedding
                    },
                    upsert=True
                )
            )

        if operations:
            try:
                result = self.collection.bulk_write(operations)
                logger.info(f"Chunks stored: {len(operations)}")
                logger.info(f"Inserted: {result.upserted_count}")
                logger.info(f"Modified: {result.modified_count}")
                return result
            except Exception as e:
                logger.error(f"Error storing chunks: {e}")
                raise

    #----------------------------------------------------------------------------------#
    def close(self):
        """Close database connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("Database connection closed")

#---------------------------------------------------------------------------------------#
