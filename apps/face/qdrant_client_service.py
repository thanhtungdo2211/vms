import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from qdrant_client.http.exceptions import UnexpectedResponse

# Qdrant configuration
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6386
QDRANT_API_KEY = None
COLLECTION_NAME = "user_features_cosine"

class QdrantFeatureStorage:
    def __init__(self):
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        self.collection_name = COLLECTION_NAME
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't"""
        try:
            existing_collections = self.client.get_collections()
            print("Colection", existing_collections)
            collection_names = [col.name for col in existing_collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=512,
                        # distance=Distance.EUCLID  # Better for face embeddings
                        distance=Distance.COSINE  # Better for face embeddings
                    )
                )
                print(f"Collection '{self.collection_name}' created")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise