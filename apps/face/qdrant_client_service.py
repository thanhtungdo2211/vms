import os
import uuid
import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from qdrant_client.http.exceptions import UnexpectedResponse

def _load_qdrant_config() -> dict:
    """Load Qdrant config from config.yaml located next to this file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("qdrant", {})
    except Exception as e:
        print(f"[qdrant_client_service] Failed to load config.yaml: {e}, using defaults")
        return {}

_cfg = _load_qdrant_config()

# Module-level constants — read from config.yaml, can still be overridden at runtime
QDRANT_HOST = _cfg.get("host", "192.168.6.16")
QDRANT_PORT = int(_cfg.get("port", 6386))
QDRANT_API_KEY = _cfg.get("api_key", None)
COLLECTION_NAME = _cfg.get("collection", "user_features_cosine")


class QdrantFeatureStorage:
    def __init__(self):
        # Always read module-level constants at instantiation time (not at import time)
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
                        distance=Distance.COSINE
                    )
                )
                print(f"Collection '{self.collection_name}' created")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise