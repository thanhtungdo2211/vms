from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue

import numpy as np
import uuid

QDRANT_HOST='qdrant_ds'
QDRANT_PORT=6386
QDRANT_API_KEY=None  # Leave empty if no authentication needed

qdrant_client = QdrantClient(
    host=QDRANT_HOST, 
    port=QDRANT_PORT,  # ← Added missing comma here
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
)

# Create collection with vector shape (1, 512) and L2 distance
collection_name = "my_vectors"

try:
    # Check if collection already exists
    existing_collections = qdrant_client.get_collections()
    collection_names = [col.name for col in existing_collections.collections]
    
    # if collection_name not in collection_names:
    #     # Create new collection
    #     qdrant_client.create_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(
    #             size=512,  # Vector dimension is 512
    #             distance=Distance.EUCLID  # L2 Euclidean distance
    #         )
    #     )
    #     print(f"Collection '{collection_name}' created successfully")
    # else:
    #     print(f"Collection '{collection_name}' already exists")
    
    # List all collections
    collections = qdrant_client.get_collections()
    for col in collections.collections:
        print(f"  - {col.name}")
    
    # Generate sample feature vectors (512-dimensional)
    sample_vectors = [
        {
            "id": 1,
            "vector": np.random.rand(512).tolist(),
            "metadata": {
                "person_id": "person_001", 
                "timestamp": "2024-01-15T10:30:00",
                "camera_id": "cam_01",
                "confidence": 0.95
            }
        },
        {
            "id": 2,
            "vector": np.random.rand(512).tolist(),
            "metadata": {
                "person_id": "person_002", 
                "timestamp": "2024-01-15T10:35:00",
                "camera_id": "cam_02",
                "confidence": 0.87
            }
        },
        {
            "id": 3,
            "vector": np.random.rand(512).tolist(),
            "metadata": {
                "person_id": "person_003", 
                "timestamp": "2024-01-15T10:40:00",
                "camera_id": "cam_01",
                "confidence": 0.92
            }
        }
    ]
    
#     # Upsert vectors
    points = [
        PointStruct(
            id=vec["id"],
            vector=vec["vector"],
            payload=vec["metadata"]
        )
        for vec in sample_vectors
    ]
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    # Use the first vector as query
    query_vector = sample_vectors[0]["vector"]
    
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5,  # Return top 5 most similar
        with_payload=True,  # Include metadata
        with_vectors=False  # Don't return vectors (save bandwidth)
    )
    print(search_results)
    # print(f" Found {len(search_results)} similar vectors:")
    for i, result in enumerate(search_results):
        print(f"  {i+1}. ID: {result.id}")
        print(f"     Score: {result.score:.4f}")
        print(f"     Person ID: {result.payload.get('person_id', 'Unknown')}")
        print(f"     Camera: {result.payload.get('camera_id', 'Unknown')}")
        print(f"     Confidence: {result.payload.get('confidence', 'Unknown')}")
    
    # SEARCH WITH FILTERS EXAMPLE
    filtered_search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="camera_id",
                    match=MatchValue(value="cam_01")
                )
            ]
        ),
        limit=3,
        with_payload=True
    )
    
    print(f"Found {len(filtered_search_results)} vectors from cam_01:")
    for i, result in enumerate(filtered_search_results):
        print(f"  {i+1}. ID: {result.id}")
        print(f"     Score: {result.score:.4f}")
        print(f"     Person ID: {result.payload.get('person_id', 'Unknown')}")
        print()
    
    # BATCH UPSERT EXAMPLE: Insert multiple vectors at once
    print("\n Batch upserting more vectors...")
    
    batch_points = []
    for i in range(4, 10):  # Add IDs 4-9
        batch_points.append(
            PointStruct(
                id=i,
                vector=np.random.rand(512).tolist(),
                payload={
                    "person_id": f"person_{i:03d}",
                    "timestamp": f"2024-01-15T{10 + i}:00:00",
                    "camera_id": f"cam_{(i % 3) + 1:02d}",
                    "confidence": np.random.uniform(0.7, 0.99)
                }
            )
        )
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=batch_points
    )
    print(f"Batch upserted {len(batch_points)} additional vectors")
    
    # Count total vectors in collection
    collection_info = qdrant_client.get_collection(collection_name)
    print(f"\n Total vectors in collection: {collection_info.points_count}")
    
    # SEARCH BY ID EXAMPLE
    print("\n Retrieving vector by ID...")
    
    retrieved_points = qdrant_client.retrieve(
        collection_name=collection_name,
        ids=[1, 2],
        with_payload=True,
        with_vectors=False
    )
    
    print(f" Retrieved {len(retrieved_points)} vectors by ID:")
    for point in retrieved_points:
        print(f"  ID: {point.id}")
        print(f"  Person ID: {point.payload.get('person_id', 'Unknown')}")
        print(f"  Camera: {point.payload.get('camera_id', 'Unknown')}")
        print()
        
except Exception as e:
    print(f"❌ Error: {e}")