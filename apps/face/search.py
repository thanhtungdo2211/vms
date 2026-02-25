import uuid
import time
from typing import List, Union, Any, Dict

from qdrant_client.models import PointStruct  # type: ignore

from apps.face.qdrant_client_service import QdrantFeatureStorage

feature_storage = QdrantFeatureStorage()


def upsert(
    person_id: str,
    features: Union[List[float], List[List[float]]],
    camera_id: str,
) -> Dict[str, Any]:
    try:
        import numpy as np
        points = []
        for feature_vector in features:
            if len(feature_vector) != 512:
                raise ValueError(f"Feature vector must be 512-dimensional, got {len(feature_vector)}")

            arr = np.array(feature_vector, dtype=np.float32)
            n = np.linalg.norm(arr)
            if n > 0:
                arr = arr / n

            payload = {"person_id": person_id, "camera_id": camera_id}
            feature_id = uuid.uuid4().hex
            point = PointStruct(id=feature_id, vector=arr.tolist(), payload=payload)
            points.append(point)

        operation_info = feature_storage.client.upsert(
            collection_name=feature_storage.collection_name,
            points=points
        )

        return {
            "status": "success",
            "operation_id": getattr(operation_info, "operation_id", None),
            "upserted_count": len(points),
            "person_id": person_id
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "upserted_count": 0}


def _do_search(query_vector_normalized: list, limit: int):
    """
    Search Qdrant using the best available API method.
    Supports both old (.search) and new (.query_points) qdrant-client versions.
    
    Returns list of ScoredPoint objects.
    """
    client = feature_storage.client
    collection = feature_storage.collection_name

    # Try .search() first (qdrant-client < 1.7)
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection,
            query_vector=query_vector_normalized,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    # Fallback to .query_points() (qdrant-client >= 1.7)
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=collection,
            query=query_vector_normalized,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        # query_points returns QueryResponse with .points attribute
        if hasattr(result, "points"):
            return result.points
        return result

    raise RuntimeError(
        "qdrant-client has neither .search() nor .query_points(). "
        "Please upgrade: pip install qdrant-client>=1.6"
    )


def search(
    query_vector: List[float],
    limit: int = 30,
    similarity_threshold: float = 0.45
):
    """
    Search similar vectors in Qdrant.
    - Compatible with all qdrant-client versions
    - Collection metric must be Cosine for score ∈ [0, 1]
    - Higher score = more similar
    """
    try:
        if len(query_vector) != 512:
            raise ValueError(f"Query vector must be 512-dimensional, got {len(query_vector)}")

        import numpy as np
        qv = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm

        search_results = _do_search(qv.tolist(), limit)

        # Filter by threshold (cosine: score ∈ [0,1], higher = better)
        hits = [res for res in search_results if res.score >= similarity_threshold]

        return hits
    except Exception as e:
        print(f"Search error: {e}")
        return []


def check_and_save_feature(
    feature_vector: List[float],
    person_id: str,
    camera_id: str,
    limit: int = 10,
    similarity_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Check if feature already exists (by cosine similarity),
    if not, upsert into Qdrant.
    """
    try:
        import numpy as np
        fv = np.array(feature_vector, dtype=np.float32)
        norm = np.linalg.norm(fv)
        if norm > 0:
            fv = fv / norm

        res = search(
            query_vector=fv.tolist(),
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

        if res:
            best_score = max(r.score for r in res)
            best_pid = res[0].payload.get("person_id") if res[0].payload else None
            if best_score >= similarity_threshold:
                return {
                    "status": "skipped",
                    "reason": f"similar feature already exists (score={best_score:.3f})",
                    "person_id": best_pid or person_id
                }

        up = upsert(person_id=person_id,
                     features=[fv.tolist()],
                     camera_id=camera_id)
        return up

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }