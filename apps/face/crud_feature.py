"""Remove wrongly auto-saved embeddings."""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)

# Xóa toàn bộ auto-saved của "tung" (sai)
client.delete(
    collection_name="faces",
    points_selector=Filter(must=[
        FieldCondition(key="name", match=MatchValue(value="tungdt")),
        FieldCondition(key="source", match=MatchValue(value="auto_save")),
    ])
)
print("Deleted auto-saved for 'tung'")
print(f"Total remaining: {client.count('faces').count}")