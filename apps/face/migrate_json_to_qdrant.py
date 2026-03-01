#!/usr/bin/env python3
"""
Migrate face features from JSON to Qdrant vector database.

Usage:
    # Default: reads data/face/features.json, connects to localhost:6333
    python3 apps/face/migrate_json_to_qdrant.py

    # Custom paths / host
    python3 apps/face/migrate_json_to_qdrant.py --json data/face/features.json --host localhost --port 6333

    # Force re-import even if collection already has data
    python3 apps/face/migrate_json_to_qdrant.py --force

    # Recreate collection (drops existing and creates fresh)
    python3 apps/face/migrate_json_to_qdrant.py --recreate

Collection schema:
    name: "faces"
    dim:  512
    distance: EUCLID  (matches existing np.linalg.norm L2 matching)

Point payload:
    {
        "name":      str,            # person name
        "avatar":    str | null,     # optional base64/path
        "source":    "manual",       # origin tag
        "type":      "normal"|"mask" # normal face or masked-face variant
        "image_idx": int,            # image index (supports multiple images per user)
    }

Point IDs are deterministic uuid5(name_imgN_type) so re-running is idempotent.
Multiple images per user supported: each entry in features.json can have a list
of features, or a single feature (backward compatible).
"""

import argparse
import json
import os
import sys
import time
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION = "faces"
DIM = 512
BATCH_SIZE = 100

# Fixed namespace – do NOT change after first import (IDs must stay stable)
_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_point_id(name: str, image_idx: int, suffix: str) -> str:
    """Deterministic UUID5 from (name, image_idx, suffix) — supports multiple images per user."""
    return str(uuid.uuid5(_NS, f"{name}_img{image_idx}_{suffix}"))


def normalize(vec) -> list[float]:
    """L2-normalize a feature vector (handle nested [[...]] format)."""
    arr = np.array(vec, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def ensure_collection(client: QdrantClient, recreate: bool = False) -> bool:
    """
    Create collection if it does not exist.
    If recreate=True, drops existing collection first.
    Returns True if newly created.
    """
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        if recreate:
            print(f"[Migrate] --recreate: Dropping existing collection '{COLLECTION}' ...")
            client.delete_collection(COLLECTION)
            print(f"[Migrate] Collection '{COLLECTION}' dropped.")
        else:
            return False

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
    )
    print(f"[Migrate] Created collection '{COLLECTION}' (dim={DIM}, distance=EUCLID)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Migration
# ─────────────────────────────────────────────────────────────────────────────

def migrate(json_path: str, host: str = "localhost", port: int = 6333,
            force: bool = False, recreate: bool = False) -> None:
    if not os.path.exists(json_path):
        print(f"[ERROR] features.json not found: {json_path}")
        sys.exit(1)

    print(f"[Migrate] Connecting to Qdrant at {host}:{port} ...")
    client = QdrantClient(host=host, port=port)

    # Collection setup
    created = ensure_collection(client, recreate=recreate)
    if not created and not recreate:
        count = client.count(COLLECTION).count
        if count > 0 and not force:
            print(
                f"[Migrate] Collection '{COLLECTION}' already has {count} point(s). "
                "Use --force to re-import or --recreate to drop and recreate."
            )
            return
        if count > 0:
            print(f"[Migrate] --force: re-importing over {count} existing point(s).")

    # Load JSON
    print(f"[Migrate] Loading {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points: list[PointStruct] = []
    skipped = 0
    start = time.time()

    for name, info in data.items():
        if "feature" not in info:
            print(f"[Migrate] SKIP '{name}' — no 'feature' key")
            skipped += 1
            continue

        avatar = info.get("avatar")

        # ── Support multiple images: "features" list OR single "feature" ──
        # "features" key: list of feature vectors (multiple images)
        # "feature"  key: single vector (legacy, image_idx=0)
        feature_list = info.get("features")  # list of feature vectors
        mask_list = info.get("feature_masks")  # list of mask vectors (optional)

        if feature_list is None:
            # Legacy single-feature format
            feature_list = [info["feature"]]
            mask_list = [info["feature_mask"]] if "feature_mask" in info else [None]
        elif mask_list is None:
            mask_list = [None] * len(feature_list)

        person_added = 0
        for image_idx, (feat, feat_mask) in enumerate(zip(feature_list, mask_list)):
            # ── Normal feature ──
            vec = normalize(feat)
            if len(vec) != DIM:
                print(f"[Migrate] SKIP '{name}' image_{image_idx} — unexpected dim {len(vec)}")
                skipped += 1
                continue

            points.append(PointStruct(
                id=make_point_id(name, image_idx, "normal"),
                vector=vec,
                payload={
                    "name": name,
                    "avatar": avatar,
                    "source": "manual",
                    "type": "normal",
                    "image_idx": image_idx,
                },
            ))
            person_added += 1

            # ── Masked feature (optional) ──
            if feat_mask is not None:
                vec_mask = normalize(feat_mask)
                if len(vec_mask) == DIM:
                    points.append(PointStruct(
                        id=make_point_id(name, image_idx, "mask"),
                        vector=vec_mask,
                        payload={
                            "name": name,
                            "avatar": avatar,
                            "source": "manual",
                            "type": "mask",
                            "image_idx": image_idx,
                        },
                    ))

        if person_added > 0:
            print(f"[Migrate] '{name}': {person_added} image(s) queued.")

    if not points:
        print("[Migrate] No valid features found. Nothing upserted.")
        return

    # Upsert in batches
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION, points=batch)
        done = min(i + BATCH_SIZE, len(points))
        print(f"[Migrate] Upserted {done}/{len(points)} points ...")

    elapsed_ms = (time.time() - start) * 1000
    final_count = client.count(COLLECTION).count
    print(
        f"\n[Migrate] Done in {elapsed_ms:.1f}ms.\n"
        f"  Points upserted : {len(points)}\n"
        f"  Entries skipped : {skipped}\n"
        f"  Collection total: {final_count}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate face features from features.json to Qdrant"
    )
    parser.add_argument(
        "--json",
        default="data/face/features.json",
        help="Path to features.json (default: data/face/features.json)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("QDRANT_HOST", "localhost"),
        help="Qdrant host (default: localhost, or $QDRANT_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", "6333")),
        help="Qdrant port (default: 6333, or $QDRANT_PORT)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-import even if collection already has data (upsert is idempotent)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before importing (clears all existing data)",
    )
    args = parser.parse_args()

    migrate(args.json, args.host, args.port, args.force, args.recreate)