"""
Register face embeddings into Qdrant vector database.

Usage:
    # Register a single person (single image)
    python3 apps/face/extract.py --name tungdt --image data/face/face-import-pics/tungdt.jpg

    # Register a single person with multiple images (dir of one person)
    python3 apps/face/extract.py --name tungdt --images data/face/face-import-pics/tungdt/

    # Register all people (each subdir = person name, images inside = that person's images)
    python3 apps/face/extract.py --dir data/face/face-import-pics/

    # Register flat directory (filename without extension = person name, one image per person)
    python3 apps/face/extract.py --dir data/face/face-import-pics/ --flat

    # Custom Qdrant host
    python3 apps/face/extract.py --name tungdt --image tungdt.jpg --host localhost --port 6333

    # Delete a person from database
    python3 apps/face/extract.py --delete tungdt

Two embeddings are stored per image (upserted by deterministic UUID5 IDs, so safe to re-run):
  - type="normal" : full face
  - type="mask"   : lower-right quadrant zeroed (rough masked-face simulation)

Multiple images per user are supported:
  - IDs are uuid5(name_imageN_normal) / uuid5(name_imageN_mask)
  - Re-running with same images is idempotent (same ID = upsert overwrites)
"""

import argparse
import os
import uuid

import cv2
import numpy as np

# Local imports (run from /app or project root)
from scr_onnx import SCRFD
from apps.face.arcface_onnx import IRES
from utils import align_face

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION = "faces"
DIM = 512
_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")  # same namespace as migrate script

ARCFACE_PATH = os.getenv(
    "ARCFACE_MODEL",
    "/home/jetson/tungdt/vms/data/face/models/arcface/arcface_r100.onnx",
)
SCRFD_PATH = os.getenv(
    "SCRFD_MODEL",
    "/home/jetson/tungdt/vms/data/face/models/scrfd640/scrfd_2.5g_bnkps_dynamic.onnx",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_point_id(name: str, image_idx: int, suffix: str) -> str:
    """Deterministic ID: supports multiple images per user via image_idx."""
    return str(uuid.uuid5(_NS, f"{name}_img{image_idx}_{suffix}"))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
        )
        print(f"[extract] Created collection '{COLLECTION}'")


# ─────────────────────────────────────────────────────────────────────────────
# SaveFeature
# ─────────────────────────────────────────────────────────────────────────────

class SaveFeature:
    """Detect, align, extract, and upsert face embeddings into Qdrant.
    
    Supports multiple images per user: each image gets its own pair of
    (normal, mask) vectors with a unique image index in the point ID.
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.iresnet = IRES(ARCFACE_PATH)
        self.scrfd = SCRFD(model_file=SCRFD_PATH)
        self.scrfd.prepare(-1)

        self.client = QdrantClient(host=host, port=port)
        _ensure_collection(self.client)

    def _get_next_image_idx(self, name: str) -> int:
        """Find the next available image index for this user."""
        existing, _ = self.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name))]
            ),
            with_payload=True,
            with_vectors=False,
            limit=1000,
        )
        if not existing:
            return 0
        # Extract existing image indices from payload
        indices = set()
        for point in existing:
            idx = (point.payload or {}).get("image_idx", 0)
            indices.add(idx)
        # Return next index
        i = 0
        while i in indices:
            i += 1
        return i

    def save(self, name: str, image: np.ndarray, image_idx: int | None = None) -> bool:
        """
        Detect face in image, extract embeddings, upsert both normal and
        masked variants to Qdrant.

        Args:
            name: Person name
            image: BGR image (numpy array)
            image_idx: Image index for this person (auto-detected if None)

        Returns True if face was found and saved, False otherwise.
        Point IDs are deterministic (uuid5), so calling save() with same
        name+image_idx overwrites the previous entry (safe for re-registration).
        """
        bboxes, kpss = self.scrfd.detect(image, 0.5, input_size=(640, 640))

        if len(bboxes) == 0:
            print(f"[extract] No face detected in image for '{name}'.")
            return False
        if len(bboxes) > 1:
            print(f"[extract] {len(bboxes)} faces detected for '{name}' — using best-score face.")

        # Pick the face with highest detection score
        best = int(np.argmax([b[4] for b in bboxes]))
        x1, y1, x2, y2 = bboxes[best][:4]
        lm = kpss[best]

        face_aligned = align_face(image.copy(), [x1, y1, x2, y2], lm)

        # ── Normal embedding ──
        feature = _normalize(self.iresnet.predict(face_aligned.copy()).flatten())

        # ── Masked embedding (zero out lower-right quadrant) ──
        masked = face_aligned.copy()
        masked[65:, 65:, :] = 0
        feature_mask = _normalize(self.iresnet.predict(masked).flatten())

        # Auto-detect image index if not provided
        if image_idx is None:
            image_idx = self._get_next_image_idx(name)

        points = [
            PointStruct(
                id=_make_point_id(name, image_idx, "normal"),
                vector=feature.tolist(),
                payload={
                    "name": name,
                    "source": "manual",
                    "type": "normal",
                    "image_idx": image_idx,
                },
            ),
            PointStruct(
                id=_make_point_id(name, image_idx, "mask"),
                vector=feature_mask.tolist(),
                payload={
                    "name": name,
                    "source": "manual",
                    "type": "mask",
                    "image_idx": image_idx,
                },
            ),
        ]

        self.client.upsert(collection_name=COLLECTION, points=points)
        count = self.client.count(COLLECTION).count
        print(
            f"[extract] Upserted '{name}' image_idx={image_idx} "
            f"(normal + mask). Collection total: {count}"
        )
        return True

    def delete(self, name: str) -> int:
        """Delete all vectors for a person. Returns count of deleted points."""
        self.client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=name))]
            ),
        )
        # Count remaining
        total = self.client.count(COLLECTION).count
        print(f"[extract] Deleted all vectors for '{name}'. Collection total: {total}")
        return total

    def list_users(self) -> dict[str, int]:
        """Return {name: vector_count} for all users."""
        users: dict[str, int] = {}
        offset = None
        while True:
            result, next_offset = self.client.scroll(
                collection_name=COLLECTION,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in result:
                name = (point.payload or {}).get("name", "unknown")
                users[name] = users.get(name, 0) + 1
            if next_offset is None:
                break
            offset = next_offset
        return users


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register face(s) into Qdrant")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single face image (use with --name)")
    group.add_argument("--images", help="Directory of images for ONE person (use with --name)")
    group.add_argument("--dir", help="Directory: flat mode (filename=name) or subdir mode (subdir=name)")
    group.add_argument("--delete", metavar="NAME", help="Delete all vectors for a person")
    group.add_argument("--list", action="store_true", help="List all registered users")
    parser.add_argument("--name", help="Person name (required when using --image or --images)")
    parser.add_argument("--flat", action="store_true",
                        help="With --dir: treat filenames as person names (one image per person)")
    parser.add_argument("--host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    args = parser.parse_args()

    if (args.image or args.images) and not args.name:
        parser.error("--name is required when using --image or --images")

    save_fea = SaveFeature(host=args.host, port=args.port)
    supported = {".jpg", ".jpeg", ".png", ".bmp"}

    # ── --list ──
    if args.list:
        users = save_fea.list_users()
        if not users:
            print("[extract] No users registered.")
        else:
            print(f"[extract] Registered users ({len(users)}):")
            for name, count in sorted(users.items()):
                print(f"  {name}: {count} vector(s)")
        raise SystemExit(0)

    # ── --delete ──
    if args.delete:
        save_fea.delete(args.delete)
        raise SystemExit(0)

    # ── --image (single image, single person) ──
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"[ERROR] Cannot read image: {args.image}")
        ok = save_fea.save(args.name, img)
        raise SystemExit(0 if ok else 1)

    # ── --images (multiple images, single person) ──
    if args.images:
        success = fail = 0
        for idx, fname in enumerate(sorted(os.listdir(args.images))):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue
            img = cv2.imread(os.path.join(args.images, fname))
            if img is None:
                print(f"[extract] Cannot read {fname}, skipping.")
                fail += 1
                continue
            ok = save_fea.save(args.name, img, image_idx=idx)
            if ok:
                success += 1
            else:
                fail += 1
        print(f"\n[extract] Finished '{args.name}': {success} images saved, {fail} failed.")
        raise SystemExit(0 if success > 0 else 1)

    # ── --dir flat mode: filename = person name, one image per person ──
    if args.flat:
        success = fail = 0
        for fname in sorted(os.listdir(args.dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue
            name = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(args.dir, fname))
            if img is None:
                print(f"[extract] Cannot read {fname}, skipping.")
                fail += 1
                continue
            ok = save_fea.save(name, img, image_idx=0)
            if ok:
                success += 1
            else:
                fail += 1
        print(f"\n[extract] Finished: {success} registered, {fail} failed.")
        raise SystemExit(0)

    # ── --dir subdir mode: each subdir = one person, images inside = multiple images ──
    success_persons = fail_persons = success_imgs = fail_imgs = 0
    for entry in sorted(os.listdir(args.dir)):
        subdir = os.path.join(args.dir, entry)
        if not os.path.isdir(subdir):
            # Also handle flat images in root dir
            ext = os.path.splitext(entry)[1].lower()
            if ext in supported:
                name = os.path.splitext(entry)[0]
                img = cv2.imread(os.path.join(args.dir, entry))
                if img is None:
                    fail_imgs += 1
                    continue
                ok = save_fea.save(name, img, image_idx=0)
                if ok:
                    success_imgs += 1
                else:
                    fail_imgs += 1
            continue

        # Subdir found: entry = person name
        person_name = entry
        person_success = person_fail = 0
        for idx, fname in enumerate(sorted(os.listdir(subdir))):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue
            img = cv2.imread(os.path.join(subdir, fname))
            if img is None:
                print(f"[extract] Cannot read {fname}, skipping.")
                person_fail += 1
                continue
            ok = save_fea.save(person_name, img, image_idx=idx)
            if ok:
                person_success += 1
            else:
                person_fail += 1

        if person_success > 0:
            success_persons += 1
            success_imgs += person_success
        else:
            fail_persons += 1
        fail_imgs += person_fail

    print(
        f"\n[extract] Finished: "
        f"{success_persons} persons registered, {fail_persons} failed. "
        f"({success_imgs} images saved, {fail_imgs} skipped)"
    )