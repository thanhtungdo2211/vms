"""
Face Recognition Processor - Simplified Module

This module contains face recognition components with minimal class usage:
- FaceDatabase: Face feature storage and matching
- TrackedFace/TrackerManager: Identity tracking (dataclass + manager)
- FaceRecognitionProcessor: Main processor with probes

Auto-registered with ProcessorRegistry using @register decorator.
"""

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import BatchIterator, extract_embedding, get_batch_meta, fps_probe_factory, IntervalRunner


# =============================================================================
# Constants
# =============================================================================

# OSD Colors (RGBA)
COLOR_CONFIRMED = (0.0, 1.0, 0.0, 1.0)  # Green
COLOR_UNKNOWN = (1.0, 0.5, 0.0, 1.0)    # Orange
COLOR_TEXT = (1.0, 1.0, 1.0, 1.0)       # White
COLOR_TEXT_BG = (0.0, 0.0, 0.0, 0.7)    # Black transparent

# Display settings
BORDER_WIDTH = 3
FONT_SIZE = 14
FONT_NAME = "Serif"

# Face-specific constants
SKIP_SGIE_COMPONENT_ID = 100


# =============================================================================
# Face-Specific Helper Functions
# =============================================================================

def should_skip_face(obj_meta, min_size: int = 50) -> bool:
    """Check if face should be skipped based on size (face-specific logic)"""
    rect = obj_meta.rect_params
    return rect.width < min_size or rect.height < min_size


def mark_skip_sgie(obj_meta) -> None:
    """Mark object to skip SGIE processing"""
    obj_meta.unique_component_id = SKIP_SGIE_COMPONENT_ID


# =============================================================================
# Display Functions (replaces FaceDisplay class)
# =============================================================================

def update_display(obj_meta, name: str, score: float, state: str = "confirmed") -> None:
    """Update OSD display for a detected face"""
    rect = obj_meta.rect_params
    face_w, face_h = int(rect.width), int(rect.height)

    # Border color by state
    r, g, b, a = COLOR_CONFIRMED if state == "confirmed" else COLOR_UNKNOWN
    rect.border_color.red, rect.border_color.green = r, g
    rect.border_color.blue, rect.border_color.alpha = b, a
    rect.border_width = BORDER_WIDTH

    # Display text
    display_text = f"{name} ({score:.2f}) [{face_w}x{face_h}]" if state == "confirmed" else f"[{face_w}x{face_h}]"

    text = obj_meta.text_params
    text.display_text = display_text
    text.x_offset = int(rect.left)
    text.y_offset = max(0, int(rect.top) - 25)
    text.font_params.font_name = FONT_NAME
    text.font_params.font_size = FONT_SIZE

    # Text color
    r, g, b, a = COLOR_TEXT
    text.font_params.font_color.red, text.font_params.font_color.green = r, g
    text.font_params.font_color.blue, text.font_params.font_color.alpha = b, a

    # Text background
    text.set_bg_clr = 1
    r, g, b, a = COLOR_TEXT_BG
    text.text_bg_clr.red, text.text_bg_clr.green = r, g
    text.text_bg_clr.blue, text.text_bg_clr.alpha = b, a


# =============================================================================
# Face Database (Qdrant)
# =============================================================================

class QdrantFaceDatabase:
    """
    Face feature storage and matching backed by Qdrant vector database.

    Replaces JSON-based FaceDatabase. Connects to a running Qdrant instance
    and uses nearest-neighbour search (EUCLID / L2) for face matching.

    Collection schema (created by migrate_json_to_qdrant.py):
        name: "faces", dim: 512, distance: EUCLID
        payload: {name, avatar, source, type, image_id}

    One user can have multiple vectors (multiple images).
    Point IDs are deterministic uuid5(name_imageN_type) so re-running is idempotent.
    """

    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection: str = "faces"):
        self._collection = collection
        self._client = QdrantClient(host=host, port=port)
        self.avatars: dict[str, str] = {}
        self._total = 0
        self._init_collection()
        self._load_avatars()

    def _init_collection(self) -> None:
        """Ensure collection exists; create empty one if missing."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=512, distance=Distance.EUCLID),
            )
            print(f"[QdrantFaceDB] Created empty collection '{self._collection}'")
        self._total = self._client.count(self._collection).count
        print(f"[QdrantFaceDB] Connected — collection '{self._collection}' "
              f"has {self._total} point(s)")

    def _load_avatars(self) -> None:
        """Cache avatar payloads for use in events (name → avatar string)."""
        if self._total == 0:
            return
        offset = None
        while True:
            result, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in result:
                p = point.payload or {}
                name = p.get("name")
                avatar = p.get("avatar")
                if name and avatar and name not in self.avatars:
                    self.avatars[name] = avatar
            if next_offset is None:
                break
            offset = next_offset

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """
        Nearest-neighbour search in Qdrant (real-time, no restart needed).
        Returns (person_name, l2_distance). Returns (None, inf) if DB is empty.

        New users added to Qdrant are automatically found without restarting.
        """
        if self._total == 0:
            # Refresh count in case new users were added
            self._total = self._client.count(self._collection).count
            if self._total == 0:
                return None, float("inf")

        try:
            result = self._client.query_points(
                collection_name=self._collection,
                query=embedding.tolist(),
                limit=1,
                with_payload=True,
            )
            hits = result.points
        except Exception as e:
            print(f"[QdrantFaceDB] query_points error: {e}")
            return None, float("inf")

        if not hits:
            return None, float("inf")

        hit = hits[0]
        name = (hit.payload or {}).get("name")
        # Qdrant EUCLID score IS the L2 distance
        return name, float(hit.score)

    def refresh_count(self) -> None:
        """Refresh cached total count (called periodically if needed)."""
        self._total = self._client.count(self._collection).count

    def add_embeddings(self, name: str, embeddings: list[np.ndarray],
                       source: str = "auto") -> None:
        """Upsert new embeddings for a person (used by AutoSaveWorker).
        
        Each embedding gets a unique time-based ID, supporting multiple
        vectors per user from different images.
        """
        import uuid
        _NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        points = []
        for i, emb in enumerate(embeddings):
            vec = emb.flatten()
            norm = np.linalg.norm(vec)
            vec = (vec / norm).tolist() if norm > 0 else vec.tolist()
            pid = str(uuid.uuid5(_NS, f"{name}_auto_{i}_{time.time_ns()}"))
            points.append(PointStruct(
                id=pid,
                vector=vec,
                payload={"name": name, "source": source, "type": "auto", "image_id": i},
            ))
        if points:
            self._client.upsert(collection_name=self._collection, points=points)
            self._total = self._client.count(self._collection).count
            print(f"[QdrantFaceDB] Auto-saved {len(points)} embedding(s) for '{name}'. "
                  f"Total: {self._total}")

    def count(self) -> int:
        """Return total number of points in the collection."""
        return self._total

    def count_auto_saved(self, name: str) -> int:
        """Count auto-saved embeddings for a person (source='auto_save')."""
        try:
            result = self._client.count(
                collection_name=self._collection,
                count_filter=Filter(must=[
                    FieldCondition(key="name", match=MatchValue(value=name)),
                    FieldCondition(key="source", match=MatchValue(value="auto_save")),
                ])
            )
            return result.count
        except Exception as e:
            print(f"[QdrantFaceDB] count_auto_saved error: {e}")
            return 0

# Keep alias for backward compatibility
FaceDatabase = QdrantFaceDatabase


# =============================================================================
# Face Tracker
# =============================================================================

@dataclass
class TrackedFace:
    """
    Track a face and confirm identity via consecutive matches.
    Identity is confirmed when the same person matches min_streak times.
    """
    object_id: int
    l2_threshold: float = 1.0
    min_streak: int = 3
    skip_reid: int = 3
    reid_interval: int = 30
    auto_save_buffer_size: int = 10

    label: str | None = None
    score: float = 0.0

    _person: str | None = None  # person name (was int index, now string from Qdrant)
    _streak: int = 0
    _distances: list[float] = field(default_factory=list)

    last_sgie: int = 0
    age: int = 0

    # Auto-save: ring buffer of (embedding, l2_dist) for confirmed identity
    embedding_buffer: deque = field(default_factory=deque)

    def __post_init__(self):
        # Re-create deque with correct maxlen after dataclass init
        self.embedding_buffer = deque(maxlen=self.auto_save_buffer_size)

    def should_run_sgie(self, frame: int) -> bool:
        """Check if SGIE should run based on frame interval"""
        interval = self.reid_interval if self.label else self.skip_reid
        return (frame - self.last_sgie) >= interval

    def add_match(self, person: str | None, distance: float) -> bool:
        """Add match result. Returns True if identity confirmed."""
        if person is None or distance > self.l2_threshold:
            return False
        if person != self._person:
            self._person = person
            self._distances = [distance]
            self._streak = 1
            return False
        self._streak += 1
        self._distances.append(distance)
        return self._streak >= self.min_streak

    def confirm(self, name: str) -> bool:
        """Confirm identity. Returns True if first confirmation."""
        is_new = self.label is None
        if self.label is not None and self.label != name:
            # Identity changed — discard buffer from previous person
            self.embedding_buffer.clear()
        self.label = name
        self.score = sum(self._distances) / len(self._distances) if self._distances else 0.0
        self._person, self._streak, self._distances = None, 0, []
        if is_new:
            print(f"[CONFIRMED] id={self.object_id} -> {name} (score={self.score:.3f})")
        return is_new

    def buffer_embedding(self, emb: np.ndarray, dist: float) -> None:
        """Buffer a high-quality embedding for auto-save (only when confirmed)."""
        self.embedding_buffer.append((emb.copy(), dist))


class TrackerManager:
    """Manage tracked faces per camera (source_id namespace) with auto-cleanup."""

    def __init__(self, config: dict, max_age: int = 30, cleanup_interval: int = 10):
        self.config = config
        self.max_age = max_age
        self.cleanup_interval = cleanup_interval
        self._trackers: dict[int, dict[int, TrackedFace]] = {}
        self._last_cleanup = 0

    def get(self, source_id: int, oid: int) -> TrackedFace | None:
        return self._trackers.get(source_id, {}).get(oid)

    def get_or_create(self, source_id: int, oid: int, frame: int) -> TrackedFace:
        if source_id not in self._trackers:
            self._trackers[source_id] = {}
        cam_dict = self._trackers[source_id]
        if oid not in cam_dict:
            cam_dict[oid] = TrackedFace(
                object_id=oid,
                l2_threshold=self.config.get("l2_threshold", 1.0),
                min_streak=self.config.get("min_streak", 3),
                skip_reid=self.config.get("skip_reid", 3),
                reid_interval=self.config.get("reid_interval", 30),
                auto_save_buffer_size=self.config.get("auto_save_buffer_size", 10),
                last_sgie=frame,
            )
        return cam_dict[oid]

    def cleanup(self, current_frame: int = 0) -> list[tuple[int, int]]:
        """Increment age and remove stale trackers. Returns removed list."""
        removed = []
        for source_id, cam_dict in self._trackers.items():
            to_remove = []
            for oid, t in cam_dict.items():
                t.age += 1
                if t.age > self.max_age:
                    to_remove.append(oid)
            for oid in to_remove:
                del cam_dict[oid]
                removed.append((source_id, oid))
        self._last_cleanup = current_frame
        return removed

    def auto_cleanup(self, current_frame: int) -> list[tuple[int, int]]:
        """Auto-cleanup based on frame interval. Returns removed list."""
        if current_frame - self._last_cleanup >= self.cleanup_interval:
            return self.cleanup(current_frame)
        return []

    def stats(self) -> tuple[int, int, int]:
        """Returns (total, confirmed, pending)"""
        total = confirmed = 0
        for cam_dict in self._trackers.values():
            for t in cam_dict.values():
                total += 1
                confirmed += 1 if t.label else 0
        return total, confirmed, total - confirmed


class EventSet:
    """Track sent events with frame-based storage and auto-cleanup."""

    def __init__(self, max_age: int = 30):
        self.max_age = max_age
        self._events: dict[tuple[int, int], int] = {}

    def add(self, key: tuple[int, int], frame: int) -> bool:
        """Add event. Returns True if newly added."""
        if key not in self._events:
            self._events[key] = frame
            return True
        return False

    def contains(self, key: tuple[int, int]) -> bool:
        """Check if key exists."""
        return key in self._events

    def discard(self, key: tuple[int, int]) -> None:
        """Remove key."""
        self._events.pop(key, None)

    def cleanup(self, current_frame: int) -> list[tuple[int, int]]:
        """Remove stale entries based on frame age. Returns removed list."""
        removed = []
        to_remove = [k for k, f in self._events.items() if current_frame - f > self.max_age]
        for k in to_remove:
            del self._events[k]
            removed.append(k)
        return removed

    def auto_cleanup(self, current_frame: int) -> list[tuple[int, int]]:
        """Auto-cleanup based on frame interval."""
        if current_frame % self.max_age == 0:
            return self.cleanup(current_frame)
        return []


# =============================================================================
# Auto-Save Worker (cronjob: flush embedding buffers → Qdrant)
# =============================================================================

class AutoSaveWorker:
    """
    Periodically flush buffered embeddings from confirmed trackers to Qdrant.

    Strategy:
    - Only saves embeddings below auto_save_l2_threshold (high-quality only)
    - Sorts candidates by L2 distance (best quality first)
    - Caps total auto-saved embeddings per person in Qdrant (auto_save_max_total)
    - Clears the tracker buffer after each successful save
    """

    def __init__(self, db: "QdrantFaceDatabase", tracker_manager: TrackerManager,
                 params: dict):
        self._db = db
        self._trackers = tracker_manager
        self._l2_threshold = params.get("auto_save_l2_threshold", 0.5)
        self._max_per_run = params.get("auto_save_max_per_run", 3)
        self._max_total = params.get("auto_save_max_total", 20)

    def run(self, tick: int) -> None:
        """Called by IntervalRunner on each timer tick."""
        saved_total = 0
        for cam_dict in self._trackers._trackers.values():
            for trk in cam_dict.values():
                if trk.label is None or not trk.embedding_buffer:
                    continue

                # Filter to high-quality embeddings only
                candidates = [
                    (emb, dist) for emb, dist in trk.embedding_buffer
                    if dist <= self._l2_threshold
                ]
                if not candidates:
                    continue

                # Check how many auto-saved embeddings this person already has
                current_auto = self._db.count_auto_saved(trk.label)
                slots = self._max_total - current_auto
                if slots <= 0:
                    trk.embedding_buffer.clear()
                    continue

                # Sort ascending by L2 distance (best = smallest)
                candidates.sort(key=lambda x: x[1])
                to_save = candidates[:min(self._max_per_run, slots)]

                embeddings = [emb for emb, _ in to_save]
                distances = [d for _, d in to_save]
                self._db.add_embeddings(trk.label, embeddings, source="auto_save")
                saved_total += len(embeddings)
                trk.embedding_buffer.clear()
                print(
                    f"[AutoSaveWorker] '{trk.label}' saved {len(embeddings)} embedding(s) "
                    f"(dist: {[f'{d:.3f}' for d in distances]}) "
                    f"auto_total: {current_auto + len(embeddings)}/{self._max_total}"
                )

        if saved_total > 0:
            print(f"[AutoSaveWorker] tick={tick} — flushed {saved_total} embedding(s) total")


# =============================================================================
# Main Processor (includes probes and event handling)
# =============================================================================

@ProcessorRegistry.register("recognition")
class FaceRecognitionProcessor:
    """
    Face recognition processor - all-in-one implementation.

    Combines:
    - Database loading/matching
    - Tracker management
    - Probe callbacks
    - Event emission
    - OSD display
    """

    def __init__(self, config: Dict[str, Any], sink: BaseSink, source_mapper=None):
        """Initialize face recognition processor.

        Args:
            config: Branch configuration dict
            sink: BaseSink for sending events
            source_mapper: SourceIDMapper for camera_id <-> source_id mapping
        """
        self._config = config
        self._sink = sink
        self._source_mapper = source_mapper
        params = config.get("params", {})

        # Connect to Qdrant face database
        qdrant_host = params.get("qdrant_host", os.getenv("QDRANT_HOST", "localhost"))
        qdrant_port = int(params.get("qdrant_port", os.getenv("QDRANT_PORT", 6333)))
        qdrant_collection = params.get("qdrant_collection", "faces")
        print(f"[FaceRecognitionProcessor] Connecting to Qdrant "
              f"{qdrant_host}:{qdrant_port} collection='{qdrant_collection}'...")
        self._db = QdrantFaceDatabase(
            host=qdrant_host, port=qdrant_port, collection=qdrant_collection
        )
        print(f"[FaceRecognitionProcessor] Loaded {self._db.count()} face point(s)")

        # Initialize tracker manager and event set
        self._trackers = TrackerManager(params)
        self._sent_faces = EventSet(max_age=params.get("max_age", 30))

        # Cleanup runner
        cleanup_interval = params.get("cleanup_interval", 10) * 1000
        self._cleanup_runner = IntervalRunner(cleanup_interval, self._cleanup)

        # Auto-save runner (flush embedding buffers → Qdrant periodically)
        self._auto_save_runner: IntervalRunner | None = None
        if params.get("auto_save_enabled", True):
            auto_save_interval_ms = params.get("auto_save_interval", 30) * 1000
            self._auto_save_worker = AutoSaveWorker(self._db, self._trackers, params)
            self._auto_save_runner = IntervalRunner(auto_save_interval_ms,
                                                    self._auto_save_worker.run)
            print(f"[FaceRecognitionProcessor] Auto-save enabled "
                  f"(interval={params.get('auto_save_interval', 30)}s, "
                  f"l2_threshold={params.get('auto_save_l2_threshold', 0.5)}, "
                  f"max_total={params.get('auto_save_max_total', 20)})")
        else:
            print("[FaceRecognitionProcessor] Auto-save disabled")

        print("[FaceRecognitionProcessor] Initialized")

    @property
    def name(self) -> str:
        return "recognition"
    
    def _get_stats(self) -> dict:
        """Return stats dict for """
        if not self._trackers:
            return {"total": 0, "confirmed": 0, "pending": 0}
        total, confirmed, pending = self._trackers.stats()
        return {"total": total, "confirmed": confirmed, "pending": pending}

    def get_probes(self) -> Dict[str, Callable]:
        """Return probe callbacks"""
        params = self._config.get("params", {})
        return {
            "tracker_probe": self._tracker_probe,
            "sgie_probe": self._sgie_probe,
            "recognition_fps_probe": fps_probe_factory(
                name="Recognition",
                log_interval=params.get("log_interval", 1.0),
                stats_interval=params.get("stats_interval", 10.0),
                stats_callback=self._get_stats,
            ),
        }

    # -------------------------------------------------------------------------
    # Probe Callbacks
    # -------------------------------------------------------------------------

    def _tracker_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Decide whether to skip SGIE for each face"""
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK

        min_face = self._config.get("params", {}).get("min_face_size", 50)

        for frame, obj in BatchIterator(batch):
            # Skip small faces
            if should_skip_face(obj, min_face):
                mark_skip_sgie(obj)
                continue
            # Skip if tracker says not ready for SGIE
            trk = self._trackers.get(frame.source_id, obj.object_id)
            if trk and not trk.should_run_sgie(frame.frame_num):
                mark_skip_sgie(obj)

        return Gst.PadProbeReturn.OK

    def _sgie_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Process recognition results and update display"""
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK

        for frame, obj in BatchIterator(batch):
            name, state, score = self._process_face(frame.source_id, obj, frame.frame_num)
            update_display(obj, name, score, state)

        return Gst.PadProbeReturn.OK

    def _cleanup(self, current_frame: int) -> None:
        """Cleanup stale trackers and events"""
        self._trackers.auto_cleanup(current_frame)
        self._sent_faces.auto_cleanup(current_frame)

    def _process_face(self, source_id: int, obj_meta, frame: int) -> tuple[str, str, float]:
        """Process face and return (name, state, score) for display"""
        oid = obj_meta.object_id
        trk = self._trackers.get_or_create(source_id, oid, frame)
        trk.age = 0

        emb = extract_embedding(obj_meta)
        if emb is not None:
            trk.last_sgie = frame
            name, dist = self._db.match(emb)  # Qdrant returns (name, l2_distance)
            # Buffer embedding while already confirmed (captures diverse angles/lighting)
            if trk.label is not None:
                trk.buffer_embedding(emb, dist)
            if trk.add_match(name, dist):
                if trk.confirm(name):
                    self._send_event(source_id, oid, name, frame)

        return (trk.label, "confirmed", trk.score) if trk.label else ("", "unknown", 0.0)

    def _send_event(self, source_id: int, object_id: int, name: str, frame: int) -> None:
        """Send face detection event (once per face)"""
        key = (source_id, object_id)
        if not self._sent_faces.add(key, frame):
            return

        camera_id = self._source_mapper.get_camera_id(source_id) if self._source_mapper else None
        event = {
            "type": "face_detected",
            "camera_id": camera_id,
            "source_id": source_id,
            "name": name,
            "timestamp": time.strftime("%H:%M:%S"),
            "object_id": object_id,
            "avatar": self._db.avatars.get(name),
        }
        self._sink.send_event(event)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        print(f"[FaceRecognitionProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        """Start cleanup timer when pipeline starts"""
        if self._cleanup_runner:
            self._cleanup_runner.start()
        if self._auto_save_runner:
            self._auto_save_runner.start()
        print("[FaceRecognitionProcessor] Started")

    def on_stop(self) -> None:
        """Stop cleanup timer when pipeline stops"""
        if self._auto_save_runner:
            self._auto_save_runner.stop()
        if self._cleanup_runner:
            self._cleanup_runner.stop()
        print("[FaceRecognitionProcessor] Stopped")

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Return processor statistics"""
        if not self._trackers:
            return None
        total, confirmed, pending = self._trackers.stats()
        # Count total buffered embeddings across all trackers
        buffered = sum(
            len(t.embedding_buffer)
            for cam_dict in self._trackers._trackers.values()
            for t in cam_dict.values()
        )
        return {
            "faces_in_database": self._db.count() if self._db else 0,
            "trackers_total": total,
            "trackers_confirmed": confirmed,
            "trackers_pending": pending,
            "auto_save_buffered": buffered,
        }

    @property
    def database(self) -> Optional[QdrantFaceDatabase]:
        return self._db