"""
Face Recognition Processor - Simplified Module

This module contains face recognition components with minimal class usage:
- FaceDatabase: Face feature storage and matching
- TrackedFace/TrackerManager: Identity tracking (dataclass + manager)
- FaceRecognitionProcessor: Main processor with probes

Auto-registered with ProcessorRegistry using @register decorator.
"""

import json
import os
import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional

import cv2
import numpy as np
import pyds
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import BatchIterator, extract_embedding, get_batch_meta, fps_probe_factory, IntervalRunner

from apps.face.redis_sync_service import FaceRecognitionRedisSync
from apps.common.redis_publisher import RedisEventPublisher, EVENT_TYPE_FACE
from apps.common.redis_connection import RedisConnection

import logging

logger = logging.getLogger(__name__)


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

# Event constants
DEFAULT_CROP_DIR = "data/face/crop-face"

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
# Frame Extraction Helper
# =============================================================================

def extract_frame(gst_buffer, frame_meta) -> Optional[np.ndarray]:
    """Extract OpenCV frame from GStreamer buffer."""
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_copy = np.array(n_frame, copy=True, order='C')
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        return frame_copy
    except Exception as e:
        logger.error(f"[Face] Error extracting frame: {e}")
        return None


def crop_face(frame: np.ndarray, obj_meta, padding: float = 0.1) -> Optional[np.ndarray]:
    """Crop face region from frame with optional padding."""
    try:
        rect = obj_meta.rect_params
        h, w = frame.shape[:2]
        
        # Calculate padded bbox
        pad_w = int(rect.width * padding)
        pad_h = int(rect.height * padding)
        
        x1 = max(0, int(rect.left) - pad_w)
        y1 = max(0, int(rect.top) - pad_h)
        x2 = min(w, int(rect.left + rect.width) + pad_w)
        y2 = min(h, int(rect.top + rect.height) + pad_h)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return frame[y1:y2, x1:x2].copy()
    except Exception as e:
        logger.error(f"[Face] Error cropping face: {e}")
        return None


def save_face_image(face_img: np.ndarray, crop_dir: str, camera_id: str, user_id: str) -> Optional[str]:
    """Save cropped face image to disk and return file path."""
    try:
        os.makedirs(crop_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{camera_id}_{user_id}_{timestamp}.jpg"
        filepath = os.path.join(crop_dir, filename)
        
        cv2.imwrite(filepath, face_img)
        return os.path.abspath(filepath)
    except Exception as e:
        logger.error(f"[Face] Error saving face image: {e}")
        return None


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


# # =============================================================================
# # Face Database
# # =============================================================================

class FaceDatabase:
    """
    Manages registered face features for matching using Redis.
    
    Schema: face:{user_id}:{image_name}
    Fields: name (user_id), face_image_url, embedding (512-dim)
    """

    def __init__(self, config: dict):
        self.names: list[str] = []              # Unique user_ids
        self.face_images: dict[str, list[str]] = {}  # user_id -> [image_urls]
        self.user_names: dict[str, str] = {}    # user_id -> display_name
        
        # Redis configuration
        self.redis_client = None
        self._index_name = "face_idx"
        self._doc_prefix = "face:"
        self._vector_field = "embedding"
        self._name_to_idx: dict[str, int] = {}
        
        self._load_redis(config)

    def _load_redis(self, config: dict) -> None:
        """Initialize Redis connection and verify index"""
        try:
            import redis
            from redis.commands.search.query import Query
        except ImportError:
            raise RuntimeError("redis package not found. Install with: pip install redis>=5.0.0")
        
        redis_config = {
            "host": config.get("redis_host", os.getenv("REDIS_HOST", "localhost")),
            "port": int(config.get("redis_port", os.getenv("REDIS_PORT", "6379"))),
            "db": int(config.get("redis_db", os.getenv("REDIS_DB", "0"))),
            "password": config.get("redis_password", os.getenv("REDIS_PASSWORD")),
            "decode_responses": False,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
        }
        
        start = time.time()
        self.redis_client = redis.Redis(**redis_config)
        self.redis_client.ping()
        
        try:
            info = self.redis_client.ft(self._index_name).info()
            num_docs = info.get("num_docs", 0)

            if num_docs == 0:
                raise RuntimeError(f"Redis index '{self._index_name}' is empty")

            self._load_metadata_from_redis()
            
            elapsed = (time.time() - start) * 1000
            print(f"Connected to Redis at {redis_config['host']}:{redis_config['port']} in {elapsed:.1f}ms")

        except redis.exceptions.ResponseError as e:
            raise RuntimeError(f"Redis index '{self._index_name}' not found: {e}")

    def _load_metadata_from_redis(self) -> None:
        """Load user_ids, display names and face images from Redis (multi-image support)"""
        cursor = 0
        user_images = {}  # user_id -> [image_urls]
        user_names = {}   # user_id -> display_name
        
        while True:
            cursor, keys = self.redis_client.scan(cursor, match=f"{self._doc_prefix}*", count=100)
            
            for key in keys:
                try:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    
                    # Skip RediSearch internal keys
                    if ':idx' in key_str or key_str.endswith('_idx'):
                        continue
                    
                    # Skip event streams
                    if key_str.startswith('event:'):
                        continue
                    
                    # Check if key is a hash before calling hgetall
                    key_type = self.redis_client.type(key)
                    if key_type != b"hash":
                        continue
                    
                    parts = key_str.replace(self._doc_prefix, "").split(":", 1)
                
                    if len(parts) >= 1:
                        user_id = parts[0]
                        
                        data = self.redis_client.hgetall(key)
                        
                        # Check if face_image_url field exists
                        if b"face_image_url" not in data:
                            print(f"[WARNING] Key {key_str} missing 'face_image_url' field!")
                            continue
                        
                        face_image_url = data.get(b"face_image_url", b"").decode('utf-8')
                        
                        # Get display name (fallback to user_id if not present)
                        display_name = data.get(b"name", b"").decode('utf-8') or user_id
                        
                        if user_id not in user_images:
                            user_images[user_id] = []
                            user_names[user_id] = display_name
                        
                        if face_image_url:
                            user_images[user_id].append(face_image_url)
                            
                except Exception as e:
                    print(f"[ERROR] Failed to process key {key}: {e}")
                    continue
            
            if cursor == 0:
                break
        
        # Build index
        for idx, user_id in enumerate(sorted(user_images.keys())):
            self.names.append(user_id)
            self._name_to_idx[user_id] = idx
            self.face_images[user_id] = user_images[user_id]
            self.user_names[user_id] = user_names.get(user_id, user_id)
        
        print(f"Loaded {len(self.names)} users with {sum(len(imgs) for imgs in user_images.values())} total face images")

    def match(self, embedding: np.ndarray) -> tuple[int, float]:
        """Match embedding against database using Redis vector search"""
        try:
            from redis.commands.search.query import Query
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            query_vector = embedding.astype(np.float32).tobytes()
            
            # KNN=1 to find best match across all images
            query = (
                Query(f"*=>[KNN 1 @{self._vector_field} $vec AS score]")
                .return_fields("name", "face_image_url", "score")
                .sort_by("score")
                .dialect(2)
            )
            result = self.redis_client.ft(self._index_name).search(
                query,
                query_params={"vec": query_vector}
            )
            if result.total > 0:
                doc = result.docs[0]
                redis_key = doc.id
                
                if isinstance(redis_key, bytes):
                    redis_key = redis_key.decode('utf-8')
                    
                parts = redis_key.replace(self._doc_prefix, "").split(":", 1)
                user_id = parts[0] if parts else ""
                # image_name = parts[1] if len(parts) > 1 else ""
                # print("========> IMAGE NAME ", image_name) # Check multiple face for one user_id
                
                distance = float(doc.score)
                person_idx = self._name_to_idx.get(user_id, -1)
                return person_idx, distance, user_id
            else:
                return -1, float("inf")
                
        except Exception as e:
            logger.error(f"Redis search error: {e}")
            return -1, float("inf")
    
    def get_primary_image(self, user_id: str) -> str:
        """Get primary face image URL for user (first image)"""
        images = self.face_images.get(user_id, [])
        return images[0] if images else ""

    def get_display_name(self, user_id: str) -> str:
        """Get display name for user"""
        return self.user_names.get(user_id, user_id)

    def close(self) -> None:
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None


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

    label: str | None = None
    user_id: str | None = None  # Added for storing user_id
    score: float = 0.0

    _person: int = -1
    _streak: int = 0
    _distances: list[float] = field(default_factory=list)

    last_sgie: int = 0
    age: int = 0

    def should_run_sgie(self, frame: int) -> bool:
        """Check if SGIE should run based on frame interval"""
        interval = self.reid_interval if self.label else self.skip_reid
        return (frame - self.last_sgie) >= interval

    def add_match(self, person: int, distance: float) -> bool:
        """Add match result. Returns True if identity confirmed."""
        if distance > self.l2_threshold:
            return False
        if person != self._person:
            self._person = person
            self._distances = [distance]
            self._streak = 1
            return False
        self._streak += 1
        self._distances.append(distance)
        return self._streak >= self.min_streak

    def confirm(self, name: str, user_id: str) -> bool:
        """Confirm identity. Returns True if first confirmation."""
        is_new = self.label is None
        self.label = name
        self.user_id = user_id
        self.score = sum(self._distances) / len(self._distances) if self._distances else 0.0
        self._person, self._streak, self._distances = -1, 0, []
        if is_new:
            print(f"[CONFIRMED] id={self.object_id} -> {name} (user_id={user_id}, score={self.score:.3f})")
        return is_new


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

        # Load face database
        print(f"[FaceRecognitionProcessor] Initializing face database...")
        self._db = FaceDatabase(params)
        print(f"[FaceRecognitionProcessor] Loaded {len(self._db.names)} faces")

        # Initialize tracker manager and event set
        self._trackers = TrackerManager(params)
        self._sent_faces = EventSet(max_age=params.get("max_age", 30))

        # Cleanup runner
        cleanup_interval = params.get("cleanup_interval", 10) * 1000
        self._cleanup_runner = IntervalRunner(cleanup_interval, self._cleanup)

        # Crop directory
        self._crop_dir: str = DEFAULT_CROP_DIR
        
        # Create shared Redis connection
        self._redis_conn = RedisConnection.get_instance(params, name="face")
        
        # Use shared client for publisher
        self._redis_publisher = RedisEventPublisher(
            params, 
            redis_client=self._redis_conn.client
        )
        
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
        """Process recognition results, crop faces, and update display"""
        gst_buffer = info.get_buffer()
        batch = get_batch_meta(gst_buffer)
        if not batch:
            return Gst.PadProbeReturn.OK

        # Cache for extracted frames per batch_id
        frame_cache: Dict[int, np.ndarray] = {}

        for frame, obj in BatchIterator(batch):
            name, state, score, event_data = self._process_face(frame.source_id, obj, frame.frame_num)
            update_display(obj, name, score, state)
            
            # If we have a pending event, extract frame and crop face
            if event_data is not None:
                # Extract frame if not cached
                if frame.batch_id not in frame_cache:
                    extracted = extract_frame(gst_buffer, frame)
                    if extracted is not None:
                        frame_cache[frame.batch_id] = extracted
                
                # Crop and save face
                if frame.batch_id in frame_cache:
                    face_img = crop_face(frame_cache[frame.batch_id], obj)
                    if face_img is not None:
                        camera_id = event_data.get("cam_id", str(frame.source_id))
                        user_id = event_data.get("user_id", "unknown")
                        
                        filepath = save_face_image(face_img, self._crop_dir, camera_id, user_id)
                        if filepath:
                            self._publish_event(event_data, filepath)

        return Gst.PadProbeReturn.OK

    def _cleanup(self, current_frame: int) -> None:
        """Cleanup stale trackers and events"""
        self._trackers.auto_cleanup(current_frame)
        self._sent_faces.auto_cleanup(current_frame)

    def _process_face(self, source_id: int, obj_meta, frame: int) -> tuple[str, str, float, Optional[dict]]:
        """Process face and return (name, state, score, event_data)"""
        oid = obj_meta.object_id
        trk = self._trackers.get_or_create(source_id, oid, frame)
        trk.age = 0
        event_data = None

        emb = extract_embedding(obj_meta)
        if emb is not None:
            trk.last_sgie = frame
            person, dist, user_id = self._db.match(emb)
            if trk.add_match(person, dist):
                # user_id comes from Redis key, display_name from 'name' field
                display_name = self._db.get_display_name(user_id)
                if trk.confirm(display_name, user_id):
                    event_data = self._prepare_event(source_id, oid, user_id, display_name, frame)

        return (trk.label, "confirmed", trk.score, event_data) if trk.label else ("", "unknown", 0.0, None)

    def _prepare_event(self, source_id: int, object_id: int, user_id: str, display_name: str, frame: int) -> Optional[dict]:
        """Prepare event data (without image path yet)"""
        key = (source_id, object_id)
        if not self._sent_faces.add(key, frame):
            return None

        camera_id = self._source_mapper.get_camera_id(source_id) if self._source_mapper else str(source_id)
        
        # Return event components for use with shared publisher
        return {
            "cam_id": camera_id,
            "user_id": user_id,
            "display_name": display_name,
        }

    def _publish_event(self, event_data: dict, image_path: str) -> None:
        """Publish event using own Redis publisher"""
        if not self._redis_publisher:
            logger.warning("[FaceRecognitionProcessor] Redis publisher not available")
            return
        
        cam_id = event_data.get("cam_id", "unknown")
        user_id = event_data.get("user_id", "unknown")
        display_name = event_data.get("display_name", "unknown")
        
        message_id = self._redis_publisher.publish(
            cam_id=cam_id,
            event_type=EVENT_TYPE_FACE,
            data={
                "accessUser": display_name,
                "accessId": user_id,
            },
            images=[image_path] if image_path else [],
        )
        
        if message_id:
            print(f"[Event] Published: {display_name} on camera {cam_id} (id={message_id})")
        else:
            print(f"[Event] Failed to publish: {display_name}")
        
        # Also send to sink if available
        if self._sink:
            self._sink.send_event({
                "camId": cam_id,
                "aiType": EVENT_TYPE_FACE,
                "data": {"accessUser": display_name, "accessId": user_id},
                "images": [image_path] if image_path else [],
            })

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any, camera_manager=None) -> None:
        """Called when pipeline is built
        
        Args:
            pipeline: GStreamer pipeline
            branch_info: BranchInfo for this branch
            camera_manager: Optional MultibranchCameraManager instance (for dynamic camera ops)
        """
        print(f"[FaceRecognitionProcessor] Pipeline built, branch: {branch_info.name}")
        self._camera_manager = camera_manager

        # Start Redis sync if enabled
        params = self._config.get("params", {})
        redis_sync_cfg = params.get("redis_sync", {}) or {}
        if not redis_sync_cfg.get("enabled", False):
            logger.info("[FaceRecognitionProcessor] Redis sync disabled (params.redis_sync.enabled=false)")
            return

        if not self._camera_manager:
            logger.error("[FaceRecognitionProcessor] Cannot start Redis sync: camera_manager not available")
            return

        # Use shared redis connection params
        host = params.get("redis_host", "localhost")
        port = int(params.get("redis_port", 6380))
        key = redis_sync_cfg.get("key", "ND_KhuonMat:0")
        poll_interval = float(redis_sync_cfg.get("poll_interval", 5.0))

        def _on_add(camera_id: str, rtsp_url: str) -> None:
            success = self._camera_manager.add_camera(camera_id, rtsp_url, "recognition")
            if success:
                logger.info("[FaceRecognitionProcessor] Added camera %s", camera_id)
            else:
                logger.error("[FaceRecognitionProcessor] Failed to add camera %s", camera_id)

        def _on_remove(camera_id: str) -> None:
            self._camera_manager.remove_camera(camera_id)
            logger.info("[FaceRecognitionProcessor] Removed camera %s", camera_id)

        def _on_restart(camera_id: str, new_rtsp_url: str) -> None:
            self._camera_manager.remove_camera(camera_id)
            time.sleep(0.2)
            success = self._camera_manager.add_camera(camera_id, new_rtsp_url, "recognition")
            if success:
                logger.info("[FaceRecognitionProcessor] Restarted camera %s", camera_id)
            else:
                logger.error("[FaceRecognitionProcessor] Failed to restart camera %s", camera_id)

        self._redis_sync = FaceRecognitionRedisSync(
            redis_host=host,
            redis_port=port,
            redis_key=key,
            poll_interval=poll_interval,
            on_add=_on_add,
            on_remove=_on_remove,
            on_restart=_on_restart,
            redis_client=self._redis_conn.client,
        )
        self._redis_sync.start()

    def on_start(self) -> None:
        """Start cleanup timer when pipeline starts"""
        if self._cleanup_runner:
            self._cleanup_runner.start()
        print("[FaceRecognitionProcessor] Started")

    def on_stop(self) -> None:
        """Stop cleanup timer when pipeline stops"""
        # Stop Redis sync
        if self._redis_sync:
            try:
                self._redis_sync.stop()
            except Exception:
                pass
            self._redis_sync = None

        if self._cleanup_runner:
            self._cleanup_runner.stop()
        print("[FaceRecognitionProcessor] Stopped")

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Return processor statistics"""
        if not self._trackers:
            return None
        total, confirmed, pending = self._trackers.stats()
        return {
            "faces_in_database": len(self._db.names) if self._db else 0,
            "trackers_total": total,
            "trackers_confirmed": confirmed,
            "trackers_pending": pending,
        }

    @property
    def database(self) -> Optional[FaceDatabase]:
        return self._db