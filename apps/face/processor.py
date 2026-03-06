"""
Face Recognition Processor - Hybrid Memory + Qdrant Module

This module contains face recognition components:
- FaceDatabase: Load features from PostgreSQL (in-memory cosine matching)
- Qdrant fallback: Search unknown faces in Qdrant vector DB
- Frontal face check: Landmark-based face angle verification
- TrackedFace/TrackerManager: Identity tracking with voting
- FaceRecognitionProcessor: Main processor with probes

Auto-registered with ProcessorRegistry using @register decorator.
"""

import json
import math
import os
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List, Tuple

import numpy as np
import cv2
import pyds
from datetime import datetime
from queue import Queue, Full

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import BatchIterator, extract_embedding, get_batch_meta, fps_probe_factory, IntervalRunner
from apps.face.http_event_sender import HttpEventSender

# Database imports
from apps.face.pg_database import SessionLocal
from apps.face.models_sql import AccessUser, AccessEvent
from sqlalchemy.orm import joinedload
from sqlalchemy import func
HAS_DB = True

# Qdrant imports
from apps.face.search import search as qdrant_search, upsert as qdrant_upsert, is_available as qdrant_is_available
HAS_QDRANT = True



# =============================================================================
# Constants
# =============================================================================

# OSD Colors (RGBA)
COLOR_CONFIRMED = (0.0, 1.0, 0.0, 1.0)  # Green
COLOR_UNKNOWN = (1.0, 0.5, 0.0, 1.0)    # Orange
COLOR_TEXT = (1.0, 1.0, 1.0, 1.0)       # White
COLOR_TEXT_BG = (0.0, 0.0, 0.0, 0.7)    # Black transparent

CV_COLOR_KNOWN = (0, 255, 0)      # BGR
CV_COLOR_UNKNOWN = (0, 128, 255)  # BGR
CV_COLOR_TEXT = (255, 255, 255)

# Display settings
BORDER_WIDTH = 3
FONT_SIZE = 14
FONT_NAME = "Serif"

# Face-specific constants
SKIP_SGIE_COMPONENT_ID = 100

# Image save directories
CROP_FACE_DIR = "data/face/crop-face"
FULL_FRAME_DIR = "data/face/full-frame"


# =============================================================================
# Frame Extraction Helpers
# =============================================================================

def save_image(img: np.ndarray, directory: str, prefix: str) -> Optional[str]:
    """Save image to directory. Returns filepath or None."""
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{ts}.jpg"
    filepath = os.path.join(directory, filename)
    ok = cv2.imwrite(filepath, img)
    if not ok:
        print(f"[SaveImage] FAILED to write {filepath}")
        return None
    return filepath


def _is_unknown_identity(name: Optional[str], person_id: Optional[str]) -> bool:
    n = (name or "").strip().lower()
    if n in {"", "unknown", "unknow", "none", "null", "Unknown"}:
        return True
    if person_id and name and person_id.strip() == name.strip():
        return True
    return False


def draw_event_bbox(frame: np.ndarray, obj_meta, name: str, person_id: Optional[str]) -> None:
    h, w = frame.shape[:2]

    if isinstance(obj_meta, dict):
        left = float(obj_meta.get("left", 0.0))
        top = float(obj_meta.get("top", 0.0))
        width = float(obj_meta.get("width", 0.0))
        height = float(obj_meta.get("height", 0.0))
    else:
        rect = getattr(obj_meta, "rect_params", None)
        if rect is None:
            return
        left = float(rect.left)
        top = float(rect.top)
        width = float(rect.width)
        height = float(rect.height)

    x1 = max(0, int(left))
    y1 = max(0, int(top))
    x2 = min(w - 1, int(left + width))
    y2 = min(h - 1, int(top + height))
    if x2 <= x1 or y2 <= y1:
        return

    unknown = _is_unknown_identity(name, person_id)
    color = CV_COLOR_UNKNOWN if unknown else CV_COLOR_KNOWN
    label = "Unknown" if unknown else (name or "Unknown")
    text = f"{label} [{x2 - x1}x{y2 - y1}]"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)

    tx = x1
    ty = y1 - 8
    if ty - th - base < 0:
        ty = min(h - base - 2, y1 + th + base + 8)

    cv2.rectangle(
        frame,
        (max(0, tx - 2), max(0, ty - th - base - 2)),
        (min(w - 1, tx + tw + 2), min(h - 1, ty + base + 2)),
        color,
        -1,
    )
    cv2.putText(frame, text, (tx, ty), font, scale, CV_COLOR_TEXT, thickness, cv2.LINE_AA)

# =============================================================================
# Face-Specific Helper Functions
# =============================================================================

def should_skip_face(obj_meta, min_size: int = 50) -> bool:
    """Check if face should be skipped based on size"""
    rect = obj_meta.rect_params
    return rect.width < min_size or rect.height < min_size


def mark_skip_sgie(obj_meta) -> None:
    """Mark object to skip SGIE processing"""
    obj_meta.unique_component_id = SKIP_SGIE_COMPONENT_ID


def is_frontal_face(landmarks, angle_threshold=15, symmetry_threshold=0.3) -> bool:
    """
    Check if face is frontal based on 5-point landmarks.

    Args:
        landmarks: List of 5 points [(x1,y1), ..., (x5,y5)]
                   Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        angle_threshold: Roll angle threshold (degrees)
        symmetry_threshold: Symmetry threshold (0-1)

    Returns:
        bool: True if frontal face
    """
    min_eye_distance = 5

    if len(landmarks) != 5:
        return False

    points = np.array(landmarks, dtype=np.float32)
    left_eye, right_eye, nose, left_mouth, right_mouth = points

    # 0. Basic eye distance check
    eye_distance = np.linalg.norm(right_eye - left_eye)
    if eye_distance < min_eye_distance:
        return False

    # 1. Roll check (head tilt left/right)
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    eye_angle = math.degrees(math.atan2(eye_dy, eye_dx))
    if abs(eye_angle) > angle_threshold:
        return False

    # 2. Yaw check (nose deviation from eye center)
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_deviation = abs(nose[0] - eye_center_x)
    nose_symmetry_ratio = nose_deviation / eye_distance
    if nose_symmetry_ratio > symmetry_threshold:
        return False

    # 3. Yaw check (mouth deviation)
    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
    mouth_deviation = abs(mouth_center_x - eye_center_x)
    mouth_symmetry_ratio = mouth_deviation / eye_distance
    if mouth_symmetry_ratio > symmetry_threshold:
        return False

    # 4. Pitch check (eye-mouth symmetry)
    mouth_center = (left_mouth + right_mouth) / 2
    d1 = np.linalg.norm(left_eye - mouth_center)
    d2 = np.linalg.norm(right_eye - mouth_center)
    if d1 == 0 or d2 == 0:
        return False
    pitch_symmetry = min(d1, d2) / max(d1, d2)
    if pitch_symmetry < (1 - symmetry_threshold):
        return False

    # 5. Eye vertical difference (pitch indicator)
    eye_vertical_diff = abs(left_eye[1] - right_eye[1])
    if eye_vertical_diff > 0.5 * eye_distance:
        return False

    return True


def extract_landmarks(obj_meta, frame_shape=None) -> Optional[list]:
    """
    Extract 5-point landmarks from object metadata mask_params.

    Args:
        obj_meta: NvDsObjectMeta
        frame_shape: (h, w, c) of the frame for scale computation

    Returns:
        List of 5 (x, y) tuples or None
    """
    try:
        import pyds
        mp = obj_meta.mask_params
        if hasattr(mp, "get_mask_array"):
            arr = mp.get_mask_array()
            if len(arr) == 14 and frame_shape is not None:
                h, w = frame_shape[0], frame_shape[1]

                def det_scale(in_h, in_w, out_h=640, out_w=640):
                    r1 = in_h / in_w
                    r2 = out_h / out_w
                    if r1 > r2:
                        return out_h / in_h
                    else:
                        return out_w / in_w

                sc = det_scale(h, w, mp.height, mp.width)
                landmarks = [
                    (arr[1] / sc, arr[0] / sc),
                    (arr[3] / sc, arr[2] / sc),
                    (arr[5] / sc, arr[4] / sc),
                    (arr[7] / sc, arr[6] / sc),
                    (arr[9] / sc, arr[8] / sc),
                ]
                return landmarks
    except Exception:
        pass
    return None


# =============================================================================
# Display Functions
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
    if state == "confirmed":
        display_text = f"{name} ({score:.2f}) [{face_w}x{face_h}]"
    else:
        display_text = f"[{face_w}x{face_h}]"

    text = obj_meta.text_params
    text.display_text = display_text
    text.x_offset = int(rect.left)
    text.y_offset = max(0, int(rect.top) - 25)
    text.font_params.font_name = FONT_NAME
    text.font_params.font_size = FONT_SIZE

    r, g, b, a = COLOR_TEXT
    text.font_params.font_color.red, text.font_params.font_color.green = r, g
    text.font_params.font_color.blue, text.font_params.font_color.alpha = b, a

    text.set_bg_clr = 1
    r, g, b, a = COLOR_TEXT_BG
    text.text_bg_clr.red, text.text_bg_clr.green = r, g
    text.text_bg_clr.blue, text.text_bg_clr.alpha = b, a


# =============================================================================
# Face Database — Hybrid (PostgreSQL in-memory + JSON fallback)
# =============================================================================

class FaceDatabase:
    """
    Manages registered face features for matching.
    
    Primary: Load from PostgreSQL AccessUser + user_features (in-memory cosine similarity)
    """

    def __init__(self, db_url: str = None, use_db: bool = True):
        # Per-person data
        self.names: list[str] = []
        self.person_ids: list[str] = []

        # Flat matrix for fast cosine matching
        self.features_matrix: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self.features_owner_idx: list[int] = []  # index into names/person_ids

        # Lookup caches
        self.person_id_to_name: dict[str, str] = {}
        self.avatars: dict[str, str] = {}

        self._lock = threading.Lock()

        if use_db and HAS_DB:
            self._load_from_db()

    # ---- PostgreSQL loader ----

    def _load_from_db(self) -> None:
        """Load face features from PostgreSQL (AccessUser + user_features)"""
        start = time.time()
        try:
            with SessionLocal() as session:
                users = (
                    session.query(AccessUser)
                    .options(joinedload(AccessUser.features))
                    .all()
                )

                names: list[str] = []
                person_ids: list[str] = []
                all_feats: list[np.ndarray] = []
                owner_idx: list[int] = []
                pid_to_name: dict[str, str] = {}

                for u in users:
                    name = getattr(u, "name", "Unknown")

                    # Collect user features
                    user_feats: list[np.ndarray] = []
                    if u.features:
                        for f in u.features:
                            try:
                                vec = f.feature
                                if isinstance(vec, str):
                                    vec = json.loads(vec)
                                arr = np.array(vec, dtype=np.float32)
                                if arr.shape == (512,):
                                    n = np.linalg.norm(arr)
                                    if n > 0:
                                        arr = arr / n
                                    user_feats.append(arr)
                            except Exception:
                                pass

                    feats_np = (
                        np.vstack(user_feats).astype(np.float32)
                        if user_feats
                        else np.empty((0, 512), dtype=np.float32)
                    )

                    # --- Auto-link person_id if missing (from logic_infer.py) ---
                    if not u.person_id:
                        pid = self._auto_link_person_id(u, feats_np, session)
                    else:
                        pid = u.person_id

                    if not user_feats:
                        continue

                    # Deduplicate by person_id
                    if pid in pid_to_name:
                        # Already seen this pid, just append features
                        for feat in user_feats:
                            all_feats.append(feat)
                            owner_idx.append(person_ids.index(pid))
                        continue

                    idx = len(names)
                    names.append(name)
                    person_ids.append(pid)
                    pid_to_name[pid] = name

                    for feat in user_feats:
                        all_feats.append(feat)
                        owner_idx.append(idx)

                with self._lock:
                    self.names = names
                    self.person_ids = person_ids
                    self.person_id_to_name = pid_to_name
                    if all_feats:
                        self.features_matrix = np.vstack(all_feats).astype(np.float32)
                    else:
                        self.features_matrix = np.empty((0, 512), dtype=np.float32)
                    self.features_owner_idx = owner_idx

            elapsed = (time.time() - start) * 1000
            print(f"[FaceDB] Loaded {len(self.names)} users, "
                  f"{self.features_matrix.shape[0]} vectors from PostgreSQL in {elapsed:.1f}ms")
        except Exception as e:
            print(f"[FaceDB] Error loading from DB: {e}")

    def _auto_link_person_id(self, user, feats_np: np.ndarray, session) -> str:
        """
        Auto-link person_id for user missing it.
        1. Search Qdrant with user's features to find existing person_id
        2. If found, use the best match
        3. If not found, generate new UUID and seed Qdrant
        4. Commit to PostgreSQL
        
        Mirrors logic from logic_infer.py FaceRecognizer.refresh_users_id_and_features()
        """
        pid = None

        # Step 1: Try to find existing person_id from Qdrant
        if HAS_QDRANT and qdrant_is_available() and feats_np.shape[0] > 0:
            try:
                candidate_ids = []
                scores = {}
                # Sample a few vectors (don't search all if too many)
                indices = range(0, feats_np.shape[0], max(1, feats_np.shape[0] // 5))
                for i in indices:
                    vec = feats_np[i]
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / n
                    hits = qdrant_search(
                        query_vector=vec.tolist(), limit=10, similarity_threshold=0.35
                    )
                    if hits:
                        for hit in hits:
                            payload = getattr(hit, "payload", None)
                            if isinstance(payload, list) and len(payload) > 0:
                                payload = payload[0]
                            if not isinstance(payload, dict):
                                continue
                            p = payload.get("person_id")
                            score = float(getattr(hit, "score", 0))
                            if p:
                                candidate_ids.append(p)
                                scores[p] = max(scores.get(p, 0.0), score)

                candidate_ids = list(set(candidate_ids))
                if candidate_ids:
                    # Pick best using DB frequency + score weighting
                    try:
                        normalized = [p.strip().lower() for p in candidate_ids if p]
                        counts = (
                            session.query(AccessEvent.person_id, func.count(AccessEvent.id))
                            .filter(func.lower(func.trim(AccessEvent.person_id)).in_(normalized))
                            .group_by(AccessEvent.person_id)
                            .all()
                        )
                        count_map = {p.lower(): cnt for p, cnt in counts}
                        best_pid = None
                        best_weight = -1
                        for p in normalized:
                            cnt = count_map.get(p.lower(), 0)
                            s = scores.get(p, 0.0)
                            weight = cnt * 2 + s * 10
                            if weight > best_weight:
                                best_weight = weight
                                best_pid = p
                        pid = best_pid
                    except Exception:
                        # Fallback: highest score
                        pid = max(scores, key=lambda x: scores[x]) if scores else None
            except Exception as e:
                print(f"[FaceDB] Qdrant auto-link error: {e}")

        # Step 2: Generate new UUID if no match found
        if not pid:
            pid = str(uuid.uuid4())
            # Seed Qdrant with this user's features
            if HAS_QDRANT and qdrant_is_available() and feats_np.shape[0] > 0:
                threading.Thread(
                    target=self._seed_qdrant,
                    args=(pid, feats_np),
                    daemon=True,
                ).start()

        # Step 3: Persist to PostgreSQL
        try:
            user.person_id = pid
            session.commit()
            print(f"[FaceDB] Auto-linked user '{user.name}' -> person_id={pid}")
        except Exception:
            session.rollback()

        return pid

    @staticmethod
    def _seed_qdrant(person_id: str, feats_np: np.ndarray) -> None:
        """Upsert user features to Qdrant (background thread)."""
        try:
            features = [f.tolist() for f in feats_np]
            if features:
                qdrant_upsert(person_id=person_id, features=features, camera_id="import")
        except Exception as e:
            print(f"[FaceDB] Seed Qdrant error: {e}")

    # ---- Matching ----

    def match_cosine(self, embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Match embedding using cosine similarity (dot product on normalized vectors).

        Returns:
            (person_id, name, similarity_score)
            Returns (None, None, -1.0) if no match or empty database.
        """
        with self._lock:
            if self.features_matrix.shape[0] == 0:
                return None, None, -1.0
            try:
                sims = np.dot(self.features_matrix, embedding)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                owner = self.features_owner_idx[best_idx]
                return self.person_ids[owner], self.names[owner], best_sim
            except Exception as e:
                print(f"[FaceDB] match_cosine error: {e}")
                return None, None, -1.0

    def match_l2(self, embedding: np.ndarray) -> Tuple[int, float]:
        """Match using L2 distance (legacy compatibility). Returns (person_idx, distance)"""
        with self._lock:
            if self.features_matrix.shape[0] == 0:
                return -1, float("inf")
            distances = np.linalg.norm(self.features_matrix - embedding, axis=1)
            best_raw = int(np.argmin(distances))
            owner = self.features_owner_idx[best_raw]
            return owner, float(distances[best_raw])

    def get_name(self, person_id: str) -> str:
        """Get user name by person_id"""
        return self.person_id_to_name.get(person_id, "Unknown")

    def refresh(self) -> None:
        """Reload features from PostgreSQL"""
        if HAS_DB:
            self._load_from_db()


# =============================================================================
# Face Tracker (enhanced with voting + Qdrant support)
# =============================================================================

@dataclass
class TrackedFace:
    """
    Track a face with voting-based identity confirmation.

    Supports:
    - High-confidence instant match (cosine >= pg_sim_high)
    - Medium-confidence voting (pg_sim_low <= cosine < pg_sim_high)
    - Qdrant fallback for unknown faces
    - New person creation after min_unknown_frames
    """
    object_id: int

    # Thresholds
    l2_threshold: float = 1.0
    min_streak: int = 3
    skip_reid: int = 3
    reid_interval: int = 30

    # Identity state
    label: str | None = None
    person_id: str | None = None
    score: float = 0.0

    # Voting
    votes: Dict[str, int] = field(default_factory=dict)
    unknown_count: int = 0

    # Internal
    _person: int = -1
    _streak: int = 0
    _distances: list[float] = field(default_factory=list)

    last_sgie: int = 0
    last_match_frame: int = 0
    last_qdrant_time: float = 0.0
    age: int = 0

    # Feature accumulation for new person
    feature_accum: list = field(default_factory=list)

    def should_run_sgie(self, frame: int) -> bool:
        """Check if SGIE should run based on frame interval"""
        interval = self.reid_interval if self.label else self.skip_reid
        return (frame - self.last_sgie) >= interval

    def add_match(self, person: int, distance: float) -> bool:
        """Add match result (L2 mode). Returns True if identity confirmed."""
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

    def confirm(self, name: str, pid: str = None) -> bool:
        """Confirm identity. Returns True if first confirmation."""
        is_new = self.label is None
        self.label = name
        if pid:
            self.person_id = pid
        self.score = sum(self._distances) / len(self._distances) if self._distances else 0.0
        self._person, self._streak, self._distances = -1, 0, []
        self.votes = {}
        self.unknown_count = 0
        if is_new:
            print(f"[CONFIRMED] oid={self.object_id} -> {name} (pid={pid}, score={self.score:.3f})")
        return is_new

    def add_vote(self, pid: str) -> int:
        """Add a vote for person_id. Returns current vote count."""
        if not isinstance(self.votes, dict):
            self.votes = {}
        self.votes[pid] = self.votes.get(pid, 0) + 1
        return self.votes[pid]

    def set_identity(self, pid: str, name: str, frame_num: int):
        """Set confirmed identity"""
        self.person_id = pid
        self.label = name
        self.last_match_frame = frame_num
        self.votes = {}
        self.unknown_count = 0


class TrackerManager:
    """Manage tracked faces per camera with auto-cleanup."""

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
        """Increment age and remove stale trackers."""
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
        if current_frame - self._last_cleanup >= self.cleanup_interval:
            return self.cleanup(current_frame)
        return []

    def stats(self) -> tuple[int, int, int]:
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
        if key not in self._events:
            self._events[key] = frame
            return True
        return False

    def contains(self, key: tuple[int, int]) -> bool:
        return key in self._events

    def discard(self, key: tuple[int, int]) -> None:
        self._events.pop(key, None)

    def cleanup(self, current_frame: int) -> list[tuple[int, int]]:
        removed = []
        to_remove = [k for k, f in self._events.items() if current_frame - f > self.max_age]
        for k in to_remove:
            del self._events[k]
            removed.append(k)
        return removed

    def auto_cleanup(self, current_frame: int) -> list[tuple[int, int]]:
        if current_frame % self.max_age == 0:
            return self.cleanup(current_frame)
        return []


# =============================================================================
# Main Processor (Hybrid: In-memory PG + Qdrant fallback)
# =============================================================================

@ProcessorRegistry.register("recognition")
class FaceRecognitionProcessor:
    def __init__(self, config: Dict[str, Any], sink: BaseSink, source_mapper=None):
        self._config = config
        self._sink = sink
        self._source_mapper = source_mapper
        params = config.get("params", {})

        # ---- Thresholds ----
        self.pg_sim_high = params.get("pg_sim_high", 0.58)
        self.pg_sim_low = params.get("pg_sim_low", 0.45)
        self.pg_sim_floor = params.get("pg_sim_floor", 0.40)
        self.qdrant_threshold = params.get("qdrant_threshold", 0.40)
        self.qdrant_low_threshold = params.get("qdrant_low_threshold", 0.40)
        self.min_unknown_frames = params.get("min_unknown_frames_before_create", 20)
        self.recheck_cooldown_sec = params.get("recheck_cooldown_sec", 3.0)
        self.stable_frames = params.get("stable_frames", 15)
        self.vote_threshold = params.get("vote_threshold", 2)
        self.qdrant_cooldown_sec = params.get("qdrant_cooldown_sec", 5.0)

        # ---- Frontal face config ----
        self.check_frontal = params.get("is_front_face", True)
        self.angle_threshold = params.get("angle_threshold", 15)
        self.symmetry_threshold = params.get("symmetry_threshold", 0.3)
        self.min_face_size = params.get("min_face_size", 50)
        self.frame_shape = None

        # ---- Alert cooldown ----
        self.alert_cooldown_sec = params.get("alert_cooldown_sec", 20)
        self.last_alert_at: Dict[tuple, float] = {}

        # ---- Database config ----
        db_cfg = config.get("database", {})
        use_db = db_cfg.get("enabled", True) and HAS_DB

        # ---- Qdrant config (override module-level constants) ----
        qdrant_cfg = config.get("qdrant", {})
        if qdrant_cfg:
            import apps.face.qdrant_client_service as _qcs
            from apps.face.search import reinit_storage
            if qdrant_cfg.get("host"):
                _qcs.QDRANT_HOST = qdrant_cfg["host"]
            if qdrant_cfg.get("port"):
                _qcs.QDRANT_PORT = int(qdrant_cfg["port"])
            if qdrant_cfg.get("api_key"):
                _qcs.QDRANT_API_KEY = qdrant_cfg["api_key"]
            if qdrant_cfg.get("collection"):
                _qcs.COLLECTION_NAME = qdrant_cfg["collection"]
            # Re-initialize storage with updated config
            reinit_storage()
        
        # Check Qdrant availability
        qdrant_available = HAS_QDRANT and qdrant_is_available()
        print(f"[FaceRecognitionProcessor] use_db={use_db}, HAS_DB={HAS_DB}, HAS_QDRANT={HAS_QDRANT}, Qdrant_Available={qdrant_available}")

        # ---- Load face database ----
        self._db = FaceDatabase(use_db=use_db)

        # ---- Tracker manager & event tracking ----
        self._trackers = TrackerManager(params)
        self._sent_faces = EventSet(max_age=params.get("max_age", 30))

        # ---- New person management ----
        self.pending_new: Dict[tuple, dict] = {}
        self.new_person_cache: Dict[str, tuple] = {}
        self.flush_interval_sec = params.get("flush_interval_sec", 5)

        # ---- Cleanup runner ----
        cleanup_interval = params.get("cleanup_interval", 10) * 1000
        self._cleanup_runner = IntervalRunner(cleanup_interval, self._cleanup)

        # ---- DB refresh interval ----
        self._refresh_interval_sec = params.get("refresh_interval", 30)
        self._last_refresh = time.time()

        # ---- HTTP Event Sender ----
        http_cfg = params.get("http_event", {})
        self._http_sender = HttpEventSender(http_cfg)

        # ---- Image save directories ----
        os.makedirs(CROP_FACE_DIR, exist_ok=True)
        os.makedirs(FULL_FRAME_DIR, exist_ok=True)

        # ---- Pending HTTP events ----
        self._pending_http_events: Dict[tuple, dict] = {}
        self._frame_event_queue: Queue = Queue(maxsize=50)
        self._frame_dispatcher_running = False
        self._frame_dispatcher_thread: Optional[threading.Thread] = None
        self._connected_sources: set = set()
        
        print(f"[FaceRecognitionProcessor] Initialized: "
              f"{len(self._db.names)} users, "
              f"{self._db.features_matrix.shape[0]} vectors, "
              f"frontal_check={self.check_frontal}")

    @property
    def name(self) -> str:
        return "recognition"

    def _get_stats(self) -> dict:
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

    # =========================================================================
    # Probe Callbacks
    # =========================================================================

    def _tracker_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Decide whether to skip SGIE for each face (pre-SGIE filter)"""
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK

        for frame, obj in BatchIterator(batch):
            # Skip small faces
            if should_skip_face(obj, self.min_face_size):
                mark_skip_sgie(obj)
                continue

            # Frontal face check (if landmarks available and enabled)
            if self.check_frontal:
                # Try to extract landmarks; skip non-frontal
                landmarks = extract_landmarks(obj, self.frame_shape)
                if landmarks is not None:
                    if not is_frontal_face(landmarks, self.angle_threshold, self.symmetry_threshold):
                        mark_skip_sgie(obj)
                        continue

            # Skip if tracker says not ready for SGIE
            trk = self._trackers.get(frame.source_id, obj.object_id)
            if trk and not trk.should_run_sgie(frame.frame_num):
                mark_skip_sgie(obj)

        return Gst.PadProbeReturn.OK

    def _sgie_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Process recognition results; queue events for frame dispatcher (appsink)."""
        gst_buffer = info.get_buffer()
        batch = get_batch_meta(gst_buffer)
        if not batch:
            return Gst.PadProbeReturn.OK

        # Periodic DB refresh
        now = time.time()
        if now - self._last_refresh >= self._refresh_interval_sec:
            self._last_refresh = now
            threading.Thread(target=self._db.refresh, daemon=True).start()

        # Periodic flush new persons to Qdrant
        self._flush_new_persons()

        for frame, obj in BatchIterator(batch):
            sid = frame.source_id

            # Auto-connect appsink chain for new sources
            if sid not in self._connected_sources and hasattr(self._sink, "connect_source"):
                if self._sink.connect_source(sid):
                    self._connected_sources.add(sid)

            name, state, score = self._process_face(sid, obj, frame.frame_num)
            update_display(obj, name, score, state)

            key = (sid, obj.object_id)
            if key not in self._pending_http_events:
                continue

            event_info = self._pending_http_events.pop(key)
            rect = obj.rect_params
            bbox = (int(rect.left), int(rect.top), int(rect.width), int(rect.height))
            obj_meta_snapshot = {
                "left": float(rect.left),
                "top": float(rect.top),
                "width": float(rect.width),
                "height": float(rect.height),
            }

            try:
                self._frame_event_queue.put_nowait({
                    "event_info": event_info,
                    "bbox": bbox,
                    "source_id": sid,
                    "obj_meta": obj_meta_snapshot,
                })
            except Full:
                print(f"[SgieProbe] Frame event queue full, dropping event for {key}")

        return Gst.PadProbeReturn.OK

    def _frame_dispatcher_worker(self) -> None:
        """Background thread: get frame from per-camera appsink, crop face, send HTTP event."""
        while self._frame_dispatcher_running:
            try:
                item = self._frame_event_queue.get(timeout=0.5)
            except Exception:
                continue

            sid = item.get("source_id")
            full_frame = None

            if hasattr(self._sink, "get_frame") and sid is not None:
                full_frame = self._sink.get_frame(sid)
            elif hasattr(self._sink, "get_latest_frame"):
                full_frame = self._sink.get_latest_frame()

            if full_frame is None:
                print(f"[FrameDispatcher] No frame for source {sid}, dropping event")
                continue

            l, t, w, h = item["bbox"]
            h_img, w_img = full_frame.shape[:2]
            padding = 0.1
            x1 = max(0, int(l - w * padding))
            y1 = max(0, int(t - h * padding))
            x2 = min(w_img, int(l + w + w * padding))
            y2 = min(h_img, int(t + h + h * padding))
            face_crop = full_frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else full_frame

            self._save_and_send_http(item["event_info"], full_frame, face_crop, item.get("obj_meta"))                 
    
    # =========================================================================
    # Face Processing Pipeline
    # =========================================================================

    def _process_face(self, source_id: int, obj_meta, frame: int) -> tuple[str, str, float]:
        """
        Process a single face through the hybrid matching pipeline.

        Returns:
            (display_name, state, score)
        """
        oid = obj_meta.object_id
        trk = self._trackers.get_or_create(source_id, oid, frame)
        trk.age = 0

        emb = extract_embedding(obj_meta)
        if emb is None:
            # No embedding — return existing state
            if trk.label:
                return trk.label, "confirmed", trk.score
            return "", "unknown", 0.0

        # Normalize embedding
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        trk.last_sgie = frame

        # If already confirmed and still stable, skip re-matching
        if trk.person_id and (frame - trk.last_match_frame) < self.stable_frames:
            return trk.label or trk.person_id, "confirmed", trk.score

        # ----- Hybrid matching -----
        person_id, name = self._match_face(emb, source_id, oid, trk, frame)
        print("====> P_ID", person_id)
        print("====> NAME", name)
        if person_id:
            return name or person_id, "confirmed", trk.score
        return "", "unknown", 0.0

    def _match_face(
        self, embedding: np.ndarray, source_id: int, oid: int,
        trk: TrackedFace, frame: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Hybrid matching pipeline:
        A. High-confidence PG match → instant confirm
        B. Medium PG match → voting
        C. Low/no PG match → Qdrant fallback
        D. Still unknown → create new person

        Returns:
            (person_id, display_name) or (None, None)
        """
        # ---- Step A & B: In-memory PostgreSQL cosine match ----
        best_pid, best_name, best_sim = self._db.match_cosine(embedding)

        # A. High confidence — instant match
        if best_pid and best_sim >= self.pg_sim_high:
            trk.set_identity(best_pid, best_name, frame)
            trk.score = best_sim
            self._send_event(source_id, oid, best_name, frame, best_pid)
            return best_pid, best_name

        # B. Medium confidence — voting
        if best_pid and self.pg_sim_low <= best_sim < self.pg_sim_high:
            vote_count = trk.add_vote(best_pid)
            if vote_count >= self.vote_threshold:
                trk.set_identity(best_pid, best_name, frame)
                trk.score = best_sim
                self._send_event(source_id, oid, best_name, frame, best_pid)
                return best_pid, best_name

        # ---- Step C: Qdrant fallback ----
        if HAS_QDRANT and qdrant_is_available() and (best_sim < self.pg_sim_floor or not best_pid):
            now = time.time()
            if (now - trk.last_qdrant_time) >= self.qdrant_cooldown_sec:
                trk.last_qdrant_time = now
                q_pid, q_name, q_score = self._qdrant_match(embedding)
                if q_pid:
                    trk.set_identity(q_pid, q_name, frame)
                    trk.score = q_score
                    self._send_event(source_id, oid, q_name, frame, q_pid)
                    return q_pid, q_name

        # ---- Step D: Create new person if unknown for too long ----
        if HAS_QDRANT and qdrant_is_available():
            new_result = self._handle_unknown(embedding, source_id, oid, trk, frame)
            if new_result:
                return new_result

        return None, None

    # =========================================================================
    # Qdrant Integration
    # =========================================================================

    def _qdrant_match(self, embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Search Qdrant for matching face.

        Returns:
            (person_id, name, best_score) or (None, None, 0.0)
        """
        try:
            hits = qdrant_search(
                query_vector=embedding.tolist(),
                limit=10,
                similarity_threshold=self.qdrant_threshold,
            )
            if not hits:
                return None, None, 0.0

            # Collect candidates
            candidates: Dict[str, float] = {}
            for h in hits:
                payload = getattr(h, "payload", None)
                if isinstance(payload, list) and len(payload) > 0:
                    payload = payload[0]
                if not isinstance(payload, dict):
                    continue
                pid = payload.get("person_id")
                score = float(getattr(h, "score", 0))
                if pid:
                    candidates[pid] = max(candidates.get(pid, 0.0), score)

            if not candidates:
                return None, None, 0.0

            # Pick best candidate (by score, or use DB frequency)
            best_pid = self._pick_best_candidate(candidates)
            if best_pid:
                name = self._db.get_name(best_pid)
                return best_pid, name, candidates[best_pid]

            return None, None, 0.0
        except Exception as e:
            print(f"[WARN] _qdrant_match error: {e}")
            return None, None, 0.0

    def _pick_best_candidate(self, candidates: Dict[str, float]) -> Optional[str]:
        """Pick best person_id from candidates using DB frequency + score weighting."""
        if not candidates:
            return None

        if HAS_DB:
            try:
                with SessionLocal() as session:
                    return self._get_most_used_person_id(session, list(candidates.keys()), candidates)
            except Exception:
                pass

        # Fallback: highest score
        return max(candidates, key=lambda pid: candidates[pid])

    def _get_most_used_person_id(
        self, session, candidate_ids: list, scores: Dict[str, float] = None
    ) -> Optional[str]:
        """Select person_id weighted by DB frequency + match score."""
        if not candidate_ids:
            return None
        try:
            normalized = [pid.strip().lower() for pid in candidate_ids if pid]
            counts = (
                session.query(AccessEvent.person_id, func.count(AccessEvent.id))
                .filter(func.lower(func.trim(AccessEvent.person_id)).in_(normalized))
                .group_by(AccessEvent.person_id)
                .all()
            )
            count_map = {pid.lower(): cnt for pid, cnt in counts}

            best_pid = None
            best_weight = -1
            for pid in normalized:
                cnt = count_map.get(pid.lower(), 0)
                score = scores.get(pid, 0.0) if scores else 0.0
                weight = cnt * 2 + score * 10
                if weight > best_weight:
                    best_weight = weight
                    best_pid = pid
            if best_pid:
                return best_pid
            if scores:
                return max(scores, key=lambda x: scores[x])
        except Exception as e:
            print(f"[WARN] _get_most_used_person_id error: {e}")

        return sorted(candidate_ids)[0] if candidate_ids else None

    # =========================================================================
    # Unknown Face Handling (New Person Creation)
    # =========================================================================

    def _handle_unknown(
        self, embedding: np.ndarray, source_id: int, oid: int,
        trk: TrackedFace, frame: int
    ) -> Optional[Tuple[str, str]]:
        """
        Handle faces that don't match PG or Qdrant.
        After min_unknown_frames, create a new person_id and upsert to Qdrant.

        Returns:
            (person_id, name) if new person created, else None.
        """
        trk.unknown_count = getattr(trk, "unknown_count", 0) + 1
        key = (source_id, oid)

        if key not in self.pending_new:
            self.pending_new[key] = {
                "start_frame": frame,
                "feature_accum": [embedding.copy()],
                "last_check": time.time(),
            }
            return None

        info = self.pending_new[key]
        info["feature_accum"].append(embedding.copy())

        # Not enough frames yet
        if (frame - info["start_frame"]) < self.min_unknown_frames:
            return None

        # Recheck Qdrant with lower threshold before creating
        now = time.time()
        if (now - info["last_check"]) > self.recheck_cooldown_sec:
            info["last_check"] = now
            try:
                hits = qdrant_search(
                    query_vector=embedding.tolist(),
                    limit=10,
                    similarity_threshold=self.qdrant_low_threshold,
                )
                if hits:
                    candidates = {}
                    for h in hits:
                        payload = getattr(h, "payload", None)
                        if isinstance(payload, list) and len(payload) > 0:
                            payload = payload[0]
                        if not isinstance(payload, dict):
                            continue
                        pid = payload.get("person_id")
                        score = float(getattr(h, "score", 0))
                        if pid:
                            candidates[pid] = max(candidates.get(pid, 0.0), score)
                    if candidates:
                        best = self._pick_best_candidate(candidates)
                        if best:
                            name = self._db.get_name(best)
                            trk.set_identity(best, name, frame)
                            trk.score = candidates[best]
                            del self.pending_new[key]
                            self._send_event(source_id, oid, name, frame, best)
                            return best, name
            except Exception as e:
                print(f"[WARN] recheck qdrant error: {e}")

        # Create new person
        return self._create_new_person(key, trk, embedding, frame, source_id, oid)

    def _create_new_person(
        self, key: tuple, trk: TrackedFace, embedding: np.ndarray,
        frame: int, source_id: int, oid: int
    ) -> Tuple[str, str]:
        """Create a new person_id and schedule Qdrant upsert."""
        info = self.pending_new.pop(key, None)

        # Average accumulated features
        avg_feat = embedding.copy()
        if info and info["feature_accum"]:
            try:
                mat = np.stack(info["feature_accum"], axis=0)
                avg_feat = np.mean(mat, axis=0).astype(np.float32)
                n = np.linalg.norm(avg_feat)
                if n > 0:
                    avg_feat = avg_feat / n
            except Exception:
                pass

        new_pid = str(uuid.uuid4())
        new_name = new_pid  # Unknown person, name = pid

        trk.set_identity(new_pid, new_name, frame)
        trk.score = 0.0

        # Cache for async upsert
        self.new_person_cache[new_pid] = (avg_feat, time.time())

        print(f"[NEW PERSON] oid={oid} -> pid={new_pid}")
        self._send_event(source_id, oid, new_name, frame, new_pid)
        return new_pid, new_name

    def _flush_new_persons(self) -> None:
        """Flush new person features to Qdrant (async)."""
        if not HAS_QDRANT or not qdrant_is_available():
            return
        now = time.time()
        to_remove = []
        for pid, (vec, t) in list(self.new_person_cache.items()):
            if now - t > 2.0:
                threading.Thread(
                    target=self._async_upsert,
                    args=(pid, vec, "auto"),
                    daemon=True,
                ).start()
                to_remove.append(pid)
        for pid in to_remove:
            self.new_person_cache.pop(pid, None)

    def _async_upsert(self, person_id: str, vec: np.ndarray, camera_id: str) -> None:
        """Async upsert to Qdrant."""
        try:
            qdrant_upsert(person_id=person_id, features=[vec.tolist()], camera_id=camera_id)
        except Exception as e:
            print(f"[WARN] Qdrant upsert error: {e}")

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _send_event(
        self, source_id: int, object_id: int, name: str,
        frame: int, person_id: str = None
    ) -> None:
        """Send face detection event with cooldown. Queues HTTP event for _sgie_probe."""
        # Cooldown check
        key = (source_id, object_id)
        now = time.time()
        pid_key = (source_id, person_id or name)
        last = self.last_alert_at.get(pid_key, 0)
        if now - last < self.alert_cooldown_sec:
            return
        self.last_alert_at[pid_key] = now

        # Dedup via EventSet
        if not self._sent_faces.add(key, frame):
            return

        camera_id = self._source_mapper.get_camera_id(source_id) if self._source_mapper else None

        # Queue HTTP event — will be processed in _sgie_probe where gst_buffer is available
        self._pending_http_events[key] = {
            "person_id": person_id or name,
            "name": name,
            "camera_id": camera_id or str(source_id),
            "source_id": source_id,
        }

        event = {
            "type": "face_detected",
            "camera_id": camera_id,
            "source_id": source_id,
            "name": name,
            "person_id": person_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "object_id": object_id,
            "avatar": self._db.avatars.get(name),
        }
        self._sink.send_event(event)

    # =========================================================================
    # HTTP Event: Save Images + Send
    # =========================================================================

    def _save_and_send_http(self, event_info: dict, full_frame: np.ndarray, face_crop: np.ndarray, obj_meta):
        """Save crop + annotated full frame images to disk, then queue HTTP event send."""
        person_id = event_info["person_id"]
        name = event_info.get("name")
        camera_id = event_info.get("camera_id", "0")
        prefix = f"{camera_id}_{person_id}"

        annotated_full = full_frame.copy()
        if obj_meta is not None:
            draw_event_bbox(annotated_full, obj_meta, name, person_id)

        crop_path = save_image(face_crop, CROP_FACE_DIR, prefix)
        if crop_path:
            print(f"[HttpEvent] Crop saved: {crop_path}")

        full_path = save_image(annotated_full, FULL_FRAME_DIR, prefix)
        if full_path:
            print(f"[HttpEvent] Full saved: {full_path}")

        try:
            stream_id = str(int(camera_id))
        except ValueError:
            digits = "".join(c for c in camera_id if c.isdigit())
            stream_id = digits if digits else "0"
            print(f"[HttpEvent] camera_id={camera_id} -> stream_id={stream_id}")

        self._http_sender.send(
            person_id=person_id,
            full_frame=annotated_full,
            face_crop=face_crop,
            stream_id=stream_id,
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def _cleanup(self, current_frame: int) -> None:
        """Cleanup stale trackers, events, and alert cooldowns."""
        self._trackers.auto_cleanup(current_frame)
        self._sent_faces.auto_cleanup(current_frame)

        # Clean old alert cooldowns (older than 5 minutes)
        now = time.time()
        stale_keys = [k for k, t in self.last_alert_at.items() if now - t > 300]
        for k in stale_keys:
            del self.last_alert_at[k]

        # Clean stale pending_new entries
        stale_pending = [
            k for k, v in self.pending_new.items()
            if now - v.get("last_check", now) > 60
        ]
        for k in stale_pending:
            del self.pending_new[k]

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        print(f"[FaceRecognitionProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        if self._cleanup_runner:
            self._cleanup_runner.start()
        self._http_sender.start()
        # Start frame event dispatcher
        self._frame_dispatcher_running = True
        self._frame_dispatcher_thread = threading.Thread(
            target=self._frame_dispatcher_worker,
            daemon=True,
            name="frame-event-dispatcher",
        )
        self._frame_dispatcher_thread.start()
        print("[FaceRecognitionProcessor] Started")

    def on_stop(self) -> None:
        if self._cleanup_runner:
            self._cleanup_runner.stop()
        self._http_sender.stop()
        # Stop frame event dispatcher
        self._frame_dispatcher_running = False
        if self._frame_dispatcher_thread:
            self._frame_dispatcher_thread.join(timeout=3.0)
            self._frame_dispatcher_thread = None
        self._flush_new_persons()
        print("[FaceRecognitionProcessor] Stopped")

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Return processor statistics"""
        total, confirmed, pending = self._trackers.stats() if self._trackers else (0, 0, 0)
        return {
            "faces_in_database": len(self._db.names) if self._db else 0,
            "vectors_in_memory": self._db.features_matrix.shape[0] if self._db else 0,
            "trackers_total": total,
            "trackers_confirmed": confirmed,
            "trackers_pending": pending,
            "pending_new_persons": len(self.pending_new),
            "qdrant_available": HAS_QDRANT and qdrant_is_available(),
            "db_available": HAS_DB,
            "frontal_check": self.check_frontal,
        }

    @property
    def database(self) -> Optional[FaceDatabase]:
        return self._db

