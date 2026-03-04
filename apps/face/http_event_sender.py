"""
Async HTTP event sender for face recognition events.

Sends access events with face crop + full frame images to the configured endpoint.
Uses a background thread + queue to avoid blocking the DeepStream pipeline.
"""

import io
import time
import threading
from queue import Queue, Full
from datetime import datetime, timezone
from typing import Optional

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    print("[WARN] httpx not available, HTTP event sending disabled")


class HttpEventSender:
    """
    Send face access events via HTTP POST (multipart/form-data).

    Format matches send_access_event.py:
      - POST /api/v1/event/access-events
      - Fields: person_id, stream_id, time_access
      - Files: image_face (cropped), image_full (full frame)

    Uses a background thread with a queue so GStreamer pipeline is never blocked.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: http_event section from config.yaml, e.g.:
                base_url: "http://192.168.6.39:5555"
                endpoint: "/api/v1/event/access-events"
                token: ""
                stream_id: 1
                timeout: 60.0
        """
        self._base_url = config.get("base_url", "").rstrip("/")
        self._endpoint = config.get("endpoint", "/api/v1/event/access-events")
        self._token = str(config.get("token", "") or "").strip()
        self._timeout = float(config.get("timeout", 60.0))
        self._max_queue = int(config.get("max_queue", 50))

        self._url = f"{self._base_url}{self._endpoint}"
        self._queue: Queue = Queue(maxsize=self._max_queue)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if not self._base_url:
            print("[HttpEventSender] No base_url configured, disabled")
        elif not HAS_HTTPX:
            print("[HttpEventSender] httpx not installed, disabled")
        else:
            print(f"[HttpEventSender] Ready -> {self._url}")

    @property
    def enabled(self) -> bool:
        return bool(self._base_url) and HAS_HTTPX and HAS_CV2

    def start(self):
        """Start the background sender thread."""
        if not self.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="http-event-sender")
        self._thread.start()
        print("[HttpEventSender] Worker started")

    def stop(self):
        """Stop the background sender thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            # Put sentinel to unblock queue.get()
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
            self._thread.join(timeout=5.0)
        print("[HttpEventSender] Worker stopped")

    def send(
        self,
        person_id: str,
        full_frame: np.ndarray,
        face_crop: np.ndarray,
        stream_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        """
        Queue an event for async sending. Non-blocking.

        Args:
            person_id: The recognized person ID
            full_frame: Full frame as numpy array (BGR, HWC)
            face_crop: Cropped face as numpy array (BGR, HWC)
            stream_id: Override default stream_id
            extra: Extra metadata (ignored in HTTP but logged)
        """
        if not self.enabled or not self._running:
            return

        # Encode images to JPEG bytes immediately (numpy arrays may be recycled)
        try:
            _, face_jpg = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, full_jpg = cv2.imencode(".jpg", full_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as e:
            print(f"[HttpEventSender] encode error: {e}")
            return

        event = {
            "person_id": person_id,
            # "stream_id": stream_id or self._default_stream_id,
            "stream_id": stream_id,
            "time_access": datetime.utcnow().replace(microsecond=0).isoformat(),
            "face_jpg": face_jpg.tobytes(),
            "full_jpg": full_jpg.tobytes(),
        }

        # print("===> Camera", event.get("stream_id"))
        
        try:
            self._queue.put_nowait(event)
        except Full:
            print("[HttpEventSender] Queue full, dropping event")

    def _worker(self):
        """Background worker: consume queue and POST events."""
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
            except Exception:
                continue

            if event is None:
                continue

            self._post_event(event)

    def _post_event(self, event: dict):
        """POST a single event to the server."""
        try:
            headers = {}
            if self._token:
                token = self._token
                if not token.lower().startswith("bearer "):
                    token = f"Bearer {token}"
                headers["Authorization"] = token

            data = {
                "person_id": event["person_id"],
                "stream_id": event["stream_id"],
                "time_access": event["time_access"],
            }

            files = {
                "image_face": ("face.jpg", io.BytesIO(event["face_jpg"]), "image/jpeg"),
                "image_full": ("full.jpg", io.BytesIO(event["full_jpg"]), "image/jpeg"),
            }

            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url, data=data, files=files, headers=headers)

            if resp.is_success:
                print(f"[HttpEventSender] OK {resp.status_code} person={event['person_id']}")
            else:
                print(f"[HttpEventSender] FAIL {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[HttpEventSender] POST error: {e}")