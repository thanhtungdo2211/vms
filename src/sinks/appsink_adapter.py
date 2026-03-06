"""Appsink Adapter with nvstreamdemux — per-camera frame capture.

Creates nvstreamdemux internally.  For each source_id that is connected,
a chain  queue → nvvideoconvert → capsfilter(I420) → appsink  is linked
to the corresponding demux src pad.  Each appsink callback stores a BGR
frame indexed by source_id so processors can call get_frame(source_id).

Pattern follows StreamPublisher (same demux pad management).
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink
from src.common import make_element, link_chain

logger = logging.getLogger(__name__)

STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


class AppsinkAdapter(BaseSink):
    """Demux-based appsink adapter — one appsink per camera.

    create() returns an nvstreamdemux element whose static *sink* pad
    is linked to by the upstream pipeline (sgie).  Per-camera chains
    are connected later via connect_source().
    """

    _counter = 0

    def __init__(self, max_buffers: int = 2, drop: bool = True, max_cameras: int = 16):
        self._max_buffers = max_buffers
        self._drop = drop
        self._max_cameras = max_cameras

        self._pipeline: Optional[Gst.Pipeline] = None
        self._demux: Optional[Gst.Element] = None

        self._frames: Dict[int, np.ndarray] = {}
        self._chains: Dict[int, list[Gst.Element]] = {}
        self._pads: Dict[int, Gst.Pad] = {}
        self._lock = threading.Lock()

        AppsinkAdapter._counter += 1
        self._id = AppsinkAdapter._counter

    # ------------------------------------------------------------------
    # BaseSink interface
    # ------------------------------------------------------------------

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        self._pipeline = pipeline
        self._demux = make_element(
            "nvstreamdemux", f"appsink_demux{self._id}", {}
        )
        pipeline.add(self._demux)

        # Pre-request pads (required while pipeline is in NULL state)
        for i in range(self._max_cameras):
            pad = self._demux.request_pad_simple(f"src_{i}")
            if pad:
                logger.info(f"[AppsinkAdapter] Pre-requested src_{i}")

        return self._demux

    def start(self) -> None:
        pass

    def stop(self) -> None:
        with self._lock:
            self._frames.clear()

    # ------------------------------------------------------------------
    # Per-camera chain management
    # ------------------------------------------------------------------

    def connect_source(self, source_id: int) -> bool:
        """Create and link an appsink chain for *source_id*.

        Safe to call multiple times for the same source_id (no-op if
        already connected).
        """
        with self._lock:
            if source_id in self._chains:
                return True

        if self._demux is None or self._pipeline is None:
            logger.error("[AppsinkAdapter] connect_source called before create()")
            return False

        p = f"apsrc{self._id}_{source_id}"
        try:
            chain = [
                make_element("queue", f"{p}_q", {"max-size-buffers": 2, "leaky": 2}),
                make_element("nvvideoconvert", f"{p}_conv", {"nvbuf-memory-type": 3}),
                make_element("capsfilter", f"{p}_caps", {
                    "caps": Gst.Caps.from_string("video/x-raw, format=I420"),
                }),
            ]

            appsink = Gst.ElementFactory.make("appsink", f"{p}_sink")
            if not appsink:
                raise RuntimeError("Cannot create element: appsink")
            appsink.set_property("emit-signals", True)
            appsink.set_property("drop", self._drop)
            appsink.set_property("max-buffers", self._max_buffers)
            appsink.set_property("sync", False)
            appsink.connect("new-sample", self._make_callback(source_id))
            chain.append(appsink)

            for elem in chain:
                self._pipeline.add(elem)
            link_chain(chain)

            # Sync element states to pipeline
            _, cur_state, _ = self._pipeline.get_state(0)
            for elem in chain:
                elem.sync_state_with_parent()

            # Link demux src pad → queue sink pad
            demux_pad = self._get_demux_pad(source_id)
            if not demux_pad:
                raise RuntimeError(f"Cannot get demux pad src_{source_id}")

            queue_sink = chain[0].get_static_pad("sink")
            ret = demux_pad.link(queue_sink)
            if ret != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link demux src_{source_id}: {ret}")

            with self._lock:
                self._chains[source_id] = chain
                self._pads[source_id] = demux_pad

            logger.info(f"[AppsinkAdapter] Connected source {source_id}")
            return True

        except Exception as e:
            logger.error(f"[AppsinkAdapter] connect_source({source_id}) failed: {e}",
                         exc_info=True)
            return False

    def disconnect_source(self, source_id: int) -> bool:
        """Unlink and remove the appsink chain for *source_id*."""
        with self._lock:
            chain = self._chains.pop(source_id, None)
            demux_pad = self._pads.pop(source_id, None)
            self._frames.pop(source_id, None)

        if not chain:
            return False

        try:
            if demux_pad:
                queue_sink = chain[0].get_static_pad("sink")
                if queue_sink and queue_sink.is_linked():
                    demux_pad.unlink(queue_sink)

            for elem in reversed(chain):
                elem.set_state(Gst.State.NULL)
                elem.get_state(STATE_CHANGE_TIMEOUT)
                self._pipeline.remove(elem)

            logger.info(f"[AppsinkAdapter] Disconnected source {source_id}")
            return True

        except Exception as e:
            logger.error(f"[AppsinkAdapter] disconnect_source({source_id}) failed: {e}",
                         exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Demux pad helpers (same pattern as StreamPublisher)
    # ------------------------------------------------------------------

    def _get_demux_pad(self, source_id: int) -> Optional[Gst.Pad]:
        pad_name = f"src_{source_id}"
        it = self._demux.iterate_src_pads()
        while True:
            result, pad = it.next()
            if result == Gst.IteratorResult.DONE:
                break
            if result == Gst.IteratorResult.OK and pad.get_name() == pad_name:
                return pad
        return self._demux.request_pad_simple(pad_name)

    # ------------------------------------------------------------------
    # Appsink callback factory
    # ------------------------------------------------------------------

    def _make_callback(self, source_id: int):
        """Return a new-sample callback bound to *source_id*."""
        def on_new_sample(sink) -> Gst.FlowReturn:
            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK

            buf = sample.get_buffer()
            caps = sample.get_caps()
            struct = caps.get_structure(0)
            w = struct.get_value("width")
            h = struct.get_value("height")
            fmt = struct.get_value("format")

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK

            try:
                bgr = _decode(map_info.data, w, h, fmt)
                if bgr is not None:
                    with self._lock:
                        self._frames[source_id] = bgr
            finally:
                buf.unmap(map_info)

            return Gst.FlowReturn.OK

        return on_new_sample

    # ------------------------------------------------------------------
    # Public API for processors
    # ------------------------------------------------------------------

    def get_frame(self, source_id: int) -> Optional[np.ndarray]:
        """Return a copy of the latest BGR frame for *source_id*, or None."""
        with self._lock:
            f = self._frames.get(source_id)
            return f.copy() if f is not None else None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Backward-compat: return any available frame (first found)."""
        with self._lock:
            for f in self._frames.values():
                return f.copy()
            return None

    def connected_sources(self) -> list[int]:
        with self._lock:
            return list(self._chains.keys())


# ======================================================================
# Module-level decode helper (avoids per-instance method overhead)
# ======================================================================

def _decode(data: bytes, w: int, h: int, fmt: str) -> Optional[np.ndarray]:
    if fmt == "I420":
        yuv = np.frombuffer(data, dtype=np.uint8).reshape((h * 3 // 2, w))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    if fmt == "NV12":
        yuv = np.frombuffer(data, dtype=np.uint8).reshape((h * 3 // 2, w))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if fmt == "RGBA":
        rgba = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if fmt == "BGRx":
        bgrx = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))
        return bgrx[:, :, :3].copy()
    if fmt == "BGR":
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
    return None
