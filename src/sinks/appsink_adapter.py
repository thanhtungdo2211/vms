"""Appsink Adapter - Captures frames as numpy arrays for downstream processing."""

import threading
from typing import Optional, Tuple

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink
from src.common import make_element, link_chain, get_nvvidconv_props


class AppsinkAdapter(BaseSink):
    """Appsink adapter that stores the latest frame as a BGR numpy array.

    Upstream can be NVMM or system memory in any common format (I420, NV12, RGBA, etc.).
    The adapter adds nvvideoconvert → capsfilter(I420) → appsink, then converts
    I420 → BGR in the new-sample callback via OpenCV.
    """

    _counter = 0

    def __init__(self, max_buffers: int = 2, drop: bool = True):
        self._max_buffers = max_buffers
        self._drop = drop
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        AppsinkAdapter._counter += 1
        self._id = AppsinkAdapter._counter

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        p = f"appsink{self._id}"

        chain = [
            make_element("queue", f"{p}_q", {"max-size-buffers": 2, "leaky": 2}),
            make_element("nvvideoconvert", f"{p}_conv", get_nvvidconv_props()),
            make_element("capsfilter", f"{p}_caps", {
                "caps": Gst.Caps.from_string("video/x-raw, format=I420"),
            }),
        ]

        # chain = [
        #     make_element("queue", f"{p}_q", {"max-size-buffers": 2, "leaky": 2})
        # ]        

        appsink = Gst.ElementFactory.make("appsink", f"{p}_sink")
        if not appsink:
            raise RuntimeError("Cannot create element: appsink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("drop", self._drop)
        appsink.set_property("max-buffers", self._max_buffers)
        appsink.set_property("sync", False)
        appsink.connect("new-sample", self._on_new_sample)
        chain.append(appsink)

        for elem in chain:
            pipeline.add(elem)
        link_chain(chain)

        return chain[0]

    # ------------------------------------------------------------------

    def _on_new_sample(self, sink) -> Gst.FlowReturn:
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
            bgr = self._decode(map_info.data, w, h, fmt)
            if bgr is not None:
                with self._lock:
                    self._latest_frame = bgr
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    @staticmethod
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

    # ------------------------------------------------------------------
    # Public API for processors
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent BGR frame, or None."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ------------------------------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        with self._lock:
            self._latest_frame = None
