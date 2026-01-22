"""Common utilities for DeepStream pipeline."""

import ctypes
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import yaml

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

_gst_lock = threading.Lock()


# =============================================================================
# Platform Detection
# =============================================================================

@dataclass
class PlatformInfo:
    """Platform config: Jetson vs dGPU."""
    is_jetson: bool
    name: str
    nvbuf_memory_type: int  # 0=DEFAULT, 3=CUDA_UNIFIED
    hw_encoder: str         # "nvv4l2h264enc" or ""
    compute_hw: int         # 1=GPU, 2=VIC


@lru_cache(maxsize=1)
def detect_platform() -> PlatformInfo:
    """Detect Jetson vs dGPU, return optimal settings."""
    is_jetson = os.path.exists("/etc/nv_tegra_release") or os.path.exists("/proc/device-tree/model")
    if is_jetson:
        return PlatformInfo(True, "jetson", 0, "nvv4l2h264enc", 2)
    return PlatformInfo(False, "dgpu", 3, "", 1)


def get_encoder_element(bitrate: int) -> Tuple[str, dict]:
    """Get encoder (factory, props). Hardware first, fallback x264enc."""
    platform = detect_platform()

    if platform.hw_encoder and Gst.ElementFactory.find(platform.hw_encoder):
        return (
            platform.hw_encoder,
            {"bitrate": bitrate, "preset-level": 1, "iframeinterval": 30,
             "control-rate": 1, "maxperf-enable": True}
        )

    return (
        "x264enc",
        {"bitrate": bitrate // 1000, "speed-preset": "ultrafast",
         "tune": "zerolatency", "threads": 4, "bframes": 0, "key-int-max": 30}
    )


def get_nvvidconv_props() -> dict:
    """Get platform-optimal nvvideoconvert properties."""
    p = detect_platform()
    return {"compute-hw": p.compute_hw, "nvbuf-memory-type": p.nvbuf_memory_type}


# =============================================================================
# Config Loader
# =============================================================================

def load_config(path: str) -> dict:
    """Load YAML with ${VAR:default} expansion."""
    with open(path) as f:
        text = f.read()
    expanded = re.sub(r'\$\{(\w+):?([^}]*)\}',
                      lambda m: os.environ.get(m.group(1), m.group(2) or ""), text)
    return yaml.safe_load(expanded)


# =============================================================================
# GStreamer Element Factory
# =============================================================================

def make_element(factory: str, name: Optional[str], props: dict = None) -> Gst.Element:
    """Create GStreamer element with properties (thread-safe)."""
    with _gst_lock:
        elem = Gst.ElementFactory.make(factory, name)
        if not elem:
            raise RuntimeError(f"Cannot create element: {factory}")
        for k, v in (props or {}).items():
            elem.set_property(k.replace("-", "_"), v)
        return elem


# =============================================================================
# DeepStream Metadata
# =============================================================================

def _get_pyds():
    import pyds
    return pyds


class BatchIterator:
    """Iterator for DeepStream batch metadata."""

    def __init__(self, batch):
        self.batch = batch
        self._pyds = _get_pyds()

    def __iter__(self) -> Iterator[Tuple]:
        for frame in self.frames():
            for obj in self.objects(frame):
                yield frame, obj

    def frames(self) -> Iterator:
        l_frame = self.batch.frame_meta_list
        while l_frame:
            try:
                yield self._pyds.NvDsFrameMeta.cast(l_frame.data)
                l_frame = l_frame.next
            except StopIteration:
                break

    def objects(self, frame_meta) -> Iterator:
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                yield self._pyds.NvDsObjectMeta.cast(l_obj.data)
                l_obj = l_obj.next
            except StopIteration:
                break


def extract_embedding(obj_meta, dim: int = 512) -> Optional[np.ndarray]:
    """Extract L2-normalized embedding from SGIE tensor output."""
    pyds = _get_pyds()
    l_user = obj_meta.obj_user_meta_list
    while l_user:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tensor = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tensor, 0)
                if layer:
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    emb = np.ctypeslib.as_array(ptr, shape=(dim,)).copy()
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
            l_user = l_user.next
        except StopIteration:
            break
    return None


def get_batch_meta(buffer):
    """Get NvDsBatchMeta from GstBuffer."""
    if not buffer:
        return None
    return _get_pyds().gst_buffer_get_nvds_batch_meta(hash(buffer))


# =============================================================================
# FPS Monitor
# =============================================================================

class FPSMonitor:
    """FPS monitor with stats callback support."""

    def __init__(self, name: str, log_interval: float = 1.0,
                 stats_interval: float = 10.0, stats_callback: Optional[Callable[[], dict]] = None):
        self._name = name
        self._log_interval = log_interval
        self._stats_interval = stats_interval
        self._stats_callback = stats_callback
        self._fps_count = 0
        self._fps_start = time.time()
        self._stats_last = time.time()

    def on_frame(self) -> None:
        self._fps_count += 1

    def should_log(self) -> bool:
        return time.time() - self._fps_start >= self._log_interval

    def should_stats(self) -> bool:
        return time.time() - self._stats_last >= self._stats_interval

    def log(self) -> float:
        elapsed = time.time() - self._fps_start
        fps = self._fps_count / elapsed if elapsed > 0 else 0
        print(f"[{self._name} FPS] {fps:.1f}")
        self._fps_start = time.time()
        self._fps_count = 0
        return fps

    def log_stats(self) -> Optional[dict]:
        if self._stats_callback:
            stats = self._stats_callback()
            print(f"[{self._name} STATS] " + ", ".join(f"{k}={v}" for k, v in stats.items()))
            self._stats_last = time.time()
            return stats
        return None

    def reset(self) -> None:
        self._fps_count = 0
        self._fps_start = time.time()
        self._stats_last = time.time()

    @property
    def current_fps(self) -> float:
        elapsed = time.time() - self._fps_start
        return self._fps_count / elapsed if elapsed > 0 else 0


def fps_probe_factory(name: str, log_interval: float = 1.0, stats_interval: float = 10.0,
                      stats_callback: Optional[Callable[[], dict]] = None) -> Callable:
    """Create FPS probe callback."""
    monitor = FPSMonitor(name, log_interval, stats_interval, stats_callback)

    def fps_probe(pad, info, user_data) -> Gst.PadProbeReturn:
        buffer = info.get_buffer()
        if not buffer or not get_batch_meta(buffer):
            return Gst.PadProbeReturn.OK
        monitor.on_frame()
        if monitor.should_log():
            monitor.log()
        if monitor.should_stats():
            monitor.log_stats()
        return Gst.PadProbeReturn.OK

    return fps_probe


class IntervalRunner:
    """Run callback at fixed interval using GLib timeout."""

    def __init__(self, interval_ms: int, callback: Callable[[int], None]):
        self._interval_ms = interval_ms
        self._callback = callback
        self._source_id: Optional[int] = None
        self._count = 0

    def start(self) -> None:
        if self._source_id is None:
            self._source_id = GLib.timeout_add(self._interval_ms, self._run)

    def stop(self) -> None:
        if self._source_id is not None:
            GLib.source_remove(self._source_id)
            self._source_id = None

    def _run(self) -> bool:
        self._count += 1
        self._callback(self._count)
        return True

    @property
    def is_running(self) -> bool:
        return self._source_id is not None
