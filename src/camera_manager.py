"""Camera manager for multi-branch DeepStream pipeline."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable
from urllib.parse import parse_qs, urlparse

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.pipeline_builder import BranchInfo
from src.source_mapper import SourceIDMapper
from src.common import make_element, get_nvvidconv_props

logger = logging.getLogger(__name__)

# State transition timeout (5 seconds per state)
STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


@dataclass
class CameraInfo:
    """Camera runtime information."""
    bin: Gst.Bin
    tee: Gst.Element
    source_id: int
    uri: str
    branch_pads: dict[str, Gst.Pad]
    is_file: bool = False


class MultibranchCameraManager:
    """Add/remove cameras to multiple branches at runtime."""

    def __init__(self, pipeline: Gst.Pipeline, branches: dict[str, BranchInfo], gpu_id: int = 0):
        self.pipeline = pipeline
        self.branches = branches
        self._gpu_id = gpu_id
        self._cameras: dict[str, CameraInfo] = {}
        self._mapper = SourceIDMapper()
        self._lock = threading.Lock()
        self._last_op = 0.0
        self._stream_publisher = None

    def set_stream_publisher(self, publisher) -> None:
        """Set the stream publisher for cleanup on camera removal."""
        self._stream_publisher = publisher

    # ─────────────────────────────────────────────────────────────────────────
    # State Management
    # ─────────────────────────────────────────────────────────────────────────

    def _delay(self):
        """Wait 2s between operations."""
        wait = 2.0 - (time.time() - self._last_op)
        if wait > 0:
            time.sleep(wait)

    def _incremental_state_sync(self, element: Gst.Element, target_state: Gst.State) -> bool:
        """Sync element state incrementally: NULL → READY → PAUSED → PLAYING."""
        name = element.get_name()
        element.set_state(Gst.State.NULL)

        states = [(Gst.State.READY, "READY"), (Gst.State.PAUSED, "PAUSED"), (Gst.State.PLAYING, "PLAYING")]
        for state, state_name in states:
            if target_state >= state:
                element.set_state(state)
                ret, _, _ = element.get_state(STATE_CHANGE_TIMEOUT)
                if ret == Gst.StateChangeReturn.FAILURE:
                    logger.error(f"[CAM] Failed to set {name} to {state_name}")
                    return False
                logger.debug(f"[CAM] {name} set to {state_name}")
        return True

    def _resume_pipeline(self) -> bool:
        """Resume pipeline to PLAYING state."""
        logger.info("[CAM] Resuming pipeline to PLAYING")
        self.pipeline.set_state(Gst.State.PLAYING)
        ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.warning("[CAM] Pipeline resume returned FAILURE (may still work)")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Source Creation
    # ─────────────────────────────────────────────────────────────────────────

    def _create_pad_callback(self, tee: Gst.Element, camera_id: str,
                              linked_state: dict, event: Optional[threading.Event] = None,
                              video_prefix: str = "video") -> Callable:
        """Create unified pad-added callback for source elements."""
        lock = threading.Lock()

        def on_pad_added(_source, pad):
            with lock:
                if linked_state["done"]:
                    return
                pad_name = pad.get_name()

                # Check if this is a video pad
                is_video = pad_name.startswith("vsrc_")  # nvurisrcbin
                if not is_video:
                    caps = pad.get_current_caps() or pad.query_caps(None)
                    if caps:
                        struct = caps.get_structure(0)
                        is_video = struct and struct.get_name().startswith(video_prefix)

                if is_video:
                    sink = tee.get_static_pad("sink")
                    if sink and not sink.is_linked():
                        ret = pad.link(sink)
                        if ret == Gst.PadLinkReturn.OK:
                            linked_state["done"] = True
                            if event:
                                event.set()
                            logger.info(f"[CAM] Source pad linked to tee for {camera_id}")
                        else:
                            logger.warning(f"[CAM] Failed to link pad to tee: {ret}")

        return on_pad_added

    def _create_rtsp_source(self, camera_id: str, uri: str, source_id: int,
                            bin_elem: Gst.Bin, tee: Gst.Element, linked_state: dict) -> Gst.Element:
        """Create nvurisrcbin source for RTSP streams."""
        source = make_element("nvurisrcbin", f"nvurisrc_{camera_id}", {
            "uri": uri,
            "gpu-id": self._gpu_id,
            "disable-audio": True,
            "source-id": source_id,
            "cudadec-memtype": 0,
            "num-extra-surfaces": 0,
            "latency": 100,
            "drop-frame-interval": 0
        })
        bin_elem.add(source)

        callback = self._create_pad_callback(tee, camera_id, linked_state, video_prefix="vsrc_")
        source.connect("pad-added", callback)
        logger.info(f"[CAM] Created nvurisrcbin for RTSP: {camera_id}")
        return source

    def _create_file_source(self, camera_id: str, uri: str, bin_elem: Gst.Bin,
                            tee: Gst.Element, linked_state: dict) -> tuple[Gst.Element, threading.Event]:
        """Create uridecodebin source for file/HTTP sources."""
        source = make_element("uridecodebin", f"uridecodebin_{camera_id}", {"uri": uri})
        bin_elem.add(source)

        pad_linked_event = threading.Event()
        callback = self._create_pad_callback(tee, camera_id, linked_state, pad_linked_event)
        source.connect("pad-added", callback)
        logger.info(f"[CAM] Created uridecodebin for file: {camera_id}")
        return source, pad_linked_event

    def _is_v4l2_uri(self, uri: str) -> bool:
        """Detect V4L2 URI or direct /dev/video path."""
        return uri.startswith("v4l2://") or uri.startswith("/dev/video")

    def _normalize_v4l2_device(self, raw: str) -> str:
        """Normalize v4l2 device notation to /dev/videoN style."""
        value = (raw or "").strip()
        if not value:
            return "/dev/video0"
        if value.isdigit():
            return f"/dev/video{value}"
        if value.startswith("/dev/"):
            return value

        value = value.lstrip("/")
        if value.startswith("dev/"):
            return f"/{value}"
        if value.startswith("video"):
            return f"/dev/{value}"
        return f"/{value}"

    def _parse_v4l2_uri(self, uri: str) -> dict:
        """
        Parse V4L2 URI.

        Supported examples:
        - v4l2:///dev/video0?width=640&height=480&fps=30&format=YUYV
        - v4l2://dev/video0?format=MJPG
        - v4l2://0
        - /dev/video0
        """
        defaults = {"width": 640, "height": 480, "fps": 30, "io_mode": 2, "format": "YUYV"}

        if uri.startswith("/dev/video"):
            device = uri
            params = {}
        else:
            parsed = urlparse(uri)
            if parsed.scheme != "v4l2":
                raise ValueError(f"Unsupported V4L2 URI: {uri}")
            params = parse_qs(parsed.query)

            raw_device = params.get("device", [None])[0]
            if not raw_device:
                if parsed.netloc and parsed.path:
                    raw_device = f"{parsed.netloc}{parsed.path}"
                elif parsed.path and parsed.path != "/":
                    raw_device = parsed.path
                else:
                    raw_device = parsed.netloc

            device = self._normalize_v4l2_device(raw_device or "/dev/video0")

        if not device.startswith("/dev/video"):
            raise ValueError(
                f"Invalid V4L2 device '{device}'. Use /dev/videoN (example: /dev/video0)."
            )

        def get_int(key: str, default: int) -> int:
            val = params.get(key, [str(default)])[0]
            try:
                parsed_val = int(val)
            except ValueError as e:
                raise ValueError(f"Invalid V4L2 parameter {key}={val}") from e
            if parsed_val <= 0:
                raise ValueError(f"V4L2 parameter {key} must be > 0")
            return parsed_val

        width = get_int("width", defaults["width"])
        height = get_int("height", defaults["height"])
        framerate = get_int("framerate", defaults["fps"])
        fps = get_int("fps", framerate)
        io_mode_dash = get_int("io-mode", defaults["io_mode"])
        io_mode = get_int("io_mode", io_mode_dash)
        fmt = params.get("format", params.get("pixfmt", [defaults["format"]]))[0].upper()

        if fmt in {"YUYV", "YUY2"}:
            input_caps = f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1"
            is_mjpeg = False
        elif fmt in {"MJPG", "MJPEG", "JPEG"}:
            input_caps = f"image/jpeg,width={width},height={height},framerate={fps}/1"
            is_mjpeg = True
        else:
            raise ValueError(
                f"Unsupported V4L2 format '{fmt}'. Supported: YUYV, MJPG"
            )

        return {
            "device": device,
            "width": width,
            "height": height,
            "fps": fps,
            "io_mode": io_mode,
            "format": fmt,
            "input_caps": input_caps,
            "is_mjpeg": is_mjpeg,
        }

    def _create_v4l2_source(self, camera_id: str, uri: str, bin_elem: Gst.Bin,
                            tee: Gst.Element, linked_state: dict) -> None:
        """Create USB camera source using v4l2src -> convert -> NVMM NV12."""
        cfg = self._parse_v4l2_uri(uri)

        source = make_element("v4l2src", f"v4l2src_{camera_id}", {
            "device": cfg["device"],
            "do-timestamp": True,
            "io-mode": cfg["io_mode"],
        })
        in_caps = make_element("capsfilter", f"v4l2caps_{camera_id}", {
            "caps": Gst.Caps.from_string(cfg["input_caps"])
        })
        nvconv = make_element("nvvideoconvert", f"v4l2nvconv_{camera_id}", get_nvvidconv_props())
        out_caps = make_element("capsfilter", f"v4l2outcaps_{camera_id}", {
            "caps": Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
        })

        elements = [source, in_caps]
        if cfg["is_mjpeg"]:
            jpegdec = make_element("jpegdec", f"jpegdec_{camera_id}")
            elements.append(jpegdec)
        else:
            swconv = make_element("videoconvert", f"videoconvert_{camera_id}")
            elements.append(swconv)
        elements.extend([nvconv, out_caps])

        for elem in elements:
            bin_elem.add(elem)

        for src, dst in zip(elements, elements[1:]):
            if not src.link(dst):
                raise RuntimeError(f"Failed to link {src.get_name()} -> {dst.get_name()} for {camera_id}")

        src_pad = elements[-1].get_static_pad("src")
        sink_pad = tee.get_static_pad("sink")
        if not src_pad or not sink_pad:
            raise RuntimeError(f"Cannot get source/tee pad for V4L2 camera {camera_id}")

        ret = src_pad.link(sink_pad)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link V4L2 source to tee for {camera_id}: {ret}")

        linked_state["done"] = True
        logger.info(
            f"[CAM] Created V4L2 source: {camera_id} ({cfg['device']} {cfg['width']}x{cfg['height']}@{cfg['fps']} {cfg['format']})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Branch Linking
    # ─────────────────────────────────────────────────────────────────────────

    def _link_branch(self, bin_elem: Gst.Bin, tee: Gst.Element, camera_id: str,
                     source_id: int, branch_name: str, sync: bool = False) -> Gst.Pad:
        """Link: tee -> queue -> mux."""
        b = self.branches[branch_name]

        q = make_element("queue", None, {
            "max-size-buffers": 30,
            "max-size-bytes": 0,
            "max-size-time": 0,
            "leaky": 2
        })
        bin_elem.add(q)

        tee_src = tee.request_pad_simple("src_%u")
        tee_src.link(q.get_static_pad("sink"))

        mux_sink = b.nvstreammux.request_pad_simple(f"sink_{source_id}")
        ghost = Gst.GhostPad.new(f"g_{branch_name}_{camera_id}", q.get_static_pad("src"))
        bin_elem.add_pad(ghost)
        ghost.link(mux_sink)

        if sync:
            _, parent_state, _ = bin_elem.get_state(0)
            self._incremental_state_sync(q, parent_state)

        return tee_src

    def _unlink_branch(self, cam: CameraInfo, branch_name: str, camera_id: str,
                       use_blocking: bool = False) -> None:
        """Unlink camera from branch with optional blocking probe."""
        b = self.branches.get(branch_name)
        if not b:
            return

        tee_pad = cam.branch_pads.get(branch_name)
        probe_id = None

        # Optional blocking probe for safe unlink during PLAYING
        if use_blocking and tee_pad:
            blocked = threading.Event()
            probe_id = tee_pad.add_probe(
                Gst.PadProbeType.BLOCK_DOWNSTREAM,
                lambda p, i: (blocked.set(), Gst.PadProbeReturn.OK)[1]
            )
            blocked.wait(timeout=1.0)

        # Unlink ghost pad from mux
        ghost_pad, mux_pad = None, None
        it = cam.bin.iterate_pads()
        while True:
            ret, pad = it.next()
            if ret == Gst.IteratorResult.OK and pad.get_name() == f"g_{branch_name}_{camera_id}":
                ghost_pad = pad
                mux_pad = pad.get_peer()
                if mux_pad:
                    pad.unlink(mux_pad)
                break
            elif ret == Gst.IteratorResult.RESYNC:
                it.resync()
            elif ret != Gst.IteratorResult.OK:
                break

        # Unlink tee from queue (get queue via tee_pad peer)
        q = None
        if tee_pad:
            q_sink = tee_pad.get_peer()
            if q_sink:
                q = q_sink.get_parent()
                tee_pad.unlink(q_sink)

        # Remove probe
        if probe_id and tee_pad:
            tee_pad.remove_probe(probe_id)

        time.sleep(0.1)

        # Cleanup elements
        if q:
            q.set_state(Gst.State.NULL)
            q.get_state(Gst.CLOCK_TIME_NONE)
            time.sleep(0.05)
            cam.bin.remove(q)

        if ghost_pad:
            cam.bin.remove_pad(ghost_pad)

        if mux_pad:
            try:
                b.nvstreammux.release_request_pad(mux_pad)
            except Exception as e:
                logger.warning(f"release mux pad failed: {e}")

        if tee_pad:
            try:
                cam.tee.release_request_pad(tee_pad)
            except Exception as e:
                logger.warning(f"release tee pad failed: {e}")

        cam.branch_pads.pop(branch_name, None)
        logger.info(f"[CAM] Unlinked {camera_id} from branch {branch_name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Camera Operations
    # ─────────────────────────────────────────────────────────────────────────

    def add_camera(self, camera_id: str, uri: str, branch_name: str) -> bool:
        """Add camera to pipeline."""
        with self._lock:
            self._delay()

            # Camera already exists - must remove first
            if camera_id in self._cameras:
                logger.warning(f"[CAM] Camera {camera_id} already exists")
                return False

            # Create new camera
            return self._create_new_camera(camera_id, uri, branch_name)

    def _create_new_camera(self, camera_id: str, uri: str, branch_name: str) -> bool:
        """Create and add new camera to pipeline."""
        _, prev_state, _ = self.pipeline.get_state(0)
        is_rtsp = uri.startswith("rtsp://") or uri.startswith("rtsps://") 
        is_v4l2 = self._is_v4l2_uri(uri)
        is_file = uri.startswith("file://")

        logger.info(f"[CAM] Adding {camera_id} - state: {prev_state.value_nick}")

        try:
            # Create camera bin
            source_id = self._mapper.add(camera_id, uri)
            bin_elem = Gst.Bin.new(f"cam_{camera_id}")
            tee = make_element("tee", f"tee_{camera_id}", {"allow-not-linked": True})
            bin_elem.add(tee)

            linked_state = {"done": False}
            pad_linked_event = None

            # Create source
            if is_rtsp or is_file:
                self._create_rtsp_source(camera_id, uri, source_id, bin_elem, tee, linked_state)
            elif is_v4l2:
                self._create_v4l2_source(camera_id, uri, bin_elem, tee, linked_state)
     
            self.pipeline.add(bin_elem)

            # Link branch
            branch_pads = {}
            pad = self._link_branch(bin_elem, tee, camera_id, source_id, branch_name, sync=False)
            branch_pads[branch_name] = pad

            # Wait for decodebin dynamic pad linking
            if pad_linked_event:
                logger.info("[CAM] Waiting for uridecodebin pad linking...")
                pad_linked_event.wait(timeout=5.0)
                time.sleep(0.5)

            # Sync camera bin to PLAYING
            self._incremental_state_sync(bin_elem, Gst.State.PLAYING)

            # Always ensure pipeline is PLAYING when we have cameras
            if prev_state != Gst.State.PLAYING:
                logger.info(f"[CAM] Pipeline was in {prev_state.value_nick}, resuming to PLAYING")
                self._resume_pipeline()

            time.sleep(2.0)

            # Store camera
            self._cameras[camera_id] = CameraInfo(
                bin=bin_elem, tee=tee, source_id=source_id,
                uri=uri, branch_pads=branch_pads, is_file=is_file
            )

            self._last_op = time.time()
            logger.info(f"[CAM] Added {camera_id} to branch: {branch_name}")
            return True

        except Exception as e:
            logger.error(f"add_camera failed: {e}", exc_info=True)
            self._cleanup_failed_add(camera_id, bin_elem)
            return False

    def _cleanup_failed_add(self, camera_id: str, bin_elem: Gst.Bin) -> None:
        """Cleanup after failed add_camera."""
        try:
            if camera_id in self._cameras:
                del self._cameras[camera_id]
            self.pipeline.remove(bin_elem)
        except:
            pass
        self._mapper.remove(camera_id)

    def remove_camera(self, camera_id: str) -> bool:
        """Remove camera from pipeline with full cleanup."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam:
                return False

            try:
                logger.info(f"[CAM] Removing {camera_id}...")

                # Cleanup active streams
                if self._stream_publisher:
                    self._stream_publisher.cleanup_camera(camera_id)

                # Always perform full removal
                self._remove_camera_full(cam, camera_id)

                self._last_op = time.time()
                logger.info(f"[CAM] Removed {camera_id}")
                return True

            except Exception as e:
                logger.error(f"remove_camera failed: {e}", exc_info=True)
                return False

    def _remove_camera_full(self, cam: CameraInfo, camera_id: str) -> None:
        """Fully remove camera from pipeline."""
        # Block data flow
        for pad in cam.branch_pads.values():
            pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, lambda *_: Gst.PadProbeReturn.REMOVE)
        time.sleep(0.2)

        # Unlink from all branches
        for b in list(cam.branch_pads.keys()):
            self._unlink_branch(cam, b, camera_id)

        # Set to NULL and remove
        cam.bin.set_state(Gst.State.NULL)
        cam.bin.get_state(STATE_CHANGE_TIMEOUT)
        self.pipeline.remove(cam.bin)

        # Cleanup tracking
        self._mapper.remove(camera_id)
        del self._cameras[camera_id]

        # Wait for GPU decoder cleanup
        time.sleep(3.0)

    def add_camera_to_branch(self, camera_id: str, branch_name: str) -> bool:
        """Add camera to additional branch dynamically."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam or branch_name in cam.branch_pads or branch_name not in self.branches:
                return False

            logger.info(f"[CAM] Adding {camera_id} to branch {branch_name}")

            try:
                pad = self._link_branch(cam.bin, cam.tee, camera_id, cam.source_id, branch_name, sync=True)
                cam.branch_pads[branch_name] = pad
                self._last_op = time.time()
                logger.info(f"[CAM] Added {camera_id} to {branch_name}")
                return True
            except Exception as e:
                logger.error(f"add_camera_to_branch failed: {e}", exc_info=True)
                return False

    def remove_camera_from_branch(self, camera_id: str, branch_name: str) -> bool:
        """
        Remove camera from branch without pausing pipeline.
        Uses blocking probe for safe unlinking even when branch becomes empty.
        """
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam or branch_name not in cam.branch_pads or len(cam.branch_pads) <= 1:
                return False

            logger.info(f"[CAM] Removing {camera_id} from {branch_name}")

            try:
                # Unlink using blocking probe - safe even if branch becomes empty
                # No need to pause pipeline as blocking probe handles data flow
                self._unlink_branch(cam, branch_name, camera_id, use_blocking=True)

                self._last_op = time.time()
                logger.info(f"[CAM] Removed {camera_id} from {branch_name}")
                return True

            except Exception as e:
                logger.error(f"remove_camera_from_branch failed: {e}", exc_info=True)
                return False

    def kill_all(self) -> int:
        """Remove all cameras - requires pipeline restart to add cameras again."""
        with self._lock:
            if not self._cameras:
                return 0

            n = len(self._cameras)
            logger.info(f"[CAM] kill_all - removing {n} cameras")

            _, current_state, _ = self.pipeline.get_state(0)

            # Send EOS to all branches
            for branch_name, branch in self.branches.items():
                if branch.nvstreammux:
                    branch.nvstreammux.send_event(Gst.Event.new_eos())
            time.sleep(2.0)

            # Block all camera tees
            block_probes = []
            for cam in self._cameras.values():
                if cam.tee:
                    sink_pad = cam.tee.get_static_pad("sink")
                    if sink_pad:
                        probe_id = sink_pad.add_probe(
                            Gst.PadProbeType.BUFFER,
                            lambda p, i: Gst.PadProbeReturn.DROP
                        )
                        block_probes.append((sink_pad, probe_id))
            time.sleep(0.5)

            # State transitions
            if current_state == Gst.State.PLAYING:
                self.pipeline.set_state(Gst.State.PAUSED)
                self.pipeline.get_state(10 * Gst.SECOND)
                time.sleep(2.0)

            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.get_state(10 * Gst.SECOND)
            time.sleep(2.0)

            # Remove probes
            for sink_pad, probe_id in block_probes:
                try:
                    sink_pad.remove_probe(probe_id)
                except:
                    pass

            # Cleanup bins
            for camera_id, cam in list(self._cameras.items()):
                try:
                    cam.bin.set_state(Gst.State.NULL)
                    self.pipeline.remove(cam.bin)
                except Exception as e:
                    logger.warning(f"[CAM] cleanup error for {camera_id}: {e}")

            self._cameras.clear()
            self._mapper.clear()

            # Back to READY
            self.pipeline.set_state(Gst.State.READY)
            self.pipeline.get_state(STATE_CHANGE_TIMEOUT)

            self._last_op = time.time()
            logger.info(f"[CAM] kill_all completed - removed {n} cameras")
            return n

    # ─────────────────────────────────────────────────────────────────────────
    # Query Methods
    # ─────────────────────────────────────────────────────────────────────────

    def list_cameras(self) -> dict:
        with self._lock:
            return {k: {"uri": v.uri, "source_id": v.source_id, "branches": list(v.branch_pads.keys())}
                    for k, v in self._cameras.items()}

    def get_camera(self, camera_id: str) -> Optional[CameraInfo]:
        with self._lock:
            return self._cameras.get(camera_id)

    def has_camera(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._cameras

    def count(self) -> int:
        with self._lock:
            return len(self._cameras)

    def get_camera_branches(self, camera_id: str) -> list[str]:
        with self._lock:
            cam = self._cameras.get(camera_id)
            return list(cam.branch_pads.keys()) if cam else []

    def get_mapper(self) -> SourceIDMapper:
        """Get SourceIDMapper for probe lookups."""
        return self._mapper
