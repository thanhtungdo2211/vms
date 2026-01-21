"""Demux RTSP Publisher - Stream individual cameras WITH annotations.

Uses nvstreamdemux BEFORE OSD to split batched frames to individual streams.
Each camera gets its own OSD element for drawing annotations.

Architecture:
    Muxer → PGIE → Tracker → SGIE → nvstreamdemux → src_0 → OSD → enc → RTSP (cam0)
                                                  → src_1 → OSD → enc → RTSP (cam1)
                                                  → src_N ...

Key: OSD is per-camera (in RTSP chain), not shared across all cameras.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

if TYPE_CHECKING:
    from src.pipeline_builder import BranchInfo
    from src.camera_manager import MultibranchCameraManager

logger = logging.getLogger(__name__)

STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


@dataclass
class DemuxPublishInfo:
    """Info about active demux RTSP publish session."""
    camera_id: str
    branch: str
    location: str
    bitrate: int
    started_at: datetime
    source_id: int
    elements: list = field(default_factory=list)
    demux_pad: Optional[Gst.Pad] = None


class DemuxRtspPublisher:
    """Manage per-camera RTSP streaming with annotations via nvstreamdemux.

    Requires pipeline to have nvstreamdemux element in the branch.
    The demux splits batched NvBufSurface (after OSD) back to individual camera frames.
    """

    def __init__(
        self,
        pipeline: Gst.Pipeline,
        branches: dict[str, "BranchInfo"],
        camera_manager: "MultibranchCameraManager"
    ):
        self.pipeline = pipeline
        self.branches = branches
        self.camera_manager = camera_manager
        self._lock = threading.Lock()
        self._counter = 0

        # Per-camera publish sessions
        # (camera_id, branch) -> DemuxPublishInfo
        self._publishers: dict[Tuple[str, str], DemuxPublishInfo] = {}

        # Cache demux elements per branch
        self._demux_cache: dict[str, Gst.Element] = {}

        # Pending link requests: (branch, pad_name) -> (queue_sink, key, elements)
        self._pending_links: dict[Tuple[str, str], tuple] = {}

        # Register pad-added handler on all demux elements upfront
        self._setup_demux_handlers()

    def _setup_demux_handlers(self):
        """Pre-request pads on all demux elements.

        nvstreamdemux requires pads to be requested while pipeline is in NULL state.
        Pre-requesting pads for max_cameras ensures they're available for dynamic cameras.

        NOTE: We do NOT register pad-added handler because it causes memory corruption
        when fired during request_pad_simple. Instead, we get pads directly.
        """
        for branch_name, branch_info in self.branches.items():
            demux = self._get_demux(branch_name)
            if not demux:
                continue

            # Pre-request pads for all possible cameras (based on max_cameras/batch_size)
            # This MUST be done while pipeline is in NULL state
            max_cams = branch_info.max_cameras
            for i in range(max_cams):
                pad_name = f"src_{i}"
                pad = demux.request_pad_simple(pad_name)
                if pad:
                    logger.info(f"[DemuxRTSP] Pre-requested {branch_name}/{pad_name}")
                else:
                    logger.warning(f"[DemuxRTSP] Failed to pre-request {branch_name}/{pad_name}")

    def reset(self):
        """Reset demux publisher state after pipeline NULL.

        Call this after pipeline goes to NULL state to clear cached demux
        elements and re-request pads when pipeline returns to READY.
        """
        with self._lock:
            logger.info("[DemuxRTSP] Resetting demux publisher state")
            # Clear all publishers (they're invalid after NULL)
            self._publishers.clear()
            # Clear demux cache (element references may be stale)
            self._demux_cache.clear()
            # Clear pending links
            self._pending_links.clear()
            # Re-request pads now that pipeline is in NULL/READY
            self._setup_demux_handlers()

    def _on_demux_pad_added(self, element, pad, branch_name):
        """Handle demux pad-added signal."""
        pad_name = pad.get_name()
        logger.info(f"[DemuxRTSP] Pad added: {branch_name}/{pad_name}")

        with self._lock:
            # Check if we have a pending link request for this pad
            pending_key = (branch_name, pad_name)
            if pending_key in self._pending_links:
                queue_sink, key, elements = self._pending_links[pending_key]

                # Link the pad to queue
                result = pad.link(queue_sink)
                if result == Gst.PadLinkReturn.OK:
                    logger.info(f"[DemuxRTSP] Linked {branch_name}/{pad_name} to RTSP chain")
                    # Update stored pad reference
                    if key in self._publishers:
                        self._publishers[key].demux_pad = pad
                else:
                    logger.error(f"[DemuxRTSP] Failed to link {pad_name}: {result}")

                del self._pending_links[pending_key]

    def _get_demux(self, branch_name: str) -> Optional[Gst.Element]:
        """Get nvstreamdemux element for branch."""
        if branch_name in self._demux_cache:
            return self._demux_cache[branch_name]

        branch = self.branches.get(branch_name)
        if not branch:
            return None

        # Find demux in branch elements
        for elem in branch.elements:
            if elem.get_factory().get_name() == "nvstreamdemux":
                self._demux_cache[branch_name] = elem
                return elem

        # Try to find by name pattern
        demux = self.pipeline.get_by_name(f"{branch_name}_demux")
        if demux:
            self._demux_cache[branch_name] = demux
            return demux

        return None

    def start_publish(
        self,
        camera_id: str,
        branch_name: str,
        location: str,
        bitrate: int = 4000000
    ) -> bool:
        """Start RTSP publishing for specific camera with annotations.

        Args:
            camera_id: Camera to stream
            branch_name: Branch with nvstreamdemux
            location: RTSP server URL
            bitrate: H264 encoding bitrate

        Returns:
            True if started successfully
        """
        if not location.startswith("rtsp://"):
            logger.error(f"[DemuxRTSP] Invalid RTSP URL: {location}")
            return False

        with self._lock:
            key = (camera_id, branch_name)

            if key in self._publishers:
                logger.warning(f"[DemuxRTSP] {camera_id}/{branch_name} already publishing")
                return True  # Already publishing is success

            # Check camera exists and get source_id
            cam = self.camera_manager.get_camera(camera_id)
            if not cam:
                logger.error(f"[DemuxRTSP] Camera {camera_id} not found")
                return False

            source_id = cam["source_id"]

            # Get demux element
            demux = self._get_demux(branch_name)
            if not demux:
                logger.error(f"[DemuxRTSP] No nvstreamdemux in branch {branch_name}")
                return False

            try:
                self._counter += 1
                prefix = f"demux_rtsp_{camera_id}_{self._counter}"

                # Create RTSP encoding chain
                elements = self._create_rtsp_chain(prefix, location, bitrate)

                # Add elements to pipeline
                for elem in elements:
                    self.pipeline.add(elem)

                # Link RTSP chain elements first
                for i in range(len(elements) - 1):
                    src_elem = elements[i]
                    dest_elem = elements[i + 1]

                    # Special handling for rtspclientsink
                    if "rtspclientsink" in dest_elem.get_factory().get_name():
                        sink_pad = dest_elem.request_pad_simple("sink_%u")
                        if not sink_pad:
                            raise RuntimeError("Failed to get rtspclientsink request pad")

                        src_pad = src_elem.get_static_pad("src")
                        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                            raise RuntimeError(f"Failed to link to rtspclientsink")
                    else:
                        if not src_elem.link(dest_elem):
                            raise RuntimeError(f"Failed to link {src_elem.get_name()}")

                # Get demux src pad
                pad_name = f"src_{source_id}"

                # First try to get existing pad by iterating src pads
                demux_src = None
                it = demux.iterate_src_pads()
                while True:
                    result, pad = it.next()
                    if result == Gst.IteratorResult.DONE:
                        break
                    if result == Gst.IteratorResult.OK and pad.get_name() == pad_name:
                        demux_src = pad
                        logger.info(f"[DemuxRTSP] Found existing pad {pad_name}")
                        break

                # If pad not found, request it (may have been released previously)
                if not demux_src:
                    demux_src = demux.request_pad_simple(pad_name)
                    if demux_src:
                        logger.info(f"[DemuxRTSP] Requested pad {pad_name}")
                    else:
                        logger.warning(f"[DemuxRTSP] Failed to request pad {pad_name}")

                # Get queue sink pad for linking
                queue = elements[0]
                queue_sink = queue.get_static_pad("sink")

                if not demux_src:
                    raise RuntimeError(f"Could not get or request pad {pad_name}")

                # === CRITICAL: State transition BEFORE linking to demux ===
                # Elements must be at least PAUSED before receiving data from
                # the demux pad which is already in PLAYING state.
                # This prevents data from being dropped during state transitions.

                _, pipeline_state, _ = self.pipeline.get_state(0)
                logger.info(f"[DemuxRTSP] Pipeline state: {pipeline_state}, transitioning elements...")

                # Step 1: Transition all elements to READY
                for elem in elements:
                    ret = elem.set_state(Gst.State.READY)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.warning(f"[DemuxRTSP] {elem.get_name()} failed to go READY")
                    elem.get_state(STATE_CHANGE_TIMEOUT)

                time.sleep(0.1)

                # Step 2: Transition all elements to PAUSED (allows caps negotiation)
                for elem in elements:
                    ret = elem.set_state(Gst.State.PAUSED)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.warning(f"[DemuxRTSP] {elem.get_name()} failed to go PAUSED")
                    # Don't wait for async state change here - PAUSED may block
                    elem.get_state(Gst.CLOCK_TIME_NONE)  # Non-blocking check

                time.sleep(0.2)

                # Step 3: NOW link to demux (elements are ready to receive data)
                link_result = demux_src.link(queue_sink)
                if link_result != Gst.PadLinkReturn.OK:
                    raise RuntimeError(f"Failed to link demux pad {pad_name} to queue: {link_result}")
                logger.info(f"[DemuxRTSP] Linked {pad_name} to RTSP chain")

                # Step 4: Transition to PLAYING (data will now flow)
                time.sleep(0.2)

                if pipeline_state == Gst.State.PLAYING:
                    for elem in elements:
                        ret = elem.set_state(Gst.State.PLAYING)
                        if ret == Gst.StateChangeReturn.FAILURE:
                            logger.warning(f"[DemuxRTSP] {elem.get_name()} failed to go PLAYING")
                        elem.get_state(STATE_CHANGE_TIMEOUT)

                # Brief delay for elements to stabilize
                time.sleep(0.3)

                # Store publish info
                self._publishers[key] = DemuxPublishInfo(
                    camera_id=camera_id,
                    branch=branch_name,
                    location=location,
                    bitrate=bitrate,
                    started_at=datetime.now(),
                    source_id=source_id,
                    elements=elements,
                    demux_pad=demux_src
                )

                logger.info(f"[DemuxRTSP] Started {camera_id}/{branch_name} (src_{source_id}) → {location}")
                return True

            except Exception as e:
                logger.error(f"[DemuxRTSP] Failed to start publish: {e}", exc_info=True)

                # Cleanup
                if 'elements' in locals():
                    for elem in elements:
                        try:
                            elem.set_state(Gst.State.NULL)
                            self.pipeline.remove(elem)
                        except:
                            pass
                return False

    def stop_publish(self, camera_id: str, branch_name: str) -> bool:
        """Stop RTSP publishing for specific camera.

        Uses blocking probe pattern (like camera_manager.py) to safely stop
        data flow before unlinking. This prevents race conditions that could
        affect other cameras sharing the same nvstreamdemux.

        Args:
            camera_id: Camera to stop streaming
            branch_name: Branch identifier

        Returns:
            True if stopped successfully
        """
        with self._lock:
            key = (camera_id, branch_name)
            pub = self._publishers.get(key)

            if not pub:
                logger.warning(f"[DemuxRTSP] {camera_id}/{branch_name} not publishing")
                return False

            try:
                demux_pad = pub.demux_pad
                elements = pub.elements
                probe_id = None

                # Step 1: Block data flow on demux pad BEFORE unlinking
                # This is critical - prevents race conditions with other cameras
                if demux_pad and elements:
                    blocked = threading.Event()

                    def block_probe(pad, info):
                        blocked.set()
                        return Gst.PadProbeReturn.OK  # Keep blocking

                    logger.debug(f"[DemuxRTSP] Adding block probe for {camera_id}/{branch_name}")
                    probe_id = demux_pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, block_probe)

                    # Wait for probe to trigger (data flow blocked)
                    if not blocked.wait(timeout=2.0):
                        logger.warning(f"[DemuxRTSP] Block probe timeout for {camera_id}/{branch_name}")
                    else:
                        logger.debug(f"[DemuxRTSP] Data flow blocked for {camera_id}/{branch_name}")

                # Step 2: Unlink demux from queue (safe now that data is blocked)
                if demux_pad and elements:
                    queue = elements[0]
                    queue_sink = queue.get_static_pad("sink")
                    if queue_sink and queue_sink.is_linked():
                        demux_pad.unlink(queue_sink)
                        logger.info(f"[DemuxRTSP] Unlinked demux pad for {camera_id}/{branch_name}")

                # Step 3: Remove blocking probe (allow demux to continue for other cameras)
                if demux_pad and probe_id is not None:
                    demux_pad.remove_probe(probe_id)
                    logger.debug(f"[DemuxRTSP] Removed block probe for {camera_id}/{branch_name}")

                # Small delay to let any in-flight data in the chain clear
                time.sleep(0.1)

                # Step 4: Set elements to NULL and remove from pipeline
                for elem in reversed(elements):
                    elem.set_state(Gst.State.NULL)
                    elem.get_state(STATE_CHANGE_TIMEOUT)
                    self.pipeline.remove(elem)

                del self._publishers[key]
                logger.info(f"[DemuxRTSP] Stopped {camera_id}/{branch_name}")
                return True

            except Exception as e:
                logger.error(f"[DemuxRTSP] Failed to stop publish: {e}", exc_info=True)
                # Cleanup probe if still attached
                if probe_id is not None and pub.demux_pad:
                    try:
                        pub.demux_pad.remove_probe(probe_id)
                    except:
                        pass
                return False

    def get_status(
        self,
        camera_id: Optional[str] = None,
        branch_name: Optional[str] = None
    ) -> dict:
        """Get demux RTSP publishing status.

        Args:
            camera_id: Specific camera or None for all
            branch_name: Specific branch or None for all

        Returns:
            Status dict with publishing info
        """
        with self._lock:
            if camera_id and branch_name:
                pub = self._publishers.get((camera_id, branch_name))
                if pub:
                    return {
                        "publishing": True,
                        "annotated": True,
                        "location": pub.location,
                        "bitrate": pub.bitrate,
                        "source_id": pub.source_id,
                        "started_at": pub.started_at.isoformat()
                    }
                return {"publishing": False}

            # Return all (grouped by camera)
            result = {}
            for (cam_id, branch), pub in self._publishers.items():
                if camera_id and cam_id != camera_id:
                    continue
                if branch_name and branch != branch_name:
                    continue

                if cam_id not in result:
                    result[cam_id] = {}

                result[cam_id][branch] = {
                    "publishing": True,
                    "annotated": True,
                    "location": pub.location,
                    "bitrate": pub.bitrate,
                    "source_id": pub.source_id,
                    "started_at": pub.started_at.isoformat()
                }

            return result

    def _create_rtsp_chain(self, prefix: str, location: str, bitrate: int) -> list:
        """Create RTSP sink element chain with OSD for per-camera streaming."""
        elements = []

        # Queue for buffering - larger buffer to handle timing variations
        queue = Gst.ElementFactory.make("queue", f"{prefix}_q")
        queue.set_property("max-size-buffers", 60)  # Increased from 30
        queue.set_property("max-size-time", 2 * Gst.SECOND)  # 2 second buffer
        queue.set_property("leaky", 2)  # Drop old buffers
        elements.append(queue)

        # OSD - draw annotations on this camera's frames
        osd = Gst.ElementFactory.make("nvdsosd", f"{prefix}_osd")
        if osd:
            osd.set_property("process-mode", 1)  # GPU mode for better performance
            osd.set_property("display-text", 1)
            elements.append(osd)
        else:
            logger.warning(f"[DemuxRTSP] nvdsosd not available, skipping OSD")

        # Video converter - output to system memory for x264enc
        conv = Gst.ElementFactory.make("nvvideoconvert", f"{prefix}_conv")
        if not conv:
            conv = Gst.ElementFactory.make("videoconvert", f"{prefix}_conv")
        else:
            conv.set_property("compute-hw", 1)
            conv.set_property("nvbuf-memory-type", 3)
        elements.append(conv)

        # Caps filter - x264enc needs system memory (I420), not NVMM
        caps = Gst.ElementFactory.make("capsfilter", f"{prefix}_caps")
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))
        elements.append(caps)

        # H264 encoder - use x264enc for stability (works with system memory)
        enc = Gst.ElementFactory.make("x264enc", f"{prefix}_enc")
        enc.set_property("bitrate", bitrate // 1000)
        enc.set_property("speed-preset", "ultrafast")
        enc.set_property("tune", "zerolatency")
        enc.set_property("threads", 4)  # Use 4 threads for encoding
        enc.set_property("bframes", 0)  # No B-frames for low latency
        enc.set_property("key-int-max", 30)
        elements.append(enc)

        # H264 parser
        parse = Gst.ElementFactory.make("h264parse", f"{prefix}_parse")
        parse.set_property("config-interval", -1)
        elements.append(parse)

        # RTSP client sink with stability settings
        sink = Gst.ElementFactory.make("rtspclientsink", f"{prefix}_sink")
        sink.set_property("location", location)
        sink.set_property("protocols", 4)  # TCP - more reliable than UDP
        sink.set_property("latency", 200)  # 200ms latency for stability
        sink.set_property("do-rtsp-keep-alive", True)
        sink.set_property("timeout", 5000000)  # 5 second timeout (microseconds)
        elements.append(sink)

        return elements

    def cleanup_camera(self, camera_id: str) -> None:
        """Cleanup when camera is removed. Called by CameraManager."""
        keys_to_remove = [(cid, br) for cid, br in self._publishers if cid == camera_id]
        for key in keys_to_remove:
            try:
                self.stop_publish(key[0], key[1])
            except Exception as e:
                logger.warning(f"[DemuxRTSP] Cleanup error for {key}: {e}")
