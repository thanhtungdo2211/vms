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
        """Register pad-added handlers and pre-request pads on all demux elements.

        nvstreamdemux requires pads to be requested while pipeline is in NULL state.
        Pre-requesting pads for max_cameras ensures they're available for dynamic cameras.
        """
        for branch_name, branch_info in self.branches.items():
            demux = self._get_demux(branch_name)
            if not demux:
                continue

            # Register pad-added handler
            demux.connect("pad-added", self._on_demux_pad_added, branch_name)
            logger.info(f"[DemuxRTSP] Registered pad-added handler for {branch_name}")

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
                return False

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

                # Get pipeline state
                _, prev_state, _ = self.pipeline.get_state(0)

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

                # First try to get existing pad (might already be created dynamically)
                demux_src = demux.get_static_pad(pad_name)

                # Also check if pad exists as a dynamic pad by iterating
                if not demux_src:
                    it = demux.iterate_src_pads()
                    while True:
                        result, pad = it.next()
                        if result == Gst.IteratorResult.DONE:
                            break
                        if result == Gst.IteratorResult.OK and pad.get_name() == pad_name:
                            demux_src = pad
                            logger.info(f"[DemuxRTSP] Found existing dynamic pad {pad_name}")
                            break

                # Link queue sink to demux src if we have the pad
                queue = elements[0]
                queue_sink = queue.get_static_pad("sink")

                if demux_src:
                    # Pad exists - link directly
                    if demux_src.link(queue_sink) != Gst.PadLinkReturn.OK:
                        raise RuntimeError(f"Failed to link demux to queue")
                    logger.info(f"[DemuxRTSP] Linked existing pad {pad_name} to RTSP chain")
                else:
                    # Pad doesn't exist yet - store pending link request
                    # The global pad-added handler will link when pad appears
                    pending_key = (branch_name, pad_name)
                    self._pending_links[pending_key] = (queue_sink, key, elements)
                    logger.info(f"[DemuxRTSP] Waiting for pad {pad_name} (pending link registered)")

                # Sync element states
                for elem in elements:
                    elem.sync_state_with_parent()

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

                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                return False

    def stop_publish(self, camera_id: str, branch_name: str) -> bool:
        """Stop RTSP publishing for specific camera.

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

            demux = self._get_demux(branch_name)

            try:
                # Get pipeline state
                _, prev_state, _ = self.pipeline.get_state(0)

                # Pause pipeline
                if prev_state == Gst.State.PLAYING:
                    logger.info(f"[DemuxRTSP] Pausing pipeline for RTSP teardown")
                    self.pipeline.set_state(Gst.State.PAUSED)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    time.sleep(0.3)

                # Unlink demux from queue
                if pub.demux_pad and pub.elements:
                    queue = pub.elements[0]
                    queue_sink = queue.get_static_pad("sink")
                    if queue_sink and queue_sink.is_linked():
                        pub.demux_pad.unlink(queue_sink)

                # Release demux pad (if it was a request pad)
                # Note: nvstreamdemux pads are sometimes-pads, check if releasable
                if pub.demux_pad and demux:
                    try:
                        demux.release_request_pad(pub.demux_pad)
                    except:
                        pass  # May be static pad

                # Set elements to NULL and remove
                for elem in reversed(pub.elements):
                    elem.set_state(Gst.State.NULL)
                    elem.get_state(Gst.CLOCK_TIME_NONE)
                    self.pipeline.remove(elem)

                # Resume pipeline
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)

                del self._publishers[key]
                logger.info(f"[DemuxRTSP] Stopped {camera_id}/{branch_name}")
                return True

            except Exception as e:
                logger.error(f"[DemuxRTSP] Failed to stop publish: {e}", exc_info=True)
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
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

        # Queue for buffering
        queue = Gst.ElementFactory.make("queue", f"{prefix}_q")
        queue.set_property("max-size-buffers", 30)
        queue.set_property("leaky", 2)
        elements.append(queue)

        # OSD - draw annotations on this camera's frames
        osd = Gst.ElementFactory.make("nvdsosd", f"{prefix}_osd")
        if osd:
            osd.set_property("process-mode", 0)  # CPU mode
            osd.set_property("display-text", 1)
            elements.append(osd)
        else:
            logger.warning(f"[DemuxRTSP] nvdsosd not available, skipping OSD")

        # Video converter
        conv = Gst.ElementFactory.make("nvvideoconvert", f"{prefix}_conv")
        if not conv:
            conv = Gst.ElementFactory.make("videoconvert", f"{prefix}_conv")
        else:
            conv.set_property("compute-hw", 1)
            conv.set_property("nvbuf-memory-type", 3)
        elements.append(conv)

        # Caps filter
        caps = Gst.ElementFactory.make("capsfilter", f"{prefix}_caps")
        if conv.get_factory().get_name() == "nvvideoconvert":
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
        else:
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))
        elements.append(caps)

        # H264 encoder
        # enc = Gst.ElementFactory.make("nvv4l2h264enc", f"{prefix}_enc")
        # if not enc:
        #     enc = Gst.ElementFactory.make("x264enc", f"{prefix}_enc")
        #     enc.set_property("bitrate", bitrate // 1000)
        #     enc.set_property("speed-preset", "ultrafast")
        #     enc.set_property("tune", "zerolatency")
        # else:
        #     enc.set_property("bitrate", bitrate)
        #     enc.set_property("tuning-info-id", 2)
        #     enc.set_property("iframeinterval", 30)
        #     enc.set_property("idrinterval", 30)
        enc = Gst.ElementFactory.make("x264enc", f"{prefix}_enc")
        enc.set_property("bitrate", bitrate // 1000)
        enc.set_property("speed-preset", "ultrafast")
        enc.set_property("tune", "zerolatency")
        elements.append(enc)

        # H264 parser
        parse = Gst.ElementFactory.make("h264parse", f"{prefix}_parse")
        parse.set_property("config-interval", -1)
        elements.append(parse)

        # RTSP client sink
        sink = Gst.ElementFactory.make("rtspclientsink", f"{prefix}_sink")
        sink.set_property("location", location)
        sink.set_property("protocols", 4)  # TCP
        sink.set_property("latency", 100)
        sink.set_property("do-rtsp-keep-alive", True)
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
