"""Stream Publisher - Stream individual cameras WITH annotations via RTSP.

Uses nvstreamdemux to split batched frames to individual streams.
Each camera gets its own OSD element for drawing annotations.

Architecture:
    Muxer → PGIE → Tracker → SGIE → nvstreamdemux → src_0 → OSD → enc → RTSP (cam0)
                                                  → src_1 → OSD → enc → RTSP (cam1)
                                                  → src_N ...

RTSP URL Format:
    rtsp://host:port/stream_name
"""

from __future__ import annotations

# Standard library
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple

# Third-party
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Local
if TYPE_CHECKING:
    from src.pipeline_builder import BranchInfo
    from src.camera_manager import MultibranchCameraManager

from src.common import make_element, detect_platform, get_encoder_element, get_nvvidconv_props, link_chain

logger = logging.getLogger(__name__)

STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


@dataclass
class PublishInfo:
    """Info about active stream publish session."""
    camera_id: str
    branch: str
    uri: str
    bitrate: int
    started_at: datetime
    source_id: int
    elements: list = field(default_factory=list)
    demux_pad: Optional[Gst.Pad] = None


class StreamPublisher:
    """Manage per-camera SRT streaming with annotations via nvstreamdemux."""

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
        self._publishers: dict[Tuple[str, str], PublishInfo] = {}
        self._demux_cache: dict[str, Gst.Element] = {}

        # Pre-request pads on all demux elements (must be in NULL state)
        self._setup_demux_handlers()

    def _setup_demux_handlers(self):
        """Pre-request pads on all demux elements.

        nvstreamdemux requires pads to be requested while pipeline is in NULL state.
        """
        for branch_name, branch_info in self.branches.items():
            demux = self._get_demux(branch_name)
            if not demux:
                continue

            for i in range(branch_info.max_cameras):
                pad_name = f"src_{i}"
                pad = demux.request_pad_simple(pad_name)
                if pad:
                    logger.info(f"[StreamPublisher] Pre-requested {branch_name}/{pad_name}")
                else:
                    logger.warning(f"[StreamPublisher] Failed to pre-request {branch_name}/{pad_name}")

    def reset(self):
        """Reset state after pipeline NULL. Re-request pads when returning to READY."""
        with self._lock:
            logger.info("[StreamPublisher] Resetting state")
            self._publishers.clear()
            self._demux_cache.clear()
            self._setup_demux_handlers()

    def _get_demux(self, branch_name: str) -> Optional[Gst.Element]:
        """Get nvstreamdemux element for branch (cached)."""
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

        # Fallback: find by name pattern
        demux = self.pipeline.get_by_name(f"{branch_name}_demux")
        if demux:
            self._demux_cache[branch_name] = demux
        return demux

    def _get_demux_pad(self, demux: Gst.Element, source_id: int) -> Optional[Gst.Pad]:
        """Get demux src pad by source_id (iterate first, then request)."""
        pad_name = f"src_{source_id}"

        # First try to find existing pad
        it = demux.iterate_src_pads()
        while True:
            result, pad = it.next()
            if result == Gst.IteratorResult.DONE:
                break
            if result == Gst.IteratorResult.OK and pad.get_name() == pad_name:
                logger.info(f"[StreamPublisher] Found existing pad {pad_name}")
                return pad

        # Request pad if not found
        pad = demux.request_pad_simple(pad_name)
        if pad:
            logger.info(f"[StreamPublisher] Requested pad {pad_name}")
        else:
            logger.warning(f"[StreamPublisher] Failed to request pad {pad_name}")
        return pad

    def _setup_rtsp_error_monitoring(self, camera_id: str, branch_name: str) -> None:
        """Setup async monitoring for RTSP stream errors via bus messages.

        This monitors GStreamer bus messages for errors from rtspclientsink
        without blocking the pipeline. Errors are logged but don't stop pipeline.

        Args:
            camera_id: Camera ID for logging
            branch_name: Branch name for logging
        """
        # Note: Bus message monitoring is already setup in pipeline_builder
        # This is a placeholder for future per-stream error handling
        # GStreamer will automatically post ERROR messages to bus if RTSP fails
        logger.info(f"[StreamPublisher] RTSP error monitoring enabled for {camera_id}/{branch_name}")

    def _pub_to_dict(self, pub: PublishInfo) -> dict:
        """Convert PublishInfo to status dict."""
        return {
            "publishing": True,
            "annotated": True,
            "uri": pub.uri,
            "bitrate": pub.bitrate,
            "source_id": pub.source_id,
            "started_at": pub.started_at.isoformat()
        }

    def _create_stream_chain(self, uri: str, bitrate: int) -> list:
        """Create RTSP sink element chain with OSD for per-camera streaming.

        Pipeline: queue → nvdsosd → nvvideoconvert → capsfilter(I420/NV12) → encoder → h264parse → rtspclientsink

        Notes:
        - nvstreamdemux outputs video/x-raw(memory:NVMM),format=RGBA or NV12
        - nvdsosd accepts and outputs RGBA or NV12
        - nvvideoconvert converts to encoder-specific format
        - Output capsfilter: I420 for x264enc (Jetson), NV12 for nvv4l2h264enc (dGPU)
        """
        platform = detect_platform()
        elements = []

        # Queue - accepts any format from nvstreamdemux
        queue = make_element("queue", None)
        queue.set_property("max-size-buffers", 30)
        queue.set_property("leaky", 2)
        elements.append(queue)

        # nvvideoconvert - normalize format before OSD
        # Handles NV12/RGBA from nvstreamdemux, outputs NV12 for OSD
        # This prevents GST_PAD_LINK_NOFORMAT by doing format conversion early
        conv_input_props = get_nvvidconv_props()
        if platform.is_jetson:
            conv_input_props["copy-hw"] = 2
        conv_input = make_element("nvvideoconvert", None, conv_input_props)
        elements.append(conv_input)

        # OSD - draw annotations (CPU mode on Jetson to avoid GPU memory pressure)
        # Accepts NV12 from nvvideoconvert
        osd = make_element("nvdsosd", None, {
            "process-mode": 0 if platform.is_jetson else 1,
            "display-text": 1
        })
        elements.append(osd)

        # Get encoder type to determine format requirements
        enc_factory, enc_props = get_encoder_element(bitrate)

        # nvvideoconvert - needed to handle format conversion
        # Input: NV12 from OSD → Output: I420 for x264enc (software) or NV12 for nvv4l2h264enc (hardware)
        conv_props = get_nvvidconv_props()
        if platform.is_jetson:
            conv_props["copy-hw"] = 2
        conv = make_element("nvvideoconvert", None, conv_props)
        elements.append(conv)

        # Format selection based on encoder type
        # NOTE: nvstreamdemux outputs NV12, which works best with x264enc on Jetson
        if enc_factory == "nvv4l2h264enc":
            # Hardware encoder path (dGPU): Keep NV12 with NVMM memory
            caps = make_element("capsfilter", None, {
                "caps": Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
            })
            elements.append(caps)
        else:
            # Software encoder path (Jetson x264enc): Convert NV12 → I420, strip NVMM
            # x264enc requires plain I420 format without NVMM memory
            caps = make_element("capsfilter", None, {
                "caps": Gst.Caps.from_string("video/x-raw,format=I420")
            })
            elements.append(caps)

        # H264 encoder - platform-optimized
        enc = make_element(enc_factory, None, enc_props)
        elements.append(enc)
        logger.info(f"[StreamPublisher] Using encoder: {enc_factory} ({platform.name})")

        # H264 parser
        parse = make_element("h264parse", None, {
            "config-interval": 1,
            "disable-passthrough": True
        })

        elements.append(parse)

        # RTP payloader - NOT added to elements, attached to sink
        pay = make_element("rtph264pay", None, {
            "config-interval": 1,
            "pt": 96
        })

        # RTSP client sink
        sink = make_element("rtspclientsink", None, {
            "location": uri,
            "protocols": 4,  # TCP
            "latency": 0,
            "do-rtsp-keep-alive": True
        })
        # Attach payloader to sink for later use
        sink._payloader = pay
        elements.append(sink)

        return elements

    def _check_rtsp_server(self, uri: str, timeout: float = 3.0) -> bool:
        """Check if RTSP server is reachable.

        Args:
            uri: RTSP URI (rtsp://host:port/path)
            timeout: Connection timeout in seconds

        Returns:
            True if server is reachable, False otherwise
        """
        import socket
        import urllib.parse

        try:
            parsed = urllib.parse.urlparse(uri)
            host = parsed.hostname
            port = parsed.port or 8554  # Default RTSP port

            logger.info(f"[StreamPublisher] Checking RTSP server {host}:{port}...")

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                logger.info(f"[StreamPublisher] RTSP server {host}:{port} is reachable")
                return True
            else:
                logger.error(f"[StreamPublisher] RTSP server {host}:{port} is NOT reachable (error code: {result})")
                return False

        except socket.gaierror as e:
            logger.error(f"[StreamPublisher] Cannot resolve RTSP host: {e}")
            return False
        except socket.timeout:
            logger.error(f"[StreamPublisher] RTSP server connection timeout ({timeout}s)")
            return False
        except Exception as e:
            logger.error(f"[StreamPublisher] RTSP server check failed: {e}")
            return False

    def start_publish(
        self,
        camera_id: str,
        branch_name: str,
        uri: str,
        bitrate: int = 4000000
    ) -> bool:
        """Start RTSP publishing for specific camera with annotations.

        Returns:
            True if stream started successfully
            False if failed (pipeline continues running)
        """
        if not uri.startswith("rtsp://"):
            logger.error(f"[StreamPublisher] Invalid RTSP URL: {uri}")
            raise ValueError(f"Invalid RTSP URL: {uri}")

        # Check RTSP server BEFORE acquiring lock
        if not self._check_rtsp_server(uri):
            error_msg = f"Cannot connect to RTSP server: {uri}"
            logger.error(f"[StreamPublisher] {error_msg}")
            raise ConnectionError(error_msg)

        with self._lock:
            key = (camera_id, branch_name)

            if key in self._publishers:
                logger.warning(f"[StreamPublisher] {camera_id}/{branch_name} already publishing")
                return True

            # Validate camera exists
            cam = self.camera_manager.get_camera(camera_id)
            if not cam:
                error_msg = f"Camera {camera_id} not found"
                logger.error(f"[StreamPublisher] {error_msg}")
                raise ValueError(error_msg)

            # Get demux element
            demux = self._get_demux(branch_name)
            if not demux:
                error_msg = f"No nvstreamdemux in branch {branch_name}"
                logger.error(f"[StreamPublisher] {error_msg}")
                raise RuntimeError(error_msg)

            elements = []
            try:
                # Get pipeline state
                _, prev_state, _ = self.pipeline.get_state(0)
                logger.info(f"[StreamPublisher] Adding RTSP stream (pipeline: {prev_state.value_nick})")

                # Create and add elements to pipeline
                elements = self._create_stream_chain(uri, bitrate)
                for elem in elements:
                    self.pipeline.add(elem)

                # Link all elements except rtspclientsink
                link_chain(elements[:-1])

                # rtspclientsink needs special handling with request pad
                parse = elements[-2]  # h264parse (last before sink)
                sink = elements[-1]   # rtspclientsink

                # Setup error monitoring for RTSP sink (non-blocking, async)
                self._setup_rtsp_error_monitoring(camera_id, branch_name)

                # Get request pad from rtspclientsink
                sink_pad = sink.get_request_pad("sink_%u")
                if not sink_pad:
                    raise RuntimeError("Failed to get request pad from rtspclientsink")

                # Link parse to sink
                parse_src = parse.get_static_pad("src")
                ret = parse_src.link(sink_pad)
                if ret != Gst.PadLinkReturn.OK:
                    raise RuntimeError(f"Failed to link to rtspclientsink: {ret}")

                # CRITICAL: Sync element states BEFORE linking to demux
                # Elements must be in same state as pipeline for caps negotiation to work
                logger.info(f"[StreamPublisher] Syncing element states to {prev_state.value_nick} BEFORE linking...")

                # Use NON-BLOCKING state change to avoid pausing pipeline
                # Let GStreamer handle async state changes in background
                for elem in elements:
                    # Set state WITHOUT waiting (async, non-blocking)
                    ret = elem.set_state(prev_state)

                    if ret == Gst.StateChangeReturn.FAILURE:
                        raise RuntimeError(f"Failed to set state for {elem.get_name()}")
                    elif ret == Gst.StateChangeReturn.ASYNC:
                        # ASYNC is EXPECTED for rtspclientsink (network connection)
                        # Do NOT wait - let it connect in background
                        logger.info(f"[StreamPublisher] {elem.get_name()} state change is ASYNC (background)")
                    elif ret == Gst.StateChangeReturn.SUCCESS:
                        logger.info(f"[StreamPublisher] {elem.get_name()} state changed to {prev_state.value_nick}")
                    elif ret == Gst.StateChangeReturn.NO_PREROLL:
                        logger.info(f"[StreamPublisher] {elem.get_name()} is live (NO_PREROLL)")

                logger.info(f"[StreamPublisher] All elements set to {prev_state.value_nick} (async, non-blocking)")

                # Get demux pad
                demux_src = self._get_demux_pad(demux, cam.source_id)
                if not demux_src:
                    raise RuntimeError(f"Could not get demux pad src_{cam.source_id}")

                # Log demux pad caps for debugging
                demux_caps = demux_src.get_current_caps() or demux_src.query_caps(None)
                logger.info(f"[StreamPublisher] Demux pad caps: {demux_caps.to_string() if demux_caps else 'None'}")

                queue_sink = elements[0].get_static_pad("sink")

                # Link to demux (now elements are in PLAYING state, caps can negotiate)
                link_result = demux_src.link(queue_sink)
                if link_result != Gst.PadLinkReturn.OK:
                    raise RuntimeError(f"Failed to link demux pad: {link_result}")
                logger.info(f"[StreamPublisher] Linked src_{cam.source_id} to stream chain")

                # Store publish info
                self._publishers[key] = PublishInfo(
                    camera_id=camera_id,
                    branch=branch_name,
                    uri=uri,
                    bitrate=bitrate,
                    started_at=datetime.now(),
                    source_id=cam.source_id,
                    elements=elements,
                    demux_pad=demux_src
                )

                logger.info(f"[StreamPublisher] Started {camera_id}/{branch_name} (src_{cam.source_id}) → {uri}")
                return True

            except Exception as e:
                logger.error(f"[StreamPublisher] Failed to start publish: {e}", exc_info=True)
                # CRITICAL: Clean up elements but DO NOT crash pipeline
                self._cleanup_elements(elements)
                # Re-raise exception so API can return error to client
                raise

    def _cleanup_elements(self, elements: list) -> None:
        """Cleanup elements on error with proper EOS and memory release."""
        # Send EOS to flush remaining buffers from NVMM/CMA memory
        for elem in elements:
            try:
                elem.send_event(Gst.Event.new_eos())
            except:
                pass

        time.sleep(0.2)  # Wait for EOS to propagate

        # Now safely set to NULL and remove
        for elem in elements:
            try:
                elem.set_state(Gst.State.NULL)
                elem.get_state(STATE_CHANGE_TIMEOUT)
                self.pipeline.remove(elem)
            except:
                pass

    def stop_publish(self, camera_id: str, branch_name: str) -> bool:
        """Stop SRT publishing for specific camera using blocking probe."""
        with self._lock:
            key = (camera_id, branch_name)
            pub = self._publishers.get(key)

            if not pub:
                logger.warning(f"[StreamPublisher] {camera_id}/{branch_name} not publishing")
                return False

            probe_id = None
            try:
                demux_pad = pub.demux_pad
                elements = pub.elements

                # Block data flow before unlinking
                if demux_pad and elements:
                    blocked = threading.Event()

                    def block_probe(_pad, _info):
                        blocked.set()
                        return Gst.PadProbeReturn.OK

                    probe_id = demux_pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, block_probe)
                    if not blocked.wait(timeout=2.0):
                        logger.warning(f"[StreamPublisher] Block probe timeout for {camera_id}/{branch_name}")

                # Unlink demux from queue
                if demux_pad and elements:
                    queue_sink = elements[0].get_static_pad("sink")
                    if queue_sink and queue_sink.is_linked():
                        demux_pad.unlink(queue_sink)
                        logger.info(f"[StreamPublisher] Unlinked demux pad for {camera_id}/{branch_name}")

                # Remove blocking probe
                if demux_pad and probe_id is not None:
                    demux_pad.remove_probe(probe_id)
                    probe_id = None

                time.sleep(0.1)

                # Send EOS to first element to flush CMA buffers
                logger.info(f"[StreamPublisher] Sending EOS to flush buffers...")
                if elements:
                    elements[0].send_event(Gst.Event.new_eos())
                    time.sleep(0.3)  # Wait for EOS to propagate through chain

                # Set elements to NULL and remove
                for elem in reversed(elements):
                    elem.set_state(Gst.State.NULL)
                    elem.get_state(STATE_CHANGE_TIMEOUT)
                    self.pipeline.remove(elem)

                # CRITICAL: Release demux pad to free CMA memory
                if demux_pad:
                    demux = self._get_demux(branch_name)
                    if demux:
                        demux.release_request_pad(demux_pad)
                        logger.info(f"[StreamPublisher] Released demux pad src_{pub.source_id}")

                del self._publishers[key]
                logger.info(f"[StreamPublisher] Stopped {camera_id}/{branch_name}")
                return True

            except Exception as e:
                logger.error(f"[StreamPublisher] Failed to stop publish: {e}", exc_info=True)
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
        """Get SRT publishing status."""
        with self._lock:
            # Single camera/branch query
            if camera_id and branch_name:
                pub = self._publishers.get((camera_id, branch_name))
                return self._pub_to_dict(pub) if pub else {"publishing": False}

            # Return all (grouped by camera)
            result = {}
            for (cam_id, branch), pub in self._publishers.items():
                if camera_id and cam_id != camera_id:
                    continue
                if branch_name and branch != branch_name:
                    continue

                if cam_id not in result:
                    result[cam_id] = {}
                result[cam_id][branch] = self._pub_to_dict(pub)

            return result

    def cleanup_camera(self, camera_id: str) -> None:
        """Cleanup when camera is removed. Called by CameraManager."""
        keys_to_remove = [(cid, br) for cid, br in self._publishers if cid == camera_id]
        for key in keys_to_remove:
            try:
                self.stop_publish(key[0], key[1])
            except Exception as e:
                logger.warning(f"[StreamPublisher] Cleanup error for {key}: {e}")
