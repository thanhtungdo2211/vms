"""Stream Publisher - Stream individual cameras WITH annotations via SRT.

Uses nvstreamdemux to split batched frames to individual streams.
Each camera gets its own OSD element for drawing annotations.

Architecture:
    Muxer → PGIE → Tracker → SGIE → nvstreamdemux → src_0 → OSD → enc → SRT (cam0)
                                                  → src_1 → OSD → enc → SRT (cam1)
                                                  → src_N ...

SRT URL Format:
    srt://host:port?streamid=publish:stream_name&pkt_size=1316
"""

from __future__ import annotations

# Standard library
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Third-party
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Local
if TYPE_CHECKING:
    from src.pipeline_builder import BranchInfo
    from src.camera_manager import MultibranchCameraManager

from src.common import make_element, detect_platform, get_encoder_element, get_nvvidconv_props

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
        self._counter = 0
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

    def _set_elements_state(self, elements: list, state: Gst.State, wait: bool = True) -> bool:
        """Set all elements to target state. Returns False if any failed."""
        success = True
        for elem in elements:
            ret = elem.set_state(state)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning(f"[StreamPublisher] {elem.get_name()} failed -> {state.value_nick}")
                success = False
            if wait:
                elem.get_state(STATE_CHANGE_TIMEOUT)
        return success

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

    def _link_chain(self, elements: list) -> bool:
        """Link element chain. Returns True on success."""
        for i in range(len(elements) - 1):
            src_elem, dst_elem = elements[i], elements[i + 1]
            if not src_elem.link(dst_elem):
                raise RuntimeError(f"Failed to link {src_elem.get_name()} -> {dst_elem.get_name()}")
        return True

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

    def _parse_srt_uri(self, uri: str) -> dict:
        """Parse SRT URI and extract streamid from query params.

        srtsink requires streamid as separate property, not in URI.
        Input:  srt://host:port?streamid=publish:name&pkt_size=1316
        Output: {"uri": "srt://host:port", "streamid": "publish:name"}
        """
        parsed = urlparse(uri)
        query = parse_qs(parsed.query)

        # Build clean URI without query params
        clean_uri = f"srt://{parsed.hostname}:{parsed.port or 8890}"

        result = {"uri": clean_uri}

        # Extract streamid if present
        if "streamid" in query:
            result["streamid"] = query["streamid"][0]

        return result

    def _create_stream_chain(self, prefix: str, uri: str, bitrate: int) -> list:
        """Create SRT sink element chain with OSD for per-camera streaming.

        Pipeline: queue → nvdsosd → nvvideoconvert → capsfilter → encoder → h264parse → mpegtsmux → srtsink
        """
        platform = detect_platform()
        elements = []

        # Queue - buffering for timing variations
        queue = make_element("queue", f"{prefix}_q", {
            "max-size-buffers": 60,
            "max-size-time": 2 * Gst.SECOND,
            "leaky": 2
        })
        elements.append(queue)

        # OSD - draw annotations
        try:
            osd = make_element("nvdsosd", f"{prefix}_osd", {
                "process-mode": 1,
                "display-text": 1
            })
            elements.append(osd)
        except RuntimeError:
            logger.warning("[StreamPublisher] nvdsosd not available, skipping OSD")

        # Video converter - platform-optimized settings
        try:
            conv = make_element("nvvideoconvert", f"{prefix}_conv", get_nvvidconv_props())
        except RuntimeError:
            conv = make_element("videoconvert", f"{prefix}_conv")
        elements.append(conv)

        # Caps filter - encoder needs I420 format
        caps = make_element("capsfilter", f"{prefix}_caps", {
            "caps": Gst.Caps.from_string("video/x-raw,format=I420")
        })
        elements.append(caps)

        # H264 encoder - platform-optimized
        enc_factory, enc_name, enc_props = get_encoder_element(prefix, bitrate)
        enc = make_element(enc_factory, enc_name, enc_props)
        elements.append(enc)
        logger.info(f"[StreamPublisher] Using encoder: {enc_factory} ({platform.name})")

        # H264 parser
        parse = make_element("h264parse", f"{prefix}_parse", {
            "config-interval": -1
        })
        elements.append(parse)

        # MPEG-TS muxer (SRT typically uses MPEG-TS container)
        mux = make_element("mpegtsmux", f"{prefix}_mux", {
            "alignment": 7  # Align to 188*7 = 1316 bytes (SRT packet size)
        })
        elements.append(mux)

        # Parse URI and extract streamid
        srt_params = self._parse_srt_uri(uri)
        sink_props = {
            "uri": srt_params["uri"],
            "wait-for-connection": False,
            "sync": False
        }
        if "streamid" in srt_params:
            sink_props["streamid"] = srt_params["streamid"]

        sink = make_element("srtsink", f"{prefix}_sink", sink_props)
        elements.append(sink)

        return elements

    def start_publish(
        self,
        camera_id: str,
        branch_name: str,
        uri: str,
        bitrate: int = 4000000
    ) -> bool:
        """Start SRT publishing for specific camera with annotations."""
        if not uri.startswith("srt://"):
            logger.error(f"[StreamPublisher] Invalid SRT URL: {uri}")
            return False

        with self._lock:
            key = (camera_id, branch_name)

            if key in self._publishers:
                logger.warning(f"[StreamPublisher] {camera_id}/{branch_name} already publishing")
                return True

            # Validate camera exists
            cam = self.camera_manager.get_camera(camera_id)
            if not cam:
                logger.error(f"[StreamPublisher] Camera {camera_id} not found")
                return False

            # Get demux element
            demux = self._get_demux(branch_name)
            if not demux:
                logger.error(f"[StreamPublisher] No nvstreamdemux in branch {branch_name}")
                return False

            elements = []
            try:
                self._counter += 1
                prefix = f"stream_{camera_id}_{self._counter}"

                # Create and add elements
                elements = self._create_stream_chain(prefix, uri, bitrate)
                for elem in elements:
                    self.pipeline.add(elem)

                # Link stream chain
                self._link_chain(elements)

                # Get demux pad
                demux_src = self._get_demux_pad(demux, cam.source_id)
                if not demux_src:
                    raise RuntimeError(f"Could not get demux pad src_{cam.source_id}")

                queue_sink = elements[0].get_static_pad("sink")

                # State transitions BEFORE linking to demux
                _, pipeline_state, _ = self.pipeline.get_state(0)
                logger.info(f"[StreamPublisher] Pipeline: {pipeline_state}, transitioning elements...")

                # READY -> PAUSED -> link -> PLAYING
                self._set_elements_state(elements, Gst.State.READY)
                time.sleep(0.1)
                self._set_elements_state(elements, Gst.State.PAUSED, wait=False)
                time.sleep(0.2)

                # Link to demux
                link_result = demux_src.link(queue_sink)
                if link_result != Gst.PadLinkReturn.OK:
                    raise RuntimeError(f"Failed to link demux pad: {link_result}")
                logger.info(f"[StreamPublisher] Linked src_{cam.source_id} to stream chain")

                # Transition to PLAYING
                time.sleep(0.2)
                if pipeline_state == Gst.State.PLAYING:
                    self._set_elements_state(elements, Gst.State.PLAYING)
                time.sleep(0.3)

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
                self._cleanup_elements(elements)
                return False

    def _cleanup_elements(self, elements: list) -> None:
        """Cleanup elements on error."""
        for elem in elements:
            try:
                elem.set_state(Gst.State.NULL)
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

                    def block_probe(pad, info):
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

                # Set elements to NULL and remove
                for elem in reversed(elements):
                    elem.set_state(Gst.State.NULL)
                    elem.get_state(STATE_CHANGE_TIMEOUT)
                    self.pipeline.remove(elem)

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
