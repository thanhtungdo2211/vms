"""RTSP Publisher Manager - Dynamic on-demand RTSP stream publishing.

Provides API to start/stop RTSP publishing for branches at runtime
without pipeline restart. Uses valve-based approach for instant switching.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

if TYPE_CHECKING:
    from src.pipeline_builder import BranchInfo

logger = logging.getLogger(__name__)

# State transition timeout
STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


@dataclass
class RtspPublishInfo:
    """Info about active RTSP publish session."""
    branch: str
    location: str
    bitrate: int
    started_at: datetime
    elements: list = field(default_factory=list)
    valve: Optional[Gst.Element] = None
    tee_pad: Optional[Gst.Pad] = None


class RtspPublisherManager:
    """Manage dynamic RTSP stream publishing for branches."""

    def __init__(self, pipeline: Gst.Pipeline, branches: dict[str, "BranchInfo"]):
        self.pipeline = pipeline
        self.branches = branches
        self._publishers: dict[str, RtspPublishInfo] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def start_publish(self, branch_name: str, location: str, bitrate: int = 4000000) -> bool:
        """Start RTSP publishing for a branch.

        Creates a tee from the branch's last element (before sink) and dynamically
        adds RTSP sink chain (Queue -> Conv -> Enc -> Parse -> Pay -> Sink).

        Args:
            branch_name: Target branch name
            location: RTSP server URL (e.g., rtsp://192.168.6.14:8554/stream)
            bitrate: H264 encoding bitrate in bits/sec

        Returns:
            True if started successfully
        """
        # Input validation
        if not location.startswith("rtsp://"):
            logger.error(f"[RTSP] Invalid RTSP URL: {location}")
            return False

        with self._lock:
            if branch_name in self._publishers:
                logger.warning(f"[RTSP] Branch {branch_name} already publishing")
                return False

            branch = self.branches.get(branch_name)
            if not branch:
                logger.error(f"[RTSP] Branch {branch_name} not found")
                return False

            try:
                self._counter += 1
                prefix = f"rtsp_pub_{self._counter}"

                # Get pipeline state
                _, prev_state, _ = self.pipeline.get_state(0)

                # Pause pipeline for safe topology change
                if prev_state == Gst.State.PLAYING:
                    logger.info(f"[RTSP] Pausing pipeline for RTSP setup")
                    self.pipeline.set_state(Gst.State.PAUSED)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    time.sleep(0.3)

                # Find the element before sink (usually nvdsosd)
                # We'll tap after OSD for the RTSP stream
                source_elem = self._find_osd_element(branch)
                if not source_elem:
                    logger.error(f"[RTSP] Cannot find OSD element in branch {branch_name}")
                    if prev_state == Gst.State.PLAYING:
                        self.pipeline.set_state(Gst.State.PLAYING)
                    return False

                # Create elements for RTSP chain
                elements = self._create_rtsp_chain(prefix, location, bitrate)

                # Add elements to pipeline
                for elem in elements:
                    self.pipeline.add(elem)

                # Get OSD src pad and its current peer (sink)
                osd_src = source_elem.get_static_pad("src")
                if not osd_src:
                    raise RuntimeError("Cannot get OSD src pad")

                sink_peer = osd_src.get_peer()

                # Create tee to split stream
                tee = Gst.ElementFactory.make("tee", f"{prefix}_tee")
                tee.set_property("allow-not-linked", True)
                self.pipeline.add(tee)

                # Unlink OSD from sink
                if sink_peer:
                    osd_src.unlink(sink_peer)

                # Link: OSD -> tee
                if not source_elem.link(tee):
                    raise RuntimeError("Failed to link OSD to tee")

                # Link: tee -> original sink (first branch)
                tee_src1 = tee.get_request_pad("src_%u")
                if sink_peer:
                    tee_src1.link(sink_peer)

                # Link: tee -> RTSP chain (second branch)
                tee_src2 = tee.get_request_pad("src_%u")
                queue = elements[0]
                queue_sink = queue.get_static_pad("sink")
                tee_src2.link(queue_sink)

                # Link RTSP chain elements
                for i in range(len(elements) - 1):
                    src_elem = elements[i]
                    dest_elem = elements[i + 1]

                    # Special handling for rtspclientsink which uses request pads
                    if "rtspclientsink" in dest_elem.get_factory().get_name():
                        sink_pad = dest_elem.get_request_pad("sink_%u")
                        if not sink_pad:
                            raise RuntimeError(f"Failed to get request pad from {dest_elem.get_name()}")

                        # Attach payloader if available
                        if hasattr(dest_elem, "_payloader"):
                            sink_pad.set_property("payloader", dest_elem._payloader)

                        src_pad = src_elem.get_static_pad("src")
                        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                            raise RuntimeError(f"Failed to link {src_elem.get_name()} to {dest_elem.get_name()}")
                    else:
                        if not src_elem.link(dest_elem):
                            raise RuntimeError(f"Failed to link RTSP chain: {src_elem.get_name()}")

                # Sync element states
                for elem in [tee] + elements:
                    elem.sync_state_with_parent()

                # Resume pipeline
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)

                # Store publish info
                self._publishers[branch_name] = RtspPublishInfo(
                    branch=branch_name,
                    location=location,
                    bitrate=bitrate,
                    started_at=datetime.now(),
                    elements=[tee] + elements,
                    tee_pad=tee_src2
                )

                logger.info(f"[RTSP] Started publishing {branch_name} to {location}")
                return True

            except Exception as e:
                logger.error(f"[RTSP] Failed to start publish: {e}", exc_info=True)

                # Cleanup partially added elements
                if 'tee' in locals() and tee:
                    try:
                        self.pipeline.remove(tee)
                    except Exception:
                        pass

                if 'elements' in locals():
                    for elem in elements:
                        try:
                            self.pipeline.remove(elem)
                        except Exception:
                            pass

                # Attempt to restore state
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                return False

    def stop_publish(self, branch_name: str) -> bool:
        """Stop RTSP publishing for a branch.

        Args:
            branch_name: Target branch name

        Returns:
            True if stopped successfully
        """
        with self._lock:
            pub = self._publishers.get(branch_name)
            if not pub:
                logger.warning(f"[RTSP] Branch {branch_name} not publishing")
                return False

            try:
                # Get pipeline state
                _, prev_state, _ = self.pipeline.get_state(0)

                # Pause pipeline
                if prev_state == Gst.State.PLAYING:
                    logger.info(f"[RTSP] Pausing pipeline for RTSP teardown")
                    self.pipeline.set_state(Gst.State.PAUSED)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    time.sleep(0.3)

                branch = self.branches.get(branch_name)
                tee = pub.elements[0] if pub.elements else None

                if tee and branch:
                    # Get tee sink pad's peer (OSD/Source)
                    tee_sink = tee.get_static_pad("sink")
                    source_src = tee_sink.get_peer() if tee_sink else None

                    # Get first tee src pad's peer (original sink)
                    tee_src1 = None
                    sink_peer = None

                    # Iterate pads to find the one connected to original sink
                    it = tee.iterate_src_pads()
                    while True:
                        ret, pad = it.next()
                        if ret == Gst.IteratorResult.OK:
                            peer = pad.get_peer()
                            # If peer is NOT our RTSP chain (we know RTSP chain starts with queue)
                            # Or simpler: check if it's NOT the pad we stored for RTSP
                            if pad != pub.tee_pad:
                                tee_src1 = pad
                                sink_peer = peer
                                break
                        elif ret == Gst.IteratorResult.RESYNC:
                            it.resync()
                        else:
                            break

                    # Unlink tee
                    if source_src:
                        source_src.unlink(tee_sink)

                    if tee_src1 and sink_peer:
                        tee_src1.unlink(sink_peer)

                    # Release request pads
                    if tee_src1:
                        tee.release_request_pad(tee_src1)
                    if pub.tee_pad:
                        tee.release_request_pad(pub.tee_pad)

                    # Reconnect Source directly to sink
                    source_elem = self._find_osd_element(branch)
                    if source_elem and sink_peer:
                        sink_elem = sink_peer.get_parent_element()
                        if sink_elem:
                            source_elem.link(sink_elem)

                # Set RTSP elements to NULL and remove
                for elem in reversed(pub.elements):
                    elem.set_state(Gst.State.NULL)
                    elem.get_state(Gst.CLOCK_TIME_NONE)
                    self.pipeline.remove(elem)

                # Resume pipeline
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)

                del self._publishers[branch_name]
                logger.info(f"[RTSP] Stopped publishing {branch_name}")
                return True

            except Exception as e:
                logger.error(f"[RTSP] Failed to stop publish: {e}", exc_info=True)
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                return False

    def get_status(self, branch_name: str = None) -> dict:
        """Get RTSP publishing status.

        Args:
            branch_name: Specific branch or None for all

        Returns:
            Status dict with publishing info
        """
        with self._lock:
            if branch_name:
                pub = self._publishers.get(branch_name)
                if pub:
                    return {
                        "publishing": True,
                        "location": pub.location,
                        "bitrate": pub.bitrate,
                        "started_at": pub.started_at.isoformat()
                    }
                return {"publishing": False}

            # Return all
            return {
                name: {
                    "publishing": True,
                    "location": pub.location,
                    "bitrate": pub.bitrate,
                    "started_at": pub.started_at.isoformat()
                }
                for name, pub in self._publishers.items()
            }

    def _find_osd_element(self, branch: "BranchInfo") -> Optional[Gst.Element]:
        """Find nvdsosd or last element before sink."""
        # Try to find nvdsosd
        for elem in reversed(branch.elements):
            name = elem.get_name().lower()
            if "osd" in name or "nvdsosd" in name:
                return elem

        # Fallback: last element in chain
        if branch.elements:
            return branch.elements[-1]

        return None

    def _create_rtsp_chain(self, prefix: str, location: str, bitrate: int) -> list:
        """Create RTSP sink element chain."""
        elements = []

        # Queue for buffering
        queue = Gst.ElementFactory.make("queue", f"{prefix}_q")
        queue.set_property("max-size-buffers", 30)
        queue.set_property("leaky", 2)
        elements.append(queue)

        # Video converter
        conv = Gst.ElementFactory.make("nvvideoconvert", f"{prefix}_conv")
        if not conv:
            conv = Gst.ElementFactory.make("videoconvert", f"{prefix}_conv")
        else:
            conv.set_property("compute-hw", 1)
            conv.set_property("nvbuf-memory-type", 3)
        elements.append(conv)

        # Caps filter for NV12
        caps = Gst.ElementFactory.make("capsfilter", f"{prefix}_caps")
        if conv.get_factory().get_name() == "nvvideoconvert":
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
        else:
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420"))
        elements.append(caps)

        # H264 encoder
        enc = Gst.ElementFactory.make("nvv4l2h264enc", f"{prefix}_enc")
        if not enc:
            enc = Gst.ElementFactory.make("x264enc", f"{prefix}_enc")
            enc.set_property("bitrate", bitrate // 1000)
            enc.set_property("speed-preset", "ultrafast")
            enc.set_property("tune", "zerolatency")
        else:
            enc.set_property("bitrate", bitrate)
            enc.set_property("tuning-info-id", 2)  # Low latency
            enc.set_property("iframeinterval", 30)
            enc.set_property("idrinterval", 30)
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

        # Payloader (not added to elements list for linking, but managed by sink)
        # However, we need to track it for cleanup
        payloader = Gst.ElementFactory.make("rtph264pay", f"{prefix}_pay")
        payloader.set_property("config-interval", 1)
        payloader.set_property("pt", 96)

        # We attach payloader to the sink element object so it can be accessed later if needed,
        # but mainly we need to attach it to the request pad during linking.
        # Since _create_rtsp_chain returns a list that start_publish iterates to link,
        # we need a way to pass the payloader.
        # Let's append it to the list but handle it specially in start_publish or handle it here?

        # Current start_publish logic:
        # for i in range(len(elements) - 1): ... link elements[i] to elements[i+1]

        # If we append payloader to elements, the loop will try to link sink -> payloader which is wrong.
        # The chain is ... -> parse -> sink (via pad with payloader)

        # So we return the payloader as a special attribute of the sink or similar?
        # Or just return it in the list and handle logic update in start_publish?
        # Let's attach it to the sink instance for convenience.
        sink._payloader = payloader

        return elements
