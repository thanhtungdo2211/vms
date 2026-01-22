"""Pipeline warmup manager - pre-loads TensorRT engines using dummy video source.

Provides warmup mechanism to pre-load inference engines before real cameras
are added, reducing first-camera initialization latency.

IMPORTANT: Does NOT use num-buffers (EOS causes memory corruption with nvstreamdemux).
Uses time-based warmup with blocking probe for clean shutdown.

Usage:
    from src.warmup_manager import WarmupManager

    warmup = WarmupManager(pipeline, branches)
    if warmup.warmup(timeout=15.0):
        print("Engines ready")
"""

from __future__ import annotations

# Standard library
import logging
import time
from typing import TYPE_CHECKING, Optional

# Third-party
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Local
if TYPE_CHECKING:
    from src.pipeline_builder import BranchInfo

from src.common import make_element

logger = logging.getLogger(__name__)

STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


class WarmupManager:
    """Pre-load TensorRT engines using dummy videotestsrc.

    Creates a temporary videotestsrc bin, links to all branches,
    runs pipeline through PAUSED→PLAYING to load engines,
    then blocks data flow and keeps source linked (stability fix).
    """

    WARMUP_CAMERA_ID = "__warmup_dummy__"

    def __init__(self, pipeline: Gst.Pipeline, branches: dict[str, "BranchInfo"], gpu_id: int = 0):
        self.pipeline = pipeline
        self.branches = branches
        self._gpu_id = gpu_id
        self._warmup_bin: Optional[Gst.Bin] = None
        self._source_id = 7  # Last pre-requested pad, leaves 0-6 for real cameras
        self._linked_pads: list = []

    def warmup(self, timeout: float = 15.0, num_frames: int = 64) -> bool:
        """Run warmup sequence to pre-load TensorRT engines.

        Args:
            timeout: Maximum time to wait for warmup (seconds)
            num_frames: Target frames for warmup timing (does NOT limit actual frames)

        Returns:
            True if warmup successful, False otherwise
        """
        logger.info("[WARMUP] Starting inference engine warmup...")
        start_time = time.time()

        try:
            # Create dummy source bin
            if not self._create_dummy_bin():
                logger.error("[WARMUP] Failed to create dummy source")
                return False

            # Add to pipeline and link to first branch
            self.pipeline.add(self._warmup_bin)
            first_branch = next(iter(self.branches.keys()))
            if not self._link_to_branch(first_branch, buffer_size=30):
                logger.error("[WARMUP] Failed to link dummy source")
                self._cleanup()
                return False

            # Transition to PAUSED (triggers engine loading)
            logger.info("[WARMUP] Setting pipeline to PAUSED (loading engines)...")
            self._warmup_bin.sync_state_with_parent()
            self._set_pipeline_state(Gst.State.PAUSED, timeout)

            # Transition to PLAYING (process frames)
            logger.info("[WARMUP] Setting pipeline to PLAYING (processing warmup frames)...")
            self._set_pipeline_state(Gst.State.PLAYING, timeout)

            # Wait for frames to flow through inference
            frame_time = num_frames / 30.0
            batch_time = 2.0
            wait_time = min(frame_time + batch_time, timeout - (time.time() - start_time))

            if wait_time > 0:
                logger.info(f"[WARMUP] Processing warmup frames (~{wait_time:.1f}s)...")
                time.sleep(wait_time)

            # Cleanup and return to READY
            elapsed = time.time() - start_time
            logger.info("[WARMUP] Cleanup - returning to PAUSED state...")
            self._cleanup()

            logger.info(f"[WARMUP] Complete in {elapsed:.1f}s - engines ready")
            return True

        except Exception as e:
            logger.error(f"[WARMUP] Failed: {e}", exc_info=True)
            self._cleanup()
            return False

    def _set_pipeline_state(self, state: Gst.State, timeout: float) -> bool:
        """Set pipeline state with timeout."""
        self.pipeline.set_state(state)
        ret, _, _ = self.pipeline.get_state(int(timeout * Gst.SECOND))
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.warning(f"[WARMUP] Pipeline {state.value_nick} transition warning")
            return False
        return True

    def _create_dummy_bin(self) -> bool:
        """Create videotestsrc bin with capsfilter and tee.

        IMPORTANT: Does NOT set num-buffers to avoid EOS which causes
        memory corruption with nvstreamdemux pre-requested pads.
        """
        try:
            self._warmup_bin = Gst.Bin.new(f"cam_{self.WARMUP_CAMERA_ID}")

            # videotestsrc - black pattern, infinite stream
            src = make_element("videotestsrc", None, {
                "pattern": 2,
                "is-live": True
            })

            # capsfilter - match typical inference input size
            caps = make_element("capsfilter", None, {
                "caps": Gst.Caps.from_string("video/x-raw,format=NV12,width=640,height=640,framerate=30/1")
            })

            # nvvideoconvert - convert to GPU memory
            try:
                nvconv = make_element("nvvideoconvert", None, {
                    "gpu-id": self._gpu_id
                })
            except RuntimeError:
                nvconv = None

            # tee for fanout to multiple branches
            tee = make_element("tee", f"tee_{self.WARMUP_CAMERA_ID}", {
                "allow-not-linked": True
            })

            # Add and link elements
            elements = [e for e in [src, caps, nvconv, tee] if e]
            for elem in elements:
                self._warmup_bin.add(elem)

            # Link chain
            if not self._link_elements([src, caps, nvconv, tee]):
                return False

            logger.debug("[WARMUP] Created dummy bin")
            return True

        except Exception as e:
            logger.error(f"[WARMUP] Failed to create dummy bin: {e}")
            return False

    def _link_elements(self, elements: list) -> bool:
        """Link elements sequentially, skipping None elements."""
        filtered = [e for e in elements if e]
        for i in range(len(filtered) - 1):
            if not filtered[i].link(filtered[i + 1]):
                logger.error(f"[WARMUP] Failed to link {filtered[i].get_name()} → {filtered[i+1].get_name()}")
                return False
        return True

    def _create_queue(self, buffer_size: int = 30) -> Gst.Element:
        """Create queue element with standard properties."""
        return make_element("queue", None, {
            "max-size-buffers": buffer_size,
            "max-size-bytes": 0,
            "max-size-time": 0,
            "leaky": 2
        })

    def _link_to_branch(self, branch_name: str, buffer_size: int = 30, sync_state: bool = False) -> bool:
        """Link warmup source to a specific branch.

        Args:
            branch_name: Branch to link to
            buffer_size: Queue buffer size (30 for initial warmup, 1 for remaining)
            sync_state: Whether to sync queue state with parent (for runtime linking)

        Returns:
            True if linked successfully
        """
        tee = self._warmup_bin.get_by_name(f"tee_{self.WARMUP_CAMERA_ID}")
        if not tee:
            return False

        branch = self.branches.get(branch_name)
        if not branch:
            return False

        try:
            # Create and add queue
            queue = self._create_queue(buffer_size=buffer_size)
            self._warmup_bin.add(queue)

            # Request tee src pad and link to queue
            tee_src = tee.request_pad_simple("src_%u")
            if tee_src.link(queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                logger.warning(f"[WARMUP] Failed to link tee → queue for {branch_name}")
                return False

            # Request mux sink pad
            mux_sink = branch.nvstreammux.request_pad_simple(f"sink_{self._source_id}")
            if not mux_sink:
                logger.warning(f"[WARMUP] Failed to request mux pad for {branch_name}")
                return False

            # Create ghost pad and link
            ghost = Gst.GhostPad.new(f"g_{branch_name}_{self.WARMUP_CAMERA_ID}", queue.get_static_pad("src"))
            self._warmup_bin.add_pad(ghost)

            if sync_state:
                queue.sync_state_with_parent()

            if ghost.link(mux_sink) == Gst.PadLinkReturn.OK:
                self._linked_pads.append((branch_name, ghost, mux_sink, tee_src))
                logger.debug(f"[WARMUP] Linked to branch: {branch_name}")
                return True
            else:
                logger.warning(f"[WARMUP] Failed to link ghost → mux for {branch_name}")
                return False

        except Exception as e:
            logger.warning(f"[WARMUP] Failed to link branch {branch_name}: {e}")
            return False

    def _link_remaining_branches(self) -> None:
        """Link warmup source to all remaining branches."""
        already_linked = {pad[0] for pad in self._linked_pads}

        for branch_name in self.branches:
            if branch_name in already_linked:
                continue
            if self._link_to_branch(branch_name, buffer_size=1, sync_state=True):
                logger.info(f"[WARMUP] Added link to remaining branch: {branch_name}")

    def _cleanup(self) -> None:
        """Stop warmup source but KEEP it linked to muxer.

        STABILITY FIX: Empty nvstreammux causes CUDA memory corruption.
        Instead of unlinking, just block data flow. This keeps muxer with
        at least 1 source at all times.

        Also links to ALL remaining branches to ensure every muxer has at least 1 source.
        """
        if not self._warmup_bin:
            return

        try:
            # Set pipeline to PAUSED (NOT NULL - preserves TensorRT engines)
            logger.debug("[WARMUP] Cleanup: setting pipeline to PAUSED")
            self.pipeline.set_state(Gst.State.PAUSED)
            self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
            time.sleep(0.3)

            # Block data flow on tee sink pad
            tee = self._warmup_bin.get_by_name(f"tee_{self.WARMUP_CAMERA_ID}")
            if tee:
                sink_pad = tee.get_static_pad("sink")
                if sink_pad:
                    sink_pad.add_probe(Gst.PadProbeType.BUFFER, lambda p, i: Gst.PadProbeReturn.DROP)
            time.sleep(0.2)

            # Link to ALL remaining branches
            self._link_remaining_branches()

            # Keep warmup bin linked, just set to PAUSED
            self._warmup_bin.set_state(Gst.State.PAUSED)

            linked_names = [pad[0] for pad in self._linked_pads]
            logger.info(f"[WARMUP] Cleanup complete - warmup linked to: {linked_names}")

        except Exception as e:
            logger.warning(f"[WARMUP] Cleanup error: {e}")
            try:
                self.pipeline.set_state(Gst.State.PAUSED)
            except Exception:
                pass
