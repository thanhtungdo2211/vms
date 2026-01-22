"""Pipeline warmup manager - pre-loads TensorRT engines using dummy video source.

This module provides a warmup mechanism to pre-load inference engines before
real cameras are added, reducing first-camera initialization latency.

IMPORTANT: Does NOT use num-buffers (EOS causes memory corruption with nvstreamdemux).
Uses time-based warmup with blocking probe for clean shutdown.

Usage:
    from src.warmup_manager import WarmupManager

    warmup = WarmupManager(pipeline, branches)
    if warmup.warmup(timeout=15.0):
        print("Engines ready")
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.pipeline_builder import BranchInfo

logger = logging.getLogger(__name__)

# State transition timeout (seconds)
STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


class WarmupManager:
    """Pre-load TensorRT engines using dummy videotestsrc.

    Creates a temporary videotestsrc bin, links to all branches,
    runs pipeline through PAUSED→PLAYING to load engines,
    then removes dummy source and returns to READY state.

    IMPORTANT: Does NOT use num-buffers to avoid EOS-related memory corruption.
    """

    WARMUP_CAMERA_ID = "__warmup_dummy__"

    def __init__(self, pipeline: Gst.Pipeline, branches: dict[str, BranchInfo], gpu_id: int = 0):
        self.pipeline = pipeline
        self.branches = branches
        self._gpu_id = gpu_id
        self._warmup_bin: Optional[Gst.Bin] = None
        # Use source_id=7 (last pre-requested pad) to avoid conflicts with real cameras (0-based)
        # Real cameras use IDs starting from 0, so using the last pad leaves 0-6 for real cameras
        self._source_id = 7
        self._linked_pads: list = []  # Track linked mux pads for cleanup

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
            # Step 1: Create dummy source bin (NO num-buffers - infinite stream)
            if not self._create_dummy_bin():
                logger.error("[WARMUP] Failed to create dummy source")
                return False

            # Step 2: Add to pipeline and link to all branches
            self.pipeline.add(self._warmup_bin)
            if not self._link_to_branches():
                logger.error("[WARMUP] Failed to link dummy source to branches")
                self._cleanup()
                return False

            # Step 3: Transition to PAUSED (triggers engine loading)
            logger.info("[WARMUP] Setting pipeline to PAUSED (loading engines)...")
            self._warmup_bin.sync_state_with_parent()
            self.pipeline.set_state(Gst.State.PAUSED)

            ret, _, _ = self.pipeline.get_state(int(timeout * Gst.SECOND))
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning("[WARMUP] Pipeline PAUSED transition warning")

            # Step 4: Transition to PLAYING (process frames)
            logger.info("[WARMUP] Setting pipeline to PLAYING (processing warmup frames)...")
            self.pipeline.set_state(Gst.State.PLAYING)

            ret, _, _ = self.pipeline.get_state(int(timeout * Gst.SECOND))
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning("[WARMUP] Pipeline PLAYING transition warning")

            # Step 5: Wait for frames to flow through inference (time-based, no EOS)
            frame_time = num_frames / 30.0  # 30fps assumed
            batch_time = 2.0  # Extra time for batch assembly and inference
            wait_time = min(frame_time + batch_time, timeout - (time.time() - start_time))

            if wait_time > 0:
                logger.info(f"[WARMUP] Processing warmup frames (~{wait_time:.1f}s)...")
                time.sleep(wait_time)

            # Step 6: Cleanup and return to READY
            elapsed = time.time() - start_time
            logger.info(f"[WARMUP] Cleanup - returning to READY state...")
            self._cleanup()

            logger.info(f"[WARMUP] Complete in {elapsed:.1f}s - engines ready")
            return True

        except Exception as e:
            logger.error(f"[WARMUP] Failed: {e}", exc_info=True)
            self._cleanup()
            return False

    def _create_dummy_bin(self) -> bool:
        """Create videotestsrc bin with capsfilter and tee.

        IMPORTANT: Does NOT set num-buffers to avoid EOS which causes
        memory corruption with nvstreamdemux pre-requested pads.
        """
        try:
            self._warmup_bin = Gst.Bin.new(f"cam_{self.WARMUP_CAMERA_ID}")

            # videotestsrc - black pattern, NO num-buffers (infinite stream)
            src = Gst.ElementFactory.make("videotestsrc", f"src_{self.WARMUP_CAMERA_ID}")
            src.set_property("pattern", 2)  # black
            src.set_property("is-live", True)
            # NO num-buffers - we'll stop by blocking data flow instead

            # capsfilter - match typical inference input size
            caps = Gst.ElementFactory.make("capsfilter", f"caps_{self.WARMUP_CAMERA_ID}")
            caps.set_property("caps", Gst.Caps.from_string(
                "video/x-raw,format=NV12,width=640,height=640,framerate=30/1"
            ))

            # nvvideoconvert - convert to GPU memory for DeepStream
            nvconv = Gst.ElementFactory.make("nvvideoconvert", f"nvconv_{self.WARMUP_CAMERA_ID}")
            if nvconv:
                nvconv.set_property("gpu-id", self._gpu_id)

            # tee for fanout to multiple branches
            tee = Gst.ElementFactory.make("tee", f"tee_{self.WARMUP_CAMERA_ID}")
            tee.set_property("allow-not-linked", True)

            # Add elements to bin
            for elem in [src, caps, nvconv, tee]:
                if elem:
                    self._warmup_bin.add(elem)

            # Link: src → caps → nvconv → tee
            if not src.link(caps):
                logger.error("[WARMUP] Failed to link src → caps")
                return False
            if nvconv:
                if not caps.link(nvconv):
                    logger.error("[WARMUP] Failed to link caps → nvconv")
                    return False
                if not nvconv.link(tee):
                    logger.error("[WARMUP] Failed to link nvconv → tee")
                    return False
            else:
                if not caps.link(tee):
                    logger.error("[WARMUP] Failed to link caps → tee")
                    return False

            logger.debug("[WARMUP] Created dummy bin (no num-buffers)")
            return True

        except Exception as e:
            logger.error(f"[WARMUP] Failed to create dummy bin: {e}")
            return False

    def _link_to_branches(self) -> bool:
        """Link dummy source tee to FIRST branch nvstreammux only (for warmup).

        During warmup, we only link to ONE branch to minimize complexity.
        After warmup (in _cleanup), we add links to remaining branches while
        pipeline is PAUSED to prevent empty muxer state.
        """
        tee = self._warmup_bin.get_by_name(f"tee_{self.WARMUP_CAMERA_ID}")
        if not tee:
            return False

        linked_branches = []
        self._linked_pads = []

        # Only link to first branch for warmup
        first_branch_name = list(self.branches.keys())[0]
        branch = self.branches[first_branch_name]

        try:
            # Create queue for this branch
            queue = Gst.ElementFactory.make("queue", f"q_{self.WARMUP_CAMERA_ID}_{first_branch_name}")
            queue.set_property("max-size-buffers", 30)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)
            queue.set_property("leaky", 2)
            self._warmup_bin.add(queue)

            # Request tee src pad and link to queue
            tee_src = tee.request_pad_simple("src_%u")
            if not tee_src.link(queue.get_static_pad("sink")) == Gst.PadLinkReturn.OK:
                logger.warning(f"[WARMUP] Failed to link tee → queue for {first_branch_name}")
                return False

            # Request nvstreammux sink pad and link via ghost pad
            mux_sink = branch.nvstreammux.request_pad_simple(f"sink_{self._source_id}")
            if not mux_sink:
                logger.warning(f"[WARMUP] Failed to request mux pad for {first_branch_name}")
                return False

            ghost = Gst.GhostPad.new(f"g_{first_branch_name}_{self.WARMUP_CAMERA_ID}", queue.get_static_pad("src"))
            self._warmup_bin.add_pad(ghost)

            if ghost.link(mux_sink) == Gst.PadLinkReturn.OK:
                linked_branches.append(first_branch_name)
                self._linked_pads.append((first_branch_name, ghost, mux_sink, tee_src))
                logger.debug(f"[WARMUP] Linked to branch: {first_branch_name}")
            else:
                logger.warning(f"[WARMUP] Failed to link ghost → mux for {first_branch_name}")
                return False

        except Exception as e:
            logger.warning(f"[WARMUP] Failed to link branch {first_branch_name}: {e}")
            return False

        if linked_branches:
            logger.info(f"[WARMUP] Linked to branch: {linked_branches[0]} (warmup single branch)")
            return True
        else:
            logger.error("[WARMUP] Failed to link to any branch")
            return False

    def _link_remaining_branches(self) -> None:
        """Link warmup source to remaining branches (call after warmup, while PAUSED).

        This ensures ALL muxers have at least 1 source, preventing empty muxer
        CUDA errors when real cameras are removed.
        """
        tee = self._warmup_bin.get_by_name(f"tee_{self.WARMUP_CAMERA_ID}")
        if not tee:
            return

        # Get already linked branches
        already_linked = {pad[0] for pad in self._linked_pads}

        for branch_name, branch in self.branches.items():
            if branch_name in already_linked:
                continue

            try:
                # Create queue for this branch
                queue = Gst.ElementFactory.make("queue", f"q_{self.WARMUP_CAMERA_ID}_{branch_name}")
                queue.set_property("max-size-buffers", 1)  # Minimal buffering
                queue.set_property("max-size-bytes", 0)
                queue.set_property("max-size-time", 0)
                queue.set_property("leaky", 2)
                self._warmup_bin.add(queue)

                # Request tee src pad and link to queue
                tee_src = tee.request_pad_simple("src_%u")
                if not tee_src.link(queue.get_static_pad("sink")) == Gst.PadLinkReturn.OK:
                    logger.warning(f"[WARMUP] Failed to link tee → queue for {branch_name}")
                    continue

                # Request nvstreammux sink pad and link via ghost pad
                mux_sink = branch.nvstreammux.request_pad_simple(f"sink_{self._source_id}")
                if not mux_sink:
                    logger.warning(f"[WARMUP] Failed to request mux pad for {branch_name}")
                    continue

                ghost = Gst.GhostPad.new(f"g_{branch_name}_{self.WARMUP_CAMERA_ID}", queue.get_static_pad("src"))
                self._warmup_bin.add_pad(ghost)

                # Sync queue state to pipeline
                queue.sync_state_with_parent()

                if ghost.link(mux_sink) == Gst.PadLinkReturn.OK:
                    self._linked_pads.append((branch_name, ghost, mux_sink, tee_src))
                    logger.info(f"[WARMUP] Added link to remaining branch: {branch_name}")
                else:
                    logger.warning(f"[WARMUP] Failed to link ghost → mux for {branch_name}")

            except Exception as e:
                logger.warning(f"[WARMUP] Failed to link remaining branch {branch_name}: {e}")

    def _cleanup(self) -> None:
        """Stop warmup source but KEEP it linked to muxer.

        STABILITY FIX (2026-01-22): Empty nvstreammux causes CUDA memory corruption.
        Instead of unlinking the warmup dummy from muxer, just block its data flow.
        This keeps muxer with at least 1 source at all times, preventing crashes
        when all real cameras are removed.

        Also links warmup source to ALL remaining branches (while PAUSED) to ensure
        every muxer has at least 1 source.

        Pipeline stays in PAUSED state - camera_manager will transition to PLAYING
        when first real camera is added.
        """
        if not self._warmup_bin:
            return

        try:
            # Step 1: Set pipeline to PAUSED (NOT NULL - preserves TensorRT engines)
            logger.debug("[WARMUP] Cleanup: setting pipeline to PAUSED")
            self.pipeline.set_state(Gst.State.PAUSED)
            ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning("[WARMUP] Pipeline PAUSED transition warning")
            time.sleep(0.3)

            # Step 2: Block data flow on tee sink pad (DROP all frames)
            tee = self._warmup_bin.get_by_name(f"tee_{self.WARMUP_CAMERA_ID}")
            if tee:
                sink_pad = tee.get_static_pad("sink")
                if sink_pad:
                    def block_probe(pad, info):
                        return Gst.PadProbeReturn.DROP
                    sink_pad.add_probe(Gst.PadProbeType.BUFFER, block_probe)
            time.sleep(0.2)  # Let in-flight buffers drain

            # Step 3: Link warmup source to ALL remaining branches (while PAUSED)
            # This ensures every muxer has at least 1 source
            self._link_remaining_branches()

            # STABILITY FIX: DO NOT unlink from muxer pads
            # Keep warmup source linked to muxer to prevent empty muxer state
            # This ensures nvstreammux always has at least 1 source
            logger.debug("[WARMUP] Keeping warmup source linked to ALL muxers (stability fix)")

            # DO NOT remove warmup bin - keep it in pipeline
            # Just set it to PAUSED to stop pushing frames
            self._warmup_bin.set_state(Gst.State.PAUSED)

            linked_branch_names = [pad[0] for pad in self._linked_pads]
            logger.info(f"[WARMUP] Cleanup complete - pipeline PAUSED, warmup linked to: {linked_branch_names}")

        except Exception as e:
            logger.warning(f"[WARMUP] Cleanup error: {e}")
            # Try to at least stay in PAUSED
            try:
                self.pipeline.set_state(Gst.State.PAUSED)
            except Exception:
                pass
