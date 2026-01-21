"""Dynamic camera management with tee fanout to multiple branches.

STABILITY FIX (2026-01-13):
- PAUSE pipeline during camera add/remove for safe topology changes
- Incremental state sync: NULL → READY → PAUSED → PLAYING with waits
- Update batch size AFTER state sync completes
- Reference: /home/mq/disk2T/duy/tks_prj/infra/deepstream
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.pipeline_builder import BranchInfo
from src.source_mapper import SourceIDMapper

logger = logging.getLogger(__name__)

# State transition timeout (5 seconds per state)
STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


class MultibranchCameraManager:
    """Add/remove cameras to multiple branches at runtime."""

    def __init__(self, pipeline: Gst.Pipeline, branches: dict[str, BranchInfo], gpu_id: int = 0):
        self.pipeline = pipeline
        self.branches = branches
        self._gpu_id = gpu_id
        self._cameras: dict[str, dict] = {}  # camera_id -> {bin, tee, source_id, uri, branch_pads}
        self._mapper = SourceIDMapper()
        self._lock = threading.Lock()
        self._pad_counter = 0
        self._last_op = 0.0
        self._rtsp_publisher = None  # DemuxRtspPublisher reference

    def set_rtsp_publisher(self, publisher) -> None:
        """Set the RTSP publisher for cleanup on camera removal."""
        self._rtsp_publisher = publisher

    def _delay(self):
        """Wait 2s between operations."""
        wait = 2.0 - (time.time() - self._last_op)
        if wait > 0:
            time.sleep(wait)

    def _incremental_state_sync(self, element: Gst.Element, target_state: Gst.State) -> bool:
        """Sync element state incrementally: NULL → READY → PAUSED → PLAYING.

        Reference implementation pattern from duy/tks_prj/infra/deepstream.
        Each transition waits for completion before proceeding.
        """
        element_name = element.get_name()

        # Always start from NULL
        element.set_state(Gst.State.NULL)

        # Incremental transitions with explicit waits
        if target_state >= Gst.State.READY:
            element.set_state(Gst.State.READY)
            ret, _, _ = element.get_state(STATE_CHANGE_TIMEOUT)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error(f"[CAM-MANAGER] Failed to set {element_name} to READY")
                return False
            logger.debug(f"[CAM-MANAGER] {element_name} set to READY")

        if target_state >= Gst.State.PAUSED:
            element.set_state(Gst.State.PAUSED)
            ret, _, _ = element.get_state(STATE_CHANGE_TIMEOUT)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error(f"[CAM-MANAGER] Failed to set {element_name} to PAUSED")
                return False
            logger.debug(f"[CAM-MANAGER] {element_name} set to PAUSED")

        if target_state >= Gst.State.PLAYING:
            element.set_state(Gst.State.PLAYING)
            ret, _, _ = element.get_state(STATE_CHANGE_TIMEOUT)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error(f"[CAM-MANAGER] Failed to set {element_name} to PLAYING")
                return False
            logger.debug(f"[CAM-MANAGER] {element_name} set to PLAYING")

        return True


    def add_camera(self, camera_id: str, uri: str, branch_names: list[str]) -> bool:
        """Add camera to branches with safe pipeline state management.

        STABILITY FIX:
        - If pipeline is READY (no cameras yet), transition to PLAYING after adding
        - If pipeline is PLAYING, PAUSE during topology change to prevent race conditions
        Reference: duy/tks_prj/infra/deepstream pattern.
        """
        with self._lock:
            self._delay()
            if camera_id in self._cameras:
                return False

            branches = [b for b in branch_names if b in self.branches]
            if not branches:
                return False

            # Get current pipeline state
            _, prev_state, _ = self.pipeline.get_state(0)
            is_first_camera = len(self._cameras) == 0
            logger.info(f"[CAM-MANAGER] Adding {camera_id} - pipeline state: {prev_state.value_nick}, first_camera: {is_first_camera}")

            try:
                # STEP 1: ALWAYS pause pipeline before adding new camera
                # This prevents nvurisrcbin internal pad conflict when adding during PLAYING
                if prev_state == Gst.State.PLAYING:
                    logger.info(f"[CAM-MANAGER] Pausing pipeline for safe camera addition")
                    self.pipeline.set_state(Gst.State.PAUSED)
                    ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.error(f"[CAM-MANAGER] Failed to pause pipeline")
                        return False
                    time.sleep(0.5)  # Brief stabilization

                # STEP 2: Create camera bin elements
                source_id = self._mapper.add(camera_id, uri)
                bin_elem = Gst.Bin.new(f"cam_{camera_id}")

                tee = Gst.ElementFactory.make("tee", f"tee_{camera_id}")
                tee.set_property("allow-not-linked", True)
                bin_elem.add(tee)

                # Track linked state to avoid duplicate connections
                linked_state = {"done": False}

                # Determine source type based on URI scheme
                is_rtsp = uri.startswith("rtsp://") or uri.startswith("rtsps://")
                is_file = uri.startswith("file://")

                # Use nvurisrcbin for RTSP sources (better DeepStream integration)
                if is_rtsp:
                    source = Gst.ElementFactory.make("nvurisrcbin", f"nvurisrc_{camera_id}")
                    source.set_property("uri", uri)
                    source.set_property("gpu-id", self._gpu_id)
                    source.set_property("disable-audio", True)
                    source.set_property("source-id", source_id)
                    source.set_property("cudadec-memtype", 0)  # Device memory
                    source.set_property("num-extra-surfaces", 2)  # Extra buffers for stability

                    # STABILITY: Jitter buffer settings to prevent "decreasing timestamp" warnings
                    # These help handle network jitter and out-of-order frames from RTSP streams
                    source.set_property("latency", 500)  # 500ms jitter buffer
                    source.set_property("drop-frame-interval", 0)  # Keep all frames

                    bin_elem.add(source)

                    # Connect pad-added for nvurisrcbin -> tee (video pads: vsrc_*)
                    def on_nvurisrc_pad_added(_s, pad, t):
                        if linked_state["done"]:
                            return
                        pad_name = pad.get_name()
                        if pad_name.startswith("vsrc_"):
                            sink = t.get_static_pad("sink")
                            if not sink.is_linked():
                                ret = pad.link(sink)
                                if ret == Gst.PadLinkReturn.OK:
                                    linked_state["done"] = True
                                    logger.info(f"[CAM-MANAGER] nvurisrcbin {pad_name} linked to tee for {camera_id}")
                                else:
                                    logger.warning(f"[CAM-MANAGER] Failed to link {pad_name} to tee: {ret}")

                    source.connect("pad-added", on_nvurisrc_pad_added, tee)
                    logger.info(f"[CAM-MANAGER] Using nvurisrcbin for RTSP source: {camera_id}")

                elif is_file:
                    # For file sources, use uridecodebin with proper state handling
                    # Create an event to signal when pad linking is complete
                    pad_linked_event = threading.Event()
                    # Use threading.Lock for thread-safe pad linking (callback from GStreamer thread)
                    pad_link_lock = threading.Lock()

                    source = Gst.ElementFactory.make("uridecodebin", f"uridecodebin_{camera_id}")
                    source.set_property("uri", uri)
                    bin_elem.add(source)

                    # Connect pad-added for uridecodebin -> tee (video pads only)
                    # Thread-safe: callback is invoked from GStreamer streaming thread
                    def on_uridecodebin_pad_added(_s, pad, t, cam_id, linked, event, lock):
                        with lock:
                            if linked["done"]:
                                return
                            caps = pad.get_current_caps()
                            if not caps:
                                caps = pad.query_caps(None)
                            if caps:
                                struct = caps.get_structure(0)
                                if struct and struct.get_name().startswith("video"):
                                    sink = t.get_static_pad("sink")
                                    if sink and not sink.is_linked():
                                        ret = pad.link(sink)
                                        if ret == Gst.PadLinkReturn.OK:
                                            linked["done"] = True
                                            event.set()  # Signal that pad is linked
                                            logger.info(f"[CAM-MANAGER] uridecodebin linked to tee for {cam_id}")
                                        else:
                                            logger.warning(f"[CAM-MANAGER] Failed to link uridecodebin to tee: {ret}")

                    source.connect("pad-added", on_uridecodebin_pad_added, tee, camera_id, linked_state, pad_linked_event, pad_link_lock)
                    logger.info(f"[CAM-MANAGER] Using uridecodebin for file source: {camera_id}")

                else:
                    # File/HTTP sources: use uridecodebin -> tee
                    source = Gst.ElementFactory.make("uridecodebin", f"uridecodebin_{camera_id}")
                    source.set_property("uri", uri)
                    bin_elem.add(source)

                    # Connect pad-added for uridecodebin -> tee (video pads only)
                    def on_uridecodebin_pad_added(_s, pad, t):
                        if linked_state["done"]:
                            return
                        caps = pad.get_current_caps()
                        if not caps:
                            caps = pad.query_caps(None)
                        if caps:
                            struct = caps.get_structure(0)
                            if struct and struct.get_name().startswith("video"):
                                sink = t.get_static_pad("sink")
                                if sink and not sink.is_linked():
                                    ret = pad.link(sink)
                                    if ret == Gst.PadLinkReturn.OK:
                                        linked_state["done"] = True
                                        logger.info(f"[CAM-MANAGER] uridecodebin linked to tee for {camera_id}")
                                    else:
                                        logger.warning(f"[CAM-MANAGER] Failed to link uridecodebin to tee: {ret}")

                    source.connect("pad-added", on_uridecodebin_pad_added, tee)
                    logger.info(f"[CAM-MANAGER] Using uridecodebin for file source: {camera_id}")

                # Note: Ghost pad for external access removed - it was unused and could cause issues
                # Each branch gets its own ghost pad via _link_branch

                self.pipeline.add(bin_elem)

                # STEP 3: Link to branches FIRST (before state sync) - like reference implementation
                # Link while bin is in NULL state for safe topology change
                # NOTE: DROP probe logic removed - warmup pre-loads engines, no race condition
                branch_pads = {}

                for idx, b in enumerate(branches):
                    logger.info(f"[CAM-MANAGER] Linking branch {idx+1}/{len(branches)}: {b}")
                    pad = self._link_branch(bin_elem, tee, camera_id, source_id, b, sync=False)  # Don't sync yet
                    branch_pads[b] = pad

                # STEP 4: Now sync camera bin state (with branches already linked)
                logger.info(f"[CAM-MANAGER] Syncing camera bin to pipeline state")

                if is_first_camera or prev_state == Gst.State.READY:
                    # First camera - sync to READY, then transition with pipeline
                    if not self._incremental_state_sync(bin_elem, Gst.State.READY):
                        logger.error(f"[CAM-MANAGER] Failed to sync camera bin to READY")
                elif prev_state == Gst.State.PLAYING:
                    # Additional camera - pipeline was paused in STEP 1
                    # Sync bin to PAUSED state (matching current pipeline state)
                    if not self._incremental_state_sync(bin_elem, Gst.State.PAUSED):
                        logger.warning(f"[CAM-MANAGER] Camera bin PAUSED sync warning")

                    # For file sources, wait for uridecodebin to link its pads BEFORE resuming pipeline
                    if is_file:
                        logger.info(f"[CAM-MANAGER] Waiting for uridecodebin pad linking...")
                        if pad_linked_event.wait(timeout=5.0):
                            logger.info(f"[CAM-MANAGER] uridecodebin pad linked successfully")
                            time.sleep(0.5)  # Brief stabilization after pad link
                        else:
                            logger.warning(f"[CAM-MANAGER] Timeout waiting for uridecodebin pad linking - proceeding anyway")
                    else:
                        time.sleep(1.0)  # Wait for decoder to initialize

                # STEP 5: Store camera info before state changes
                self._cameras[camera_id] = {
                    "bin": bin_elem, "tee": tee, "source_id": source_id,
                    "uri": uri, "branch_pads": branch_pads
                }

                # STEP 6: Transition pipeline and camera to PLAYING
                if is_first_camera or prev_state == Gst.State.READY:
                    # First camera - need to start entire pipeline
                    logger.info(f"[CAM-MANAGER] First camera - transitioning pipeline to PLAYING")

                    # Set pipeline to PAUSED first (load all elements including branches)
                    logger.info(f"[CAM-MANAGER] Setting pipeline to PAUSED for safe preroll")
                    self.pipeline.set_state(Gst.State.PAUSED)
                    ret, _, _ = self.pipeline.get_state(10 * Gst.SECOND)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.warning(f"[CAM-MANAGER] Pipeline PAUSED transition warning")

                    # Wait for nvurisrcbin to connect and negotiate caps
                    time.sleep(2.0)

                    # Now set to PLAYING
                    logger.info(f"[CAM-MANAGER] Setting pipeline to PLAYING")
                    self.pipeline.set_state(Gst.State.PLAYING)
                    ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.warning(f"[CAM-MANAGER] Pipeline PLAYING transition returned FAILURE (may still work)")

                elif prev_state == Gst.State.PLAYING:
                    # Additional camera - pipeline was PAUSED in STEP 1
                    logger.info(f"[CAM-MANAGER] Additional camera - resuming pipeline")

                    # Unified state transition order for both file and RTSP:
                    # 1. Resume pipeline to PLAYING first
                    # 2. Then sync camera bin to PLAYING (as child of pipeline)
                    # This prevents race conditions with nvstreammux
                    if is_file:
                        logger.info(f"[CAM-MANAGER] Resuming pipeline to PLAYING (file source)")
                        self.pipeline.set_state(Gst.State.PLAYING)
                        ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                        if ret == Gst.StateChangeReturn.FAILURE:
                            logger.warning(f"[CAM-MANAGER] Pipeline resume returned FAILURE (may still work)")

                        # Now sync camera bin incrementally (safer than direct set_state)
                        logger.info(f"[CAM-MANAGER] Syncing camera bin to PLAYING (file source)")
                        if not self._incremental_state_sync(bin_elem, Gst.State.PLAYING):
                            logger.warning(f"[CAM-MANAGER] Camera bin PLAYING sync warning")
                    else:
                        # For RTSP sources, resume pipeline first then sync camera bin
                        logger.info(f"[CAM-MANAGER] Resuming pipeline to PLAYING")
                        self.pipeline.set_state(Gst.State.PLAYING)
                        ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                        if ret == Gst.StateChangeReturn.FAILURE:
                            logger.warning(f"[CAM-MANAGER] Pipeline resume returned FAILURE (may still work)")

                        # Sync camera bin to PLAYING
                        logger.info(f"[CAM-MANAGER] Syncing camera bin to PLAYING")
                        if not self._incremental_state_sync(bin_elem, Gst.State.PLAYING):
                            logger.warning(f"[CAM-MANAGER] Camera bin PLAYING sync warning")

                    # Wait for camera to connect and stabilize
                    time.sleep(3.0)
                    logger.info(f"[CAM-MANAGER] Additional camera stabilized")

                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Successfully added {camera_id} to branches: {branches}")
                return True

            except Exception as e:
                logger.error(f"add_camera failed: {e}", exc_info=True)
                # Attempt cleanup
                try:
                    if camera_id in self._cameras:
                        del self._cameras[camera_id]
                    self.pipeline.remove(bin_elem)
                except:
                    pass
                self._mapper.remove(camera_id)
                # Restore pipeline state if needed
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                return False

    def remove_camera(self, camera_id: str) -> bool:
        """Remove camera from all branches with proper buffer drain.

        Uses blocking probes to stop data flow, then drains buffers before
        destroying decoder to prevent CUDA memory corruption.

        When removing the last camera, pauses pipeline first to prevent
        nvstreammux/nvinfer CUDA errors with empty input.
        """
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam:
                return False

            try:
                logger.info(f"[CAM-MANAGER] Removing camera {camera_id}...")

                # Check if this is the last camera - need special handling
                is_last_camera = len(self._cameras) == 1
                prev_state = Gst.State.NULL

                if is_last_camera:
                    # For last camera, set pipeline to NULL BEFORE removing
                    # This forces TensorRT/CUDA cleanup while camera is still linked
                    # preventing race conditions with buffer cleanup
                    _, prev_state, _ = self.pipeline.get_state(0)
                    logger.info(f"[CAM-MANAGER] Last camera - setting pipeline to NULL first")
                    self.pipeline.set_state(Gst.State.NULL)
                    self.pipeline.get_state(5 * Gst.SECOND)
                    # Wait for GPU cleanup to complete
                    time.sleep(0.5)

                # Step 1: Cleanup RTSP publisher first (before removing camera resources)
                if self._rtsp_publisher:
                    self._rtsp_publisher.cleanup_camera(camera_id)

                # Step 2: Add blocking probes on all branch pads (BLOCK, not REMOVE)
                # This stops new data from entering downstream while we drain
                probe_ids = {}
                for branch_name, pad in cam["branch_pads"].items():
                    pid = pad.add_probe(
                        Gst.PadProbeType.BLOCK_DOWNSTREAM,
                        lambda *_: Gst.PadProbeReturn.OK  # Block indefinitely
                    )
                    probe_ids[branch_name] = (pad, pid)
                    logger.debug(f"[CAM-MANAGER] Blocked {camera_id} on branch {branch_name}")

                # Step 3: Send EOS to camera bin to flush decoder buffers
                # EOS propagates through decoder, flushing all in-flight buffers
                cam["bin"].send_event(Gst.Event.new_eos())

                # Step 4: Wait for buffers to drain from decoder
                # nvv4l2decoder needs time to process remaining buffers
                time.sleep(0.8)

                # Step 5: Set bin to NULL state (closes decoder FDs safely)
                cam["bin"].set_state(Gst.State.NULL)
                ret, _, _ = cam["bin"].get_state(2 * Gst.SECOND)
                if ret != Gst.StateChangeReturn.SUCCESS:
                    logger.warning(f"[CAM-MANAGER] Camera bin NULL state change: {ret}")

                # Step 6: Remove blocking probes
                for branch_name, (pad, pid) in probe_ids.items():
                    try:
                        pad.remove_probe(pid)
                    except Exception:
                        pass  # Pad may already be unlinked

                # Step 7: Unlink from all branches
                for b in list(cam["branch_pads"].keys()):
                    self._unlink_branch(cam, b, camera_id)

                # Step 8: Remove bin from pipeline
                self.pipeline.remove(cam["bin"])
                self._mapper.remove(camera_id)
                del self._cameras[camera_id]
                self._last_op = time.time()

                # If this was the last camera, set back to READY for new cameras
                if is_last_camera:
                    logger.info(f"[CAM-MANAGER] All cameras removed - setting pipeline to READY")
                    self.pipeline.set_state(Gst.State.READY)
                    self.pipeline.get_state(5 * Gst.SECOND)
                    # Reset RTSP publisher (demux pads need re-request after NULL)
                    if self._rtsp_publisher:
                        self._rtsp_publisher.reset()

                logger.info(f"[CAM-MANAGER] Camera {camera_id} removed successfully")
                return True

            except Exception as e:
                logger.error(f"remove_camera failed: {e}", exc_info=True)
                return False

    def add_camera_to_branch(self, camera_id: str, branch_name: str) -> bool:
        """Add camera to additional branch without pausing pipeline.

        FIX (2026-01-21): Removed pipeline PAUSE to prevent RTSP stream failures.
        Adding a branch link (tee → queue → mux) can be done dynamically:
        - Tee src pad is requested dynamically
        - Queue is added with proper state sync
        - Mux sink pad is requested dynamically
        This preserves existing RTSP streams that would fail on PAUSE/RESUME.
        """
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam or branch_name in cam["branch_pads"] or branch_name not in self.branches:
                return False

            # Get current pipeline state for element sync
            _, prev_state, _ = self.pipeline.get_state(0)
            logger.info(f"[CAM-MANAGER] Adding {camera_id} to branch {branch_name} (no pause)")

            try:
                # NO PAUSE: Link branch dynamically with element state sync
                # This is safe because we're only adding new pads/elements
                pad = self._link_branch(cam["bin"], cam["tee"], camera_id, cam["source_id"], branch_name, True)
                cam["branch_pads"][branch_name] = pad

                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Successfully added {camera_id} to branch {branch_name}")
                return True
            except Exception as e:
                logger.error(f"add_camera_to_branch failed: {e}", exc_info=True)
                return False

    def remove_camera_from_branch(self, camera_id: str, branch_name: str) -> bool:
        """Remove camera from branch.

        STABILITY FIX (2026-01-20):
        - When this would leave a branch empty (while other branches have cameras),
          pause the pipeline to prevent CUDA race conditions with nvstreammux.
        - This is critical because nvstreammux with 0 sources triggers inference errors.
        """
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam:
                logger.warning(f"[CAM-MANAGER] remove_camera_from_branch: camera {camera_id} not found. Existing: {list(self._cameras.keys())}")
                return False
            if branch_name not in cam["branch_pads"]:
                logger.warning(f"[CAM-MANAGER] remove_camera_from_branch: camera {camera_id} not in branch {branch_name}. Branches: {list(cam['branch_pads'].keys())}")
                return False
            if len(cam["branch_pads"]) <= 1:
                logger.info(f"[CAM-MANAGER] remove_camera_from_branch: camera {camera_id} has only {len(cam['branch_pads'])} branch(s), use remove_camera instead")
                return False

            logger.info(f"[CAM-MANAGER] Starting to remove {camera_id} from branch {branch_name}")

            # Check if this will leave the branch empty
            b = self.branches.get(branch_name)
            branch_camera_count = 0
            if b and b.nvstreammux:
                it = b.nvstreammux.iterate_sink_pads()
                while True:
                    result, pad = it.next()
                    if result == Gst.IteratorResult.OK:
                        if pad.is_linked():
                            branch_camera_count += 1
                    elif result == Gst.IteratorResult.RESYNC:
                        it.resync()
                        branch_camera_count = 0
                    else:
                        break

            will_empty_branch = branch_camera_count <= 1
            _, prev_state, _ = self.pipeline.get_state(0)

            try:
                # STABILITY FIX: Pause pipeline if this will empty the branch
                # This prevents CUDA race conditions when nvstreammux becomes empty
                if will_empty_branch and prev_state == Gst.State.PLAYING:
                    logger.info(f"[CAM-MANAGER] Pausing pipeline - branch {branch_name} will become empty")
                    self.pipeline.set_state(Gst.State.PAUSED)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    time.sleep(0.5)

                tee_pad = cam["branch_pads"].get(branch_name)

                # Step 1: Block the tee src pad to stop data flow
                blocked = threading.Event()
                probe_id = None

                def block_probe(pad, info):
                    blocked.set()
                    return Gst.PadProbeReturn.OK  # Keep blocking until we remove the probe

                if tee_pad:
                    logger.debug(f"[CAM-MANAGER] Adding block probe to tee pad for {camera_id}/{branch_name}")
                    probe_id = tee_pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, block_probe)
                    # Wait for probe to trigger (data flow blocked)
                    blocked.wait(timeout=1.0)

                logger.debug(f"[CAM-MANAGER] Data flow blocked, proceeding with unlink")

                # Step 2: Unlink the branch elements (safe now that data is blocked)
                self._unlink_branch_safe(cam, branch_name, camera_id, tee_pad, probe_id)

                # Resume pipeline if we paused it
                if will_empty_branch and prev_state == Gst.State.PLAYING:
                    logger.info(f"[CAM-MANAGER] Resuming pipeline after branch removal")
                    self.pipeline.set_state(Gst.State.PLAYING)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)

                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Successfully removed {camera_id} from branch {branch_name}")
                return True
            except Exception as e:
                logger.error(f"[CAM-MANAGER] remove_camera_from_branch failed: {e}", exc_info=True)
                # Try to restore pipeline state
                if will_empty_branch and prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                return False

    def _link_branch(self, bin_elem, tee, camera_id, source_id, branch_name, sync=False) -> Gst.Pad:
        """Link: tee -> queue -> mux.

        When sync=True, elements are synced to parent state after creation.
        """
        b = self.branches[branch_name]

        q = Gst.ElementFactory.make("queue", f"q_{camera_id}_{branch_name}")
        q.set_property("max-size-buffers", 30)
        q.set_property("max-size-bytes", 0)
        q.set_property("max-size-time", 0)
        q.set_property("leaky", 2)
        bin_elem.add(q)

        tee_src = tee.request_pad_simple("src_%u")
        tee_src.link(q.get_static_pad("sink"))

        mux_sink = b.nvstreammux.request_pad_simple(f"sink_{source_id}")
        ghost = Gst.GhostPad.new(f"g_{branch_name}_{camera_id}", q.get_static_pad("src"))
        bin_elem.add_pad(ghost)
        ghost.link(mux_sink)

        # Sync element states if requested (for dynamic branch addition)
        if sync:
            _, parent_state, _ = bin_elem.get_state(0)
            self._incremental_state_sync(q, parent_state)

        return tee_src

    def _unlink_branch_safe(self, cam: dict, branch_name: str, camera_id: str,
                            tee_pad: Optional[Gst.Pad] = None, probe_id: Optional[int] = None) -> None:
        """Safely unlink and cleanup branch elements with proper synchronization.

        This method should be called when data flow is already blocked.
        """
        b = self.branches.get(branch_name)
        if not b:
            logger.warning(f"_unlink_branch_safe: branch {branch_name} not found")
            return

        # Get elements to remove
        elements = []
        elem = cam["bin"].get_by_name(f"q_{camera_id}_{branch_name}")
        if elem:
            elements.append(elem)

        # Step 1: Unlink from nvstreammux first (while data is blocked)
        ghost_pad = None
        mux_pad = None
        it = cam["bin"].iterate_pads()
        while True:
            ret, pad = it.next()
            # Look for ghost pad with this specific camera_id pattern
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

        # Step 2: Unlink tee from queue
        if tee_pad and elements:
            q_elem = cam["bin"].get_by_name(f"q_{camera_id}_{branch_name}")
            if q_elem:
                q_sink = q_elem.get_static_pad("sink")
                if q_sink and q_sink.is_linked():
                    peer = q_sink.get_peer()
                    if peer:
                        peer.unlink(q_sink)

        # Step 3: Remove probe (allow any pending data to flush)
        if tee_pad and probe_id is not None:
            tee_pad.remove_probe(probe_id)

        # Small delay to let any in-flight data clear
        time.sleep(0.1)

        # Step 4: Set elements to NULL state and remove them
        for elem in elements:
            elem.set_state(Gst.State.NULL)
            elem.get_state(Gst.CLOCK_TIME_NONE)

        time.sleep(0.05)

        for elem in elements:
            cam["bin"].remove(elem)

        # Step 5: Remove ghost pad from bin
        if ghost_pad:
            cam["bin"].remove_pad(ghost_pad)

        # Step 6: Release the nvstreammux sink pad
        if mux_pad:
            try:
                b.nvstreammux.release_request_pad(mux_pad)
            except Exception as e:
                logger.warning(f"release nvstreammux pad failed: {e}")

        # Step 7: Release the tee src pad
        if tee_pad:
            try:
                cam["tee"].release_request_pad(tee_pad)
            except Exception as e:
                logger.warning(f"release tee pad failed: {e}")

        cam["branch_pads"].pop(branch_name, None)
        logger.info(f"Safely unlinked {camera_id} from branch {branch_name}")

    def _unlink_branch(self, cam: dict, branch_name: str, camera_id: str) -> None:
        """Unlink and cleanup branch elements (legacy - for remove_camera)."""
        b = self.branches.get(branch_name)
        if not b:
            logger.warning(f"_unlink_branch: branch {branch_name} not found")
            return

        tee_pad = cam["branch_pads"].get(branch_name)

        elem = cam["bin"].get_by_name(f"q_{camera_id}_{branch_name}")
        if elem:
            elem.set_state(Gst.State.NULL)

        time.sleep(0.15)

        elem = cam["bin"].get_by_name(f"q_{camera_id}_{branch_name}")
        if elem:
            cam["bin"].remove(elem)

        it = cam["bin"].iterate_pads()
        while True:
            ret, pad = it.next()
            if ret == Gst.IteratorResult.OK and pad.get_name() == f"g_{branch_name}_{camera_id}":
                peer = pad.get_peer()
                if peer:
                    pad.unlink(peer)
                    b.nvstreammux.release_request_pad(peer)
                cam["bin"].remove_pad(pad)
                break
            elif ret == Gst.IteratorResult.RESYNC:
                it.resync()
            elif ret != Gst.IteratorResult.OK:
                break

        if tee_pad:
            try:
                cam["tee"].release_request_pad(tee_pad)
            except Exception as e:
                logger.warning(f"release_request_pad failed: {e}")

        cam["branch_pads"].pop(branch_name, None)
        logger.info(f"Unlinked {camera_id} from branch {branch_name}")

    # Query methods
    def list_cameras(self) -> dict:
        with self._lock:
            return {k: {"uri": v["uri"], "source_id": v["source_id"], "branches": list(v["branch_pads"].keys())}
                    for k, v in self._cameras.items()}

    def get_camera(self, camera_id: str) -> Optional[dict]:
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
            return list(cam["branch_pads"].keys()) if cam else []

    def get_mapper(self) -> SourceIDMapper:
        """Get SourceIDMapper for probe lookups."""
        return self._mapper

    def kill_all(self) -> int:
        """Remove all cameras - requires pipeline restart to add cameras again.

        STABILITY FIX (2026-01-20):
        - Send EOS event to nvstreammux to signal end of stream
        - Wait for EOS to propagate (allows CUDA operations to complete)
        - Use incremental state transition: PLAYING -> PAUSED -> NULL
        - Add probes to block data flow before state change
        This prevents CUDA race conditions during cleanup.
        """
        with self._lock:
            if not self._cameras:
                return 0

            n = len(self._cameras)
            logger.info(f"[CAM-MANAGER] kill_all starting - removing {n} cameras")

            # STEP 1: Get current state
            _, current_state, _ = self.pipeline.get_state(0)
            logger.info(f"[CAM-MANAGER] Current pipeline state: {current_state.value_nick}")

            # STEP 2: Send EOS to all nvstreammux elements to signal end of stream
            # This allows inference engines to complete their CUDA operations gracefully
            logger.info("[CAM-MANAGER] Sending EOS to all branches...")
            for branch_name, branch in self.branches.items():
                if branch.nvstreammux:
                    # Send EOS downstream from nvstreammux
                    branch.nvstreammux.send_event(Gst.Event.new_eos())
                    logger.debug(f"[CAM-MANAGER] Sent EOS to {branch_name} nvstreammux")

            # Wait for EOS to propagate
            time.sleep(2.0)

            # STEP 3: Add blocking probes to all camera tees to stop any remaining data flow
            block_probes = []
            for camera_id, cam in self._cameras.items():
                tee = cam.get("tee")
                if tee:
                    sink_pad = tee.get_static_pad("sink")
                    if sink_pad:
                        def make_block_probe():
                            def probe_fn(pad, info):
                                return Gst.PadProbeReturn.DROP
                            return probe_fn

                        probe_fn = make_block_probe()
                        probe_id = sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_fn)
                        block_probes.append((sink_pad, probe_id))

            time.sleep(0.5)

            # STEP 4: Transition to PAUSED first
            if current_state == Gst.State.PLAYING:
                logger.info("[CAM-MANAGER] Transitioning to PAUSED...")
                self.pipeline.set_state(Gst.State.PAUSED)
                ret, _, _ = self.pipeline.get_state(10 * Gst.SECOND)
                if ret == Gst.StateChangeReturn.FAILURE:
                    logger.warning("[CAM-MANAGER] PAUSED transition warning")
                time.sleep(2.0)  # Allow CUDA operations to complete

            # STEP 5: Transition to NULL state
            logger.info("[CAM-MANAGER] Transitioning to NULL...")
            self.pipeline.set_state(Gst.State.NULL)
            ret, _, _ = self.pipeline.get_state(10 * Gst.SECOND)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning("[CAM-MANAGER] NULL transition warning")
            time.sleep(2.0)  # Allow CUDA resources to be fully released

            # STEP 6: Remove blocking probes
            for sink_pad, probe_id in block_probes:
                try:
                    sink_pad.remove_probe(probe_id)
                except Exception:
                    pass

            # STEP 7: Cleanup camera bins (skip unlink since pipeline is already NULL)
            for camera_id, cam in list(self._cameras.items()):
                try:
                    # Since pipeline is NULL, just remove the bin directly
                    cam["bin"].set_state(Gst.State.NULL)
                    self.pipeline.remove(cam["bin"])
                    logger.debug(f"[CAM-MANAGER] Removed camera bin: {camera_id}")
                except Exception as e:
                    logger.warning(f"[CAM-MANAGER] kill_all cleanup error for {camera_id}: {e}")

            self._cameras.clear()
            self._mapper.clear()

            # STEP 8: Transition pipeline back to READY (waiting for new cameras)
            logger.info("[CAM-MANAGER] Transitioning to READY...")
            self.pipeline.set_state(Gst.State.READY)
            ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning("[CAM-MANAGER] READY transition warning")

            self._last_op = time.time()
            logger.info(f"[CAM-MANAGER] kill_all completed - removed {n} cameras")
            return n
