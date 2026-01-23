"""Pipeline control and system endpoints."""

import traceback
from fastapi import APIRouter
from gi.repository import Gst


def create_router(server) -> APIRouter:
    """Create pipeline router with server context."""
    router = APIRouter(prefix="/api", tags=["pipeline"])

    @router.get("/health")
    async def health():
        """Quick health check endpoint."""
        return {
            "status": "healthy",
            "cameras": server.manager.count(),
            "branches": list(server.manager.branches.keys())
        }

    @router.get("/status")
    async def get_status():
        """Detailed pipeline status with error information."""
        # Get pipeline state
        pipeline = server.manager.pipeline
        ret, state, pending = pipeline.get_state(0)

        state_str = state.value_nick if state else "unknown"
        pending_str = pending.value_nick if pending != Gst.State.VOID_PENDING else None

        # Get camera details
        cameras_detail = {}
        for cam_id, cam_info in server.manager._cameras.items():
            bin_state = cam_info.bin.get_state(0)[1]
            cameras_detail[cam_id] = {
                "uri": cam_info.uri,
                "source_id": cam_info.source_id,
                "branches": list(cam_info.branch_pads.keys()),
                "state": bin_state.value_nick if bin_state else "unknown",
                "is_file": cam_info.is_file
            }

        # Get branch details
        branches_detail = {}
        for name, info in server.manager.branches.items():
            cams = [cid for cid, cam in server.manager._cameras.items() if name in cam.branch_pads]
            branches_detail[name] = {
                "max_cameras": info.max_cameras,
                "current_cameras": len(cams),
                "cameras": cams
            }

        # Get stream status
        streams = {}
        if server.stream_publisher:
            streams = server.stream_publisher.get_status()

        # Get operation queue status
        queue_size = server._op_queue.qsize()
        pending_ops = list(server._op_results.keys())[-10:]  # Last 10 operations

        return {
            "timestamp": __import__("time").time(),
            "pipeline": {
                "state": state_str,
                "pending_state": pending_str,
                "state_change_result": ret.value_nick if ret else "unknown"
            },
            "cameras": {
                "count": len(cameras_detail),
                "details": cameras_detail
            },
            "branches": branches_detail,
            "streams": streams,
            "operations": {
                "queue_size": queue_size,
                "recent_operations": pending_ops
            },
            "system": {
                "last_operation_type": server._last_op_type,
                "last_operation_time": server._last_op_time
            }
        }

    @router.get("/branches")
    async def list_branches():
        branches = {}
        for name, info in server.manager.branches.items():
            cams = [cid for cid, cam in server.manager._cameras.items() if name in cam.branch_pads]
            branches[name] = {"max_cameras": info.max_cameras, "cameras": cams}
        return {"branches": branches}

    @router.get("/operations/{op_id}")
    async def get_operation(op_id: str):
        from fastapi import HTTPException
        if op_id in server._op_results:
            return server._op_results[op_id]
        raise HTTPException(404, "Operation not found")

    @router.post("/pipeline/kill")
    async def kill_pipeline():
        op_id = server._enqueue("remove_camera", server.manager.kill_all)
        return {"status": "accepted", "operation_id": op_id}

    @router.post("/pipeline/stop")
    async def stop_pipeline():
        server._op_queue.put(("stop", "default", server.manager.kill_all, (), {}))
        server.shutdown_event.set()
        return {"status": "ok", "message": "shutdown"}

    return router
