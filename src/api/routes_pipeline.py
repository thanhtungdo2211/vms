"""Pipeline control and system endpoints."""

from fastapi import APIRouter


def create_router(server) -> APIRouter:
    """Create pipeline router with server context."""
    router = APIRouter(prefix="/api", tags=["pipeline"])

    @router.get("/health")
    async def health():
        return {
            "status": "healthy",
            "cameras": server.manager.count(),
            "branches": list(server.manager.branches.keys())
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
