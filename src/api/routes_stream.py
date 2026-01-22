"""Stream publishing endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class StreamRequest(BaseModel):
    uri: str
    bitrate: int = 4000000


def create_router(server) -> APIRouter:
    """Create stream router with server context."""
    router = APIRouter(prefix="/api", tags=["streams"])

    @router.post("/cameras/{camera_id}/branches/{branch}/stream/start")
    async def start_stream(camera_id: str, branch: str, req: StreamRequest):
        if not server.stream_publisher:
            raise HTTPException(503, "Stream publisher not available")
        if not server.manager.has_camera(camera_id):
            raise HTTPException(404, f"Camera {camera_id} not found")
        if branch not in server.manager.branches:
            raise HTTPException(404, f"Branch {branch} not found")
        op_id = server._enqueue("stream_start", server.stream_publisher.start_publish,
                                camera_id, branch, req.uri, req.bitrate)
        return {"status": "accepted", "operation_id": op_id}

    @router.post("/cameras/{camera_id}/branches/{branch}/stream/stop")
    async def stop_stream(camera_id: str, branch: str):
        if not server.stream_publisher:
            raise HTTPException(503, "Stream publisher not available")
        op_id = server._enqueue("stream_stop", server.stream_publisher.stop_publish, camera_id, branch)
        return {"status": "accepted", "operation_id": op_id}

    @router.get("/cameras/{camera_id}/branches/{branch}/stream")
    async def get_stream_status(camera_id: str, branch: str):
        if not server.stream_publisher:
            return {"publishing": False, "error": "Stream publisher not available"}
        return server.stream_publisher.get_status(camera_id, branch)

    @router.get("/streams")
    async def list_streams():
        if not server.stream_publisher:
            return {"error": "Stream publisher not available"}
        return server.stream_publisher.get_status()

    return router
