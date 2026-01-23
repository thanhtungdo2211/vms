"""Camera CRUD and branch management endpoints."""

from typing import List
from fastapi import APIRouter
from pydantic import BaseModel


class AddCameraRequest(BaseModel):
    camera_id: str
    uri: str
    branch: str


def create_router(server) -> APIRouter:
    """Create camera router with server context."""
    router = APIRouter(prefix="/api", tags=["cameras"])

    @router.get("/cameras")
    async def list_cameras():
        return {"cameras": server.manager.list_cameras()}

    @router.post("/cameras")
    async def add_camera(req: AddCameraRequest):
        op_id = server._enqueue("add_camera", server.manager.add_camera, req.camera_id, req.uri, req.branch)
        return {"status": "accepted", "operation_id": op_id}

    @router.delete("/cameras/{camera_id}")
    async def remove_camera(camera_id: str):
        op_id = server._enqueue("remove_camera", server.manager.remove_camera, camera_id)
        return {"status": "accepted", "operation_id": op_id}

    @router.post("/cameras/{camera_id}/branches/{branch}")
    async def add_to_branch(camera_id: str, branch: str):
        op_id = server._enqueue("add_branch", server.manager.add_camera_to_branch, camera_id, branch)
        return {"status": "accepted", "operation_id": op_id}

    @router.delete("/cameras/{camera_id}/branches/{branch}")
    async def remove_from_branch(camera_id: str, branch: str):
        op_id = server._enqueue("remove_branch", server.manager.remove_camera_from_branch, camera_id, branch)
        return {"status": "accepted", "operation_id": op_id}

    return router
