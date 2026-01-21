"""
CameraAPIServer - REST API for MultibranchCameraManager using FastAPI

Endpoints:
- POST   /api/cameras                              - Add camera to branches
- DELETE /api/cameras/{camera_id}                  - Remove camera entirely
- POST   /api/cameras/{camera_id}/branches/{name}  - Add to branch
- DELETE /api/cameras/{camera_id}/branches/{name}  - Remove from branch
- GET    /api/cameras                              - List all cameras
- GET    /api/branches                             - List all branches
- GET    /api/health                               - Health check
- GET    /api/operations/{op_id}                   - Get operation status
- POST   /api/pipeline/kill                        - Remove all cameras
- POST   /api/pipeline/stop                        - Stop pipeline
- POST   /api/cameras/{id}/branches/{branch}/rtsp/start  - Start per-camera RTSP
- POST   /api/cameras/{id}/branches/{branch}/rtsp/stop   - Stop per-camera RTSP
- GET    /api/cameras/{id}/branches/{branch}/rtsp/status - Get per-camera RTSP status
- GET    /api/cameras/rtsp/status                  - Get all per-camera RTSP status
"""

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from api.shutdown import stop_event

logger = logging.getLogger(__name__)


class CameraAPIServer:
    # Operation delays for pipeline stabilization
    OP_DELAYS = {
        "add_camera": 3.0,       # Adding camera needs GPU init time
        "remove_camera": 2.5,    # Removing needs cleanup time
        "add_branch": 2.0,       # Adding to branch
        "remove_branch": 2.0,    # Removing from branch
        "rtsp_start": 2.5,       # RTSP start needs encoder init
        "rtsp_stop": 1.5,        # RTSP stop is faster
        "default": 2.0,          # Default delay
    }
    MIN_OP_DELAY = 2.0  # Minimum delay between any operations

    def __init__(
        self,
        cfg,
        manager,
        demux_rtsp_publisher=None,
    ):
        self.manager = manager
        self.demux_rtsp_publisher = demux_rtsp_publisher
        self.host = cfg.get("host", "0.0.0.0")
        self.port = cfg.get("port", 8083)
        self.shutdown_event = stop_event
        self.op_queue = queue.Queue()
        self.op_results = {}
        self.op_lock = threading.Lock()
        self._running = True
        self._app = None
        self._last_op_type = None

    def _get_op_delay(self, op_type: str) -> float:
        """Get delay for operation type, considering previous operation."""
        base_delay = self.OP_DELAYS.get(op_type, self.OP_DELAYS["default"])

        # Extra delay after heavy operations
        if self._last_op_type in ("add_camera", "remove_camera"):
            base_delay = max(base_delay, 2.5)

        # Extra delay for consecutive RTSP operations
        if op_type == "rtsp_start" and self._last_op_type == "rtsp_start":
            base_delay = max(base_delay, 5.0)

        # Extra delay for RTSP after branch modifications (GPU needs more time to stabilize)
        if op_type == "rtsp_start" and self._last_op_type in ("add_branch", "remove_branch"):
            base_delay = max(base_delay, 6.0)

        return base_delay

    def _process_ops(self):
        while self._running:
            try:
                op_id, op_type, func, args, kwargs = self.op_queue.get(timeout=0.2)
                with self.op_lock:
                    now = time.time()
                    elapsed = now - self.last_op
                    delay = self._get_op_delay(op_type)
                    if elapsed < delay:
                        time.sleep(delay - elapsed)
                    self.last_op = time.time()
                    self._last_op_type = op_type

                try:
                    result = func(*args, **kwargs)
                    self.op_results[op_id] = {"status": "ok", "result": result}
                except Exception as e:
                    self.op_results[op_id] = {"status": "error", "error": str(e)}
                self.op_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Camera API")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class AddCameraRequest(BaseModel):
            camera_id: str
            uri: str
            branches: List[str] = []

        @app.get("/api/health")
        async def health():
            count = self.manager.count()
            branches = list(self.manager.branches.keys())
            return {"status": "healthy", "cameras": count, "branches": branches}

        @app.get("/api/cameras")
        async def list_cameras():
            cameras = self.manager.list_cameras()
            return {"cameras": cameras}

        @app.get("/api/branches")
        async def list_branches():
            branches = {}
            for name, info in self.manager.branches.items():
                cams = [cid for cid, cam in self.manager._cameras.items() if name in cam.get("branch_pads", {})]
                branches[name] = {"max_cameras": info.max_cameras, "current_cameras": cams}
            return {"branches": branches}

        @app.get("/api/operations/{op_id}")
        async def get_operation(op_id: str):
            result = self.op_results.get(op_id)
            if result:
                return result
            raise HTTPException(status_code=404, detail="operation not found")

        @app.post("/api/cameras")
        async def add_camera(request: AddCameraRequest):
            import uuid
            camera_id = request.camera_id
            uri = request.uri
            branches = request.branches
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, "add_camera", self.manager.add_camera, (camera_id, uri, branches), {}))
            return {"status": "accepted", "operation_id": op_id}

        @app.post("/api/pipeline/kill")
        async def kill_pipeline():
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, "remove_camera", self.manager.kill_all, (), {}))
            return {"status": "accepted", "operation_id": op_id}

        @app.post("/api/pipeline/stop")
        async def stop_pipeline():
            self.op_queue.put(("stop", "default", self.manager.kill_all, (), {}))
            if self.shutdown_event:
                self.shutdown_event.set()
            return {"status": "ok", "message": "shutdown"}

        @app.delete("/api/cameras/{camera_id}")
        async def remove_camera(camera_id: str):
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, "remove_camera", self.manager.remove_camera, (camera_id,), {}))
            return {"status": "accepted", "operation_id": op_id}

        @app.post("/api/cameras/{camera_id}/branches/{branch_name}")
        async def add_camera_to_branch(camera_id: str, branch_name: str):
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, "add_branch", self.manager.add_camera_to_branch, (camera_id, branch_name), {}))
            return {"status": "accepted", "operation_id": op_id}

        @app.delete("/api/cameras/{camera_id}/branches/{branch_name}")
        async def remove_camera_from_branch(camera_id: str, branch_name: str):
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((op_id, "remove_branch", self.manager.remove_camera_from_branch, (camera_id, branch_name), {}))
            return {"status": "accepted", "operation_id": op_id}

        # RTSP Publishing Endpoints
        class RtspStartRequest(BaseModel):
            location: str
            bitrate: int = 4000000

        # Per-Camera RTSP Publishing Endpoints
        @app.post("/api/cameras/{camera_id}/branches/{branch_name}/rtsp/start")
        async def start_per_camera_rtsp(camera_id: str, branch_name: str, request: RtspStartRequest):
            if not self.demux_rtsp_publisher:
                raise HTTPException(status_code=503, detail="Per-camera RTSP publisher not available")
            if not self.manager.has_camera(camera_id):
                raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
            if branch_name not in self.manager.branches:
                raise HTTPException(status_code=404, detail=f"Branch {branch_name} not found")
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((
                op_id,
                "rtsp_start",
                self.demux_rtsp_publisher.start_publish,
                (camera_id, branch_name, request.location, request.bitrate),
                {}
            ))
            return {"status": "accepted", "operation_id": op_id}

        @app.post("/api/cameras/{camera_id}/branches/{branch_name}/rtsp/stop")
        async def stop_per_camera_rtsp(camera_id: str, branch_name: str):
            if not self.demux_rtsp_publisher:
                raise HTTPException(status_code=503, detail="Per-camera RTSP publisher not available")
            import uuid
            op_id = str(uuid.uuid4())[:8]
            self.op_queue.put((
                op_id,
                "rtsp_stop",
                self.demux_rtsp_publisher.stop_publish,
                (camera_id, branch_name),
                {}
            ))
            return {"status": "accepted", "operation_id": op_id}

        @app.get("/api/cameras/{camera_id}/branches/{branch_name}/rtsp/status")
        async def get_per_camera_rtsp_status(camera_id: str, branch_name: str):
            if not self.demux_rtsp_publisher:
                return {"publishing": False, "error": "Per-camera RTSP publisher not available"}
            return self.demux_rtsp_publisher.get_status(camera_id, branch_name)

        @app.get("/api/cameras/rtsp/status")
        async def get_all_per_camera_rtsp_status():
            if not self.demux_rtsp_publisher:
                return {"error": "Per-camera RTSP publisher not available"}
            return self.demux_rtsp_publisher.get_status()

        return app

    def start(self):
        self._op_thread = threading.Thread(target=self._process_ops, daemon=True)
        self._op_thread.start()
        self._app = self._create_app()
        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config=config)
        threading.Thread(target=self._server.run, daemon=True).start()
        logger.info(f"[CameraAPI] Server running at http://{self.host}:{self.port}")

    def stop(self):
        self._running = False

    @property
    def last_op(self):
        return self._last_op if hasattr(self, '_last_op') else 0.0

    @last_op.setter
    def last_op(self, value):
        self._last_op = value

    @property
    def min_delay(self):
        return self.MIN_OP_DELAY
