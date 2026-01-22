"""Camera API Server - Main entry point."""

import logging
import queue
import signal
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api import routes_pipeline, routes_camera, routes_stream

logger = logging.getLogger(__name__)

_stop_event = threading.Event()


def _setup_signal_handlers():
    def on_shutdown(signum, frame):
        print(f"\n[Shutdown] Signal {signum} received...")
        _stop_event.set()
    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)


class CameraAPIServer:
    """REST API server for camera and stream management."""

    OP_DELAYS = {
        "add_camera": 3.0,
        "remove_camera": 2.5,
        "add_branch": 2.0,
        "remove_branch": 2.0,
        "stream_start": 2.5,
        "stream_stop": 1.5,
        "default": 2.0,
    }

    def __init__(self, cfg: dict, manager, stream_publisher=None):
        self.manager = manager
        self.stream_publisher = stream_publisher
        self.host = cfg.get("host", "0.0.0.0")
        self.port = cfg.get("port", 8083)
        self.shutdown_event = _stop_event

        self._op_queue = queue.Queue()
        self._op_results = {}
        self._op_lock = threading.Lock()
        self._last_op_time = 0.0
        self._last_op_type = None
        self._running = True
        self._app = None

    def _get_delay(self, op_type: str) -> float:
        delay = self.OP_DELAYS.get(op_type, self.OP_DELAYS["default"])
        if self._last_op_type in ("add_camera", "remove_camera"):
            delay = max(delay, 2.5)
        if op_type == "stream_start":
            if self._last_op_type == "stream_start":
                delay = max(delay, 5.0)
            elif self._last_op_type in ("add_branch", "remove_branch"):
                delay = max(delay, 6.0)
        return delay

    def _process_ops(self):
        while self._running:
            try:
                op_id, op_type, func, args, kwargs = self._op_queue.get(timeout=0.2)
                with self._op_lock:
                    elapsed = time.time() - self._last_op_time
                    delay = self._get_delay(op_type)
                    if elapsed < delay:
                        time.sleep(delay - elapsed)
                    self._last_op_time = time.time()
                    self._last_op_type = op_type
                try:
                    result = func(*args, **kwargs)
                    self._op_results[op_id] = {"status": "ok", "result": result}
                except Exception as e:
                    self._op_results[op_id] = {"status": "error", "error": str(e)}
                self._op_queue.task_done()
            except queue.Empty:
                continue

    def _enqueue(self, op_type: str, func, *args, **kwargs) -> str:
        op_id = str(uuid.uuid4())[:8]
        self._op_queue.put((op_id, op_type, func, args, kwargs))
        return op_id

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Camera API")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routers
        app.include_router(routes_pipeline.create_router(self))
        app.include_router(routes_camera.create_router(self))
        app.include_router(routes_stream.create_router(self))

        return app

    def start(self):
        _setup_signal_handlers()
        threading.Thread(target=self._process_ops, daemon=True).start()
        self._app = self._create_app()
        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config=config)
        threading.Thread(target=self._server.run, daemon=True).start()
        logger.info(f"[CameraAPI] http://{self.host}:{self.port}")

    def stop(self):
        self._running = False
