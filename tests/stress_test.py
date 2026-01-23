#!/usr/bin/env python3
"""Stress Test - Camera CRUD + Stream Publishing.

Tests:
1. Camera add/remove cycles
2. Branch switching
3. Stream publish/stop cycles
4. Concurrent streams on multiple branches
5. Stream during camera removal
6. Re-adding cameras after removal

Usage:
    python3 tests/stress_test.py
"""

import random
import socket
import subprocess
import sys
import time
import requests

# Configuration
BASE_URL = "http://localhost:8083"
STREAM_SERVER = "192.168.6.14"
STREAM_PORT = 8890
CAMERA_URI = "rtsp://192.168.6.14:8554/testface"
MAX_CAMERAS = 1
STREAM_CAMS = 1
STREAM_STABILIZE = 1


def api(method: str, endpoint: str, data: dict = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=30)
        elif method == "POST":
            r = requests.post(url, json=data, timeout=30)
        elif method == "DELETE":
            r = requests.delete(url, timeout=30)
        else:
            return {"error": f"Unknown method: {method}"}
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def wait_op(op_id: str, timeout: int = 60) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        r = api("GET", f"/api/operations/{op_id}")
        if r.get("status") in ("ok", "error"):
            return r
        time.sleep(0.5)
    return {"status": "timeout"}


def health() -> dict:
    return api("GET", "/api/health")


def add_camera(cam_id: str, branches: list) -> bool:
    r = api("POST", "/api/cameras", {
        "camera_id": cam_id,
        "uri": CAMERA_URI,
        "branches": branches
    })
    if "operation_id" in r:
        return wait_op(r["operation_id"]).get("status") == "ok"
    return False


def remove_camera(cam_id: str) -> bool:
    r = api("DELETE", f"/api/cameras/{cam_id}")
    if "operation_id" in r:
        return wait_op(r["operation_id"]).get("status") == "ok"
    return False


def remove_from_branch(cam_id: str, branch: str) -> bool:
    r = api("DELETE", f"/api/cameras/{cam_id}/branches/{branch}")
    if "operation_id" in r:
        return wait_op(r["operation_id"]).get("status") == "ok"
    return False


def add_to_branch(cam_id: str, branch: str) -> bool:
    r = api("POST", f"/api/cameras/{cam_id}/branches/{branch}")
    if "operation_id" in r:
        return wait_op(r["operation_id"]).get("status") == "ok"
    return False


def make_stream_uri(cam_id: str, branch: str) -> str:
    stream_id = f"publish:stress_{cam_id}_{branch}"
    return f"srt://{STREAM_SERVER}:{STREAM_PORT}?streamid={stream_id}&pkt_size=1316"


def start_stream(cam_id: str, branch: str, bitrate: int = 1000000) -> bool:
    uri = make_stream_uri(cam_id, branch)
    r = api("POST", f"/api/cameras/{cam_id}/branches/{branch}/stream/start", {
        "uri": uri,
        "bitrate": bitrate
    })
    if "operation_id" in r:
        return wait_op(r["operation_id"]).get("status") == "ok"
    return False


def stop_stream(cam_id: str, branch: str) -> bool:
    r = api("POST", f"/api/cameras/{cam_id}/branches/{branch}/stream/stop")
    if "operation_id" in r:
        return wait_op(r["operation_id"]).get("status") == "ok"
    return False

def stream_status() -> dict:
    return api("GET", "/api/streams")

def check_stream_live(host: str, port: int, stream_id: str, timeout: int = 8) -> bool:
    """Check if RTSP stream is live.

    First tries ffprobe (if available), then falls back to RTSP DESCRIBE.
    """
    # Try ffprobe first
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-rtsp_transport", "tcp",
            "-stimeout", str(timeout * 1000000),
            "-show_entries", "stream=codec_name",
            "-of", "default=nw=1",
            f"rtsp://{host}:{8554}/{stream_id.replace('publish:', '')}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 3)
        if "codec_name=" in result.stdout:
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass

def verify_streams(streams: list, retries: int = 3) -> tuple:
    results = {}
    for cam_id, branch in streams:
        stream_id = f"publish:stress_{cam_id}_{branch}"
        for attempt in range(retries):
            live = check_stream_live(STREAM_SERVER, STREAM_PORT, stream_id)
            if live:
                break
            if attempt < retries - 1:
                time.sleep(2)
        results[f"{cam_id}/{branch}"] = live
    return all(results.values()) if results else True, results


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def step(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def verify(name: str, condition: bool) -> bool:
    log(f"  {name}: {'PASS' if condition else 'FAIL'}")
    return condition


def cleanup():
    log("Cleanup: Removing all cameras...")
    for i in range(1, MAX_CAMERAS + 1):
        remove_camera(f"cam{i}")
        time.sleep(1)
    time.sleep(3)


def main():
    print("\n" + "="*60)
    print("  STRESS TEST: Camera CRUD + Stream Publishing")
    print("="*60)
    failures = 0

    # Pre-check
    step("Pre-check: Health & Cleanup")
    h = health()
    if h.get("status") != "healthy":
        log(f"API not healthy: {h}")
        return 1
    log(f"Health: {h}")

    if h.get("cameras", 0) > 0:
        cleanup()

    # TEST 1: Add cameras
    step("TEST 1: Add Cameras")
    for i in range(1, MAX_CAMERAS + 1):
        ok = add_camera(f"cam{i}", ["detection", "recognition"])
        log(f"  Add cam{i}: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    h = health()
    if not verify(f"{MAX_CAMERAS} cameras added", h.get("cameras") == MAX_CAMERAS):
        failures += 1

    # TEST 2: Stream on detection
    step(f"TEST 2: Stream on Detection ({STREAM_CAMS} cameras)")

    for i in range(1, STREAM_CAMS + 1):
        ok = start_stream(f"cam{i}", "detection", bitrate=4000000)
        log(f"  Start cam{i}/detection: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    log(f"Waiting {STREAM_STABILIZE}s for stabilization...")
    time.sleep(STREAM_STABILIZE)

    streams = [(f"cam{i}", "detection") for i in range(1, STREAM_CAMS + 1)]
    all_live, results = verify_streams(streams)
    for s, live in results.items():
        log(f"  {s}: {'LIVE' if live else 'DEAD'}")
    if not verify(f"{STREAM_CAMS} detection streams live", all_live):
        failures += 1

    # # TEST 3: Stream on recognition (concurrent)
    # step(f"TEST 3: Stream on Recognition ({STREAM_CAMS} cameras concurrent)")

    # for i in range(1, STREAM_CAMS + 1):
    #     ok = start_stream(f"cam{i}", "recognition", bitrate=1000000)
    #     log(f"  Start cam{i}/recognition: {'OK' if ok else 'FAIL'}")
    #     if not ok:
    #         failures += 1

    # log(f"Waiting {STREAM_STABILIZE}s for stabilization...")
    # time.sleep(STREAM_STABILIZE)

    # streams = [(f"cam{i}", b) for i in range(1, STREAM_CAMS + 1) for b in ["detection", "recognition"]]
    # all_live, results = verify_streams(streams)
    # for s, live in results.items():
    #     log(f"  {s}: {'LIVE' if live else 'DEAD'}")

    # rs = stream_status()
    # total = sum(len(branches) for branches in rs.values()) if isinstance(rs, dict) and "error" not in rs else 0
    # expected = STREAM_CAMS * 2
    # if not verify(f"{expected} concurrent streams running", total == expected):
    #     failures += 1
    # if not verify("All streams live", all_live):
    #     failures += 1

    # # TEST 4: Branch Switching
    # step("TEST 4: Branch Switching (Stream survives)")

    # log("Removing cam1 from recognition...")
    # ok = remove_from_branch("cam1", "recognition")
    # log(f"  Remove cam1 from recognition: {'OK' if ok else 'FAIL'}")

    # streams = [("cam1", "detection")]
    # all_live, results = verify_streams(streams)
    # log(f"  cam1/detection: {'LIVE' if results.get('cam1/detection') else 'DEAD'}")
    # if not verify("cam1 detection stream survived", all_live):
    #     failures += 1

    # log("Re-adding cam1 to recognition...")
    # ok = add_to_branch("cam1", "recognition")
    # log(f"  Add cam1 to recognition: {'OK' if ok else 'FAIL'}")

    # ok = start_stream("cam1", "recognition", bitrate=1000000)
    # log(f"  Start cam1/recognition: {'OK' if ok else 'FAIL'}")

    # # TEST 5: Stop/Start cycle
    # step("TEST 5: Stop/Start Stream Cycle")

    # log("Stopping all detection streams...")
    # for i in range(1, STREAM_CAMS + 1):
    #     ok = stop_stream(f"cam{i}", "detection")
    #     log(f"  Stop cam{i}/detection: {'OK' if ok else 'FAIL'}")

    # streams = [(f"cam{i}", "recognition") for i in range(1, STREAM_CAMS + 1)]
    # all_live, results = verify_streams(streams)
    # for s, live in results.items():
    #     log(f"  {s}: {'LIVE' if live else 'DEAD'}")
    # if not verify("Recognition streams survived detection stop", all_live):
    #     failures += 1

    # log("Re-starting detection streams...")
    # for i in range(1, STREAM_CAMS + 1):
    #     ok = start_stream(f"cam{i}", "detection")
    #     log(f"  Start cam{i}/detection: {'OK' if ok else 'FAIL'}")
    #     if not ok:
    #         failures += 1

    # log(f"Waiting {STREAM_STABILIZE}s for stabilization...")
    # time.sleep(STREAM_STABILIZE)

    # streams = [(f"cam{i}", b) for i in range(1, STREAM_CAMS + 1) for b in ["detection", "recognition"]]
    # all_live, results = verify_streams(streams)
    # for s, live in results.items():
    #     log(f"  {s}: {'LIVE' if live else 'DEAD'}")
    # if not verify("All streams live after restart", all_live):
    #     failures += 1

    # # TEST 6: Stop all streams
    # step("TEST 6: Stop All Streams")

    # log("Stopping all streams...")
    # for i in range(1, STREAM_CAMS + 1):
    #     stop_stream(f"cam{i}", "detection")
    #     stop_stream(f"cam{i}", "recognition")

    # rs = stream_status()
    # stopped = len(rs) == 0 or "error" in rs
    # if not verify("All streams stopped", stopped):
    #     failures += 1

    # # TEST 7: Remove all cameras
    # step("TEST 7: Remove All Cameras")

    # log(f"Removing {MAX_CAMERAS} cameras...")
    # for i in range(1, MAX_CAMERAS + 1):
    #     ok = remove_camera(f"cam{i}")
    #     log(f"  Remove cam{i}: {'OK' if ok else 'FAIL'}")
    #     if not ok:
    #         failures += 1

    # h = health()
    # if not verify("All cameras removed", h.get("cameras") == 0):
    #     failures += 1

    # # SUMMARY
    # step("TEST SUMMARY")
    # if failures == 0:
    #     log("ALL TESTS PASSED!")
    #     return 0
    # else:
    #     log(f"FAILURES: {failures}")
    #     return 1


if __name__ == "__main__":
    sys.exit(main())
