#!/usr/bin/env python3
"""Comprehensive Stress Test - CRUD Camera + RTSP Publishing.

Tests:
1. Rapid camera add/remove cycles
2. Branch switching (add/remove from branches)
3. Multiple RTSP publish/stop cycles
4. Concurrent RTSP streams on multiple branches
5. RTSP during camera removal
6. Re-adding cameras after removal
7. Maximum camera load

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
RTSP_SERVER = "rtsp://192.168.6.14:8554"
CAMERA_URI = "rtsp://192.168.6.14:8554/testface"
MAX_CAMERAS = 3  # Maximum cameras to test
RTSP_CAMS = 2  # Number of cameras to test RTSP (subset of MAX_CAMERAS)
RTSP_STABILIZE = 2  # Wait for RTSP streams to stabilize before verification


def api(method: str, endpoint: str, data: dict = None) -> dict:
    """Make API call."""
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
    """Wait for async operation."""
    start = time.time()
    while time.time() - start < timeout:
        r = api("GET", f"/api/operations/{op_id}")
        if r.get("status") in ("ok", "error"):
            return r
        time.sleep(0.5)
    return {"status": "timeout"}


def health() -> dict:
    """Get health status."""
    return api("GET", "/api/health")


def add_camera(cam_id: str, branches: list) -> bool:
    """Add camera to branches."""
    r = api("POST", "/api/cameras", {
        "camera_id": cam_id,
        "uri": CAMERA_URI,
        "branches": branches
    })
    if "operation_id" in r:
        result = wait_op(r["operation_id"])
        return result.get("status") == "ok"
    return False


def remove_camera(cam_id: str) -> bool:
    """Remove camera entirely."""
    r = api("DELETE", f"/api/cameras/{cam_id}")
    if "operation_id" in r:
        result = wait_op(r["operation_id"])
        return result.get("status") == "ok"
    return False


def remove_from_branch(cam_id: str, branch: str) -> bool:
    """Remove camera from branch."""
    r = api("DELETE", f"/api/cameras/{cam_id}/branches/{branch}")
    if "operation_id" in r:
        result = wait_op(r["operation_id"])
        return result.get("status") == "ok"
    return False


def add_to_branch(cam_id: str, branch: str) -> bool:
    """Add camera to branch."""
    r = api("POST", f"/api/cameras/{cam_id}/branches/{branch}")
    if "operation_id" in r:
        result = wait_op(r["operation_id"])
        return result.get("status") == "ok"
    return False


def start_rtsp(cam_id: str, branch: str, bitrate: int = 4000000) -> bool:
    """Start RTSP publishing for camera/branch."""
    location = f"{RTSP_SERVER}/stress_{cam_id}_{branch}"
    r = api("POST", f"/api/cameras/{cam_id}/branches/{branch}/rtsp/start", {
        "location": location,
        "bitrate": bitrate
    })
    if "operation_id" in r:
        result = wait_op(r["operation_id"])
        return result.get("status") == "ok"
    return False


def stop_rtsp(cam_id: str, branch: str) -> bool:
    """Stop RTSP publishing for camera/branch."""
    r = api("POST", f"/api/cameras/{cam_id}/branches/{branch}/rtsp/stop")
    if "operation_id" in r:
        result = wait_op(r["operation_id"])
        return result.get("status") == "ok"
    return False


def rtsp_status() -> dict:
    """Get all RTSP status."""
    return api("GET", "/api/cameras/rtsp/status")


def check_rtsp_live(url: str, timeout: int = 8) -> bool:
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
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 3)
        if "codec_name=" in result.stdout:
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Fallback: RTSP DESCRIBE request
    try:
        parts = url.replace("rtsp://", "").split("/", 1)
        host_port = parts[0].split(":")
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else 554

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        request = f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n\r\n"
        sock.send(request.encode())
        response = sock.recv(4096).decode()
        sock.close()

        return "RTSP/1.0 200" in response
    except Exception:
        return False


def verify_rtsp_streams(streams: list, retries: int = 3) -> tuple:
    """Verify multiple RTSP streams are live with retry.

    Args:
        streams: List of (cam_id, branch) tuples
        retries: Number of retry attempts for dead streams

    Returns:
        (all_live, results_dict)
    """
    results = {}
    for cam_id, branch in streams:
        url = f"{RTSP_SERVER}/stress_{cam_id}_{branch}"
        # Try up to retries times for each stream
        for attempt in range(retries):
            live = check_rtsp_live(url)
            if live:
                break
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retry
        results[f"{cam_id}/{branch}"] = live
    all_live = all(results.values()) if results else True
    return all_live, results


def log(msg: str):
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def step(name: str):
    """Print step header."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def verify(name: str, condition: bool) -> bool:
    """Verify condition and print result."""
    status = "PASS" if condition else "FAIL"
    log(f"  {name}: {status}")
    return condition


def cleanup():
    """Remove all cameras and stop RTSP."""
    log("Cleanup: Removing all cameras...")
    for i in range(1, MAX_CAMERAS + 1):
        remove_camera(f"cam{i}")
        time.sleep(1)
    # Extra wait for GPU resources to be released
    time.sleep(3)


def main():
    print("\n" + "="*60)
    print("  STRESS TEST: CRUD + RTSP Comprehensive")
    print("="*60)
    failures = 0

    # Pre-check
    step("Pre-check: Health & Cleanup")
    h = health()
    if h.get("status") != "healthy":
        log(f"API not healthy: {h}")
        return 1
    log(f"Health: {h}")

    # Cleanup any existing cameras
    if h.get("cameras", 0) > 0:
        cleanup()

    # =========================================================================
    # TEST 1: Add cameras (will be reused in subsequent tests)
    # =========================================================================
    step("TEST 1: Add Cameras")
    for i in range(1, MAX_CAMERAS + 1):
        ok = add_camera(f"cam{i}", ["detection", "recognition"])
        log(f"  Add cam{i}: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    h = health()
    if not verify(f"{MAX_CAMERAS} cameras added", h.get("cameras") == MAX_CAMERAS):
        failures += 1

    # =========================================================================
    # TEST 2: RTSP Publish on detection
    # =========================================================================
    step(f"TEST 2: RTSP Publish on Detection ({RTSP_CAMS} cameras)")

    for i in range(1, RTSP_CAMS + 1):
        ok = start_rtsp(f"cam{i}", "detection")
        log(f"  Start cam{i}/detection: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    log(f"Waiting {RTSP_STABILIZE}s for RTSP stabilization...")
    time.sleep(RTSP_STABILIZE)

    streams = [(f"cam{i}", "detection") for i in range(1, RTSP_CAMS + 1)]
    all_live, results = verify_rtsp_streams(streams)
    for s, live in results.items():
        log(f"  {s}: {'LIVE' if live else 'DEAD'}")
    if not verify(f"{RTSP_CAMS} detection streams live", all_live):
        failures += 1

    # =========================================================================
    # TEST 3: RTSP Publish on recognition (concurrent with detection)
    # =========================================================================
    step(f"TEST 3: RTSP Publish on Recognition ({RTSP_CAMS} cameras concurrent)")

    for i in range(1, RTSP_CAMS + 1):
        ok = start_rtsp(f"cam{i}", "recognition", bitrate=1000000)
        log(f"  Start cam{i}/recognition: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    log(f"Waiting {RTSP_STABILIZE}s for RTSP stabilization...")
    time.sleep(RTSP_STABILIZE)

    # Verify all streams (both detection and recognition)
    streams = [(f"cam{i}", b) for i in range(1, RTSP_CAMS + 1) for b in ["detection", "recognition"]]
    all_live, results = verify_rtsp_streams(streams)
    for s, live in results.items():
        log(f"  {s}: {'LIVE' if live else 'DEAD'}")

    rs = rtsp_status()
    total = sum(len(branches) for branches in rs.values())
    expected = RTSP_CAMS * 2
    if not verify(f"{expected} concurrent streams running", total == expected):
        failures += 1
    if not verify("All streams live", all_live):
        failures += 1

    # =========================================================================
    # TEST 4: Branch Switching (RTSP should survive)
    # =========================================================================
    step("TEST 4: Branch Switching (RTSP survives)")

    log("Removing cam1 from recognition (detection RTSP should survive)...")
    ok = remove_from_branch("cam1", "recognition")
    log(f"  Remove cam1 from recognition: {'OK' if ok else 'FAIL'}")

    # Verify cam1 detection RTSP still alive
    streams = [("cam1", "detection")]
    all_live, results = verify_rtsp_streams(streams)
    log(f"  cam1/detection: {'LIVE' if results.get('cam1/detection') else 'DEAD'}")
    if not verify("cam1 detection RTSP survived", all_live):
        failures += 1

    log("Re-adding cam1 to recognition...")
    ok = add_to_branch("cam1", "recognition")
    log(f"  Add cam1 to recognition: {'OK' if ok else 'FAIL'}")

    # Start recognition RTSP for cam1 again
    ok = start_rtsp("cam1", "recognition", bitrate=1000000)
    log(f"  Start cam1/recognition: {'OK' if ok else 'FAIL'}")

    # =========================================================================
    # TEST 5: Stop/Start RTSP cycle
    # =========================================================================
    step("TEST 5: Stop/Start RTSP Cycle")

    log("Stopping all detection RTSP...")
    for i in range(1, RTSP_CAMS + 1):
        ok = stop_rtsp(f"cam{i}", "detection")
        log(f"  Stop cam{i}/detection: {'OK' if ok else 'FAIL'}")

    # Verify recognition RTSP still alive
    streams = [(f"cam{i}", "recognition") for i in range(1, RTSP_CAMS + 1)]
    all_live, results = verify_rtsp_streams(streams)
    for s, live in results.items():
        log(f"  {s}: {'LIVE' if live else 'DEAD'}")
    if not verify("Recognition RTSP survived detection stop", all_live):
        failures += 1

    log("Re-starting detection RTSP...")
    for i in range(1, RTSP_CAMS + 1):
        ok = start_rtsp(f"cam{i}", "detection")
        log(f"  Start cam{i}/detection: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    log(f"Waiting {RTSP_STABILIZE}s for RTSP stabilization...")
    time.sleep(RTSP_STABILIZE)

    # Verify all streams again
    streams = [(f"cam{i}", b) for i in range(1, RTSP_CAMS + 1) for b in ["detection", "recognition"]]
    all_live, results = verify_rtsp_streams(streams)
    for s, live in results.items():
        log(f"  {s}: {'LIVE' if live else 'DEAD'}")
    if not verify("All streams live after restart", all_live):
        failures += 1

    # =========================================================================
    # TEST 6: Stop all RTSP
    # =========================================================================
    step("TEST 6: Stop All RTSP")

    log("Stopping all RTSP streams...")
    for i in range(1, RTSP_CAMS + 1):
        stop_rtsp(f"cam{i}", "detection")
        stop_rtsp(f"cam{i}", "recognition")

    rs = rtsp_status()
    if not verify("All RTSP stopped", len(rs) == 0):
        failures += 1

    # =========================================================================
    # TEST 7: Remove all cameras
    # =========================================================================
    step("TEST 7: Remove All Cameras")

    log(f"Removing {MAX_CAMERAS} cameras...")
    for i in range(1, MAX_CAMERAS + 1):
        ok = remove_camera(f"cam{i}")
        log(f"  Remove cam{i}: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    h = health()
    if not verify("All cameras removed", h.get("cameras") == 0):
        failures += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================
    step("TEST SUMMARY")
    if failures == 0:
        log("ALL TESTS PASSED!")
        return 0
    else:
        log(f"FAILURES: {failures}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
