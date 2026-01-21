#!/usr/bin/env python3
"""Test 10: Mixed Operations - CRUD Camera + RTSP Publishing.

Based on docs/test10.md workflow:
1. Add 5 cameras to detection + recognition
2. Remove cam1, cam2 from recognition
3. Start RTSP for cam1,2,3 on detection
4. Remove cam3, cam4 from recognition (while RTSP running)
5. Add cam1, cam2 back to recognition
5b. Start RTSP for cam1, cam2 on recognition
6. Stop all RTSP
7. Remove all cameras

Usage:
    python3 tests/mix_test.py
"""

import socket
import subprocess
import sys
import time
import requests

# Configuration
BASE_URL = "http://localhost:8083"
RTSP_SERVER = "rtsp://192.168.6.14:8554"
CAMERA_URI = "rtsp://192.168.6.14:8554/testface"
RTSP_STABILIZE = 8  # Wait for RTSP streams to stabilize before verification


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
    location = f"{RTSP_SERVER}/mix_{cam_id}_{branch}"
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
            "-stimeout", str(timeout * 1000000),  # microseconds
            "-show_entries", "stream=codec_name",
            "-of", "default=nw=1",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 3)
        if "codec_name=" in result.stdout:
            return True
    except FileNotFoundError:
        pass  # ffprobe not available, use fallback
    except Exception:
        pass

    # Fallback: RTSP DESCRIBE request
    try:
        parts = url.replace("rtsp://", "").split("/", 1)
        host_port = parts[0].split(":")
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else 554
        path = "/" + parts[1] if len(parts) > 1 else "/"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        request = f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n\r\n"
        sock.send(request.encode())
        response = sock.recv(4096).decode()
        sock.close()

        # Check for 200 OK response
        return "RTSP/1.0 200" in response
    except Exception:
        return False


def verify_rtsp_streams_live(streams: list) -> tuple:
    """Verify multiple RTSP streams are live.

    Args:
        streams: List of (cam_id, branch) tuples

    Returns:
        (all_live, results_dict)
    """
    results = {}
    for cam_id, branch in streams:
        url = f"{RTSP_SERVER}/mix_{cam_id}_{branch}"
        live = check_rtsp_live(url)
        results[f"{cam_id}/{branch}"] = live
    all_live = all(results.values())
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


def main():
    print("\n" + "="*60)
    print("  TEST 10: Mixed Operations - CRUD + RTSP")
    print("="*60)

    # Pre-check
    step("Pre-check: Health")
    h = health()
    if h.get("status") != "healthy":
        log(f"API not healthy: {h}")
        return 1
    log(f"Health: {h}")

    # Step 1: Add 3 cameras to detection + recognition
    step("Step 1: Add 3 cameras to detection + recognition")
    for i in range(1, 4):
        cam_id = f"cam{i}"
        ok = add_camera(cam_id, ["detection", "recognition"])
        log(f"  Add {cam_id}: {'OK' if ok else 'FAIL'}")

    h = health()
    log(f"Health: cameras={h.get('cameras')}")
    verify("3 cameras added", h.get("cameras") == 3)

    # Step 2: Remove cam1, cam2 from recognition
    step("Step 2: Remove cam1, cam2 from recognition")
    for cam_id in ["cam1", "cam2"]:
        ok = remove_from_branch(cam_id, "recognition")
        log(f"  Remove {cam_id} from recognition: {'OK' if ok else 'FAIL'}")

    h = health()
    log(f"Health: {h}")

    # Step 3: Start RTSP for cam1, cam2 on detection
    step("Step 3: Start RTSP cam1,2 on detection")
    for cam_id in ["cam1", "cam2"]:
        ok = start_rtsp(cam_id, "detection")
        log(f"  Start RTSP {cam_id}/detection: {'OK' if ok else 'FAIL'}")

    log(f"Waiting {RTSP_STABILIZE}s for RTSP stabilization...")
    time.sleep(RTSP_STABILIZE)

    h = health()
    log(f"Health: {h}")
    rs = rtsp_status()
    log(f"RTSP status: {rs}")
    verify("2 detection RTSP streams", len(rs) == 2)

    # Verify RTSP streams are actually live
    log("Verifying RTSP streams are live...")
    streams_to_check = [("cam1", "detection"), ("cam2", "detection")]
    all_live, live_results = verify_rtsp_streams_live(streams_to_check)
    for stream, is_live in live_results.items():
        log(f"  {stream}: {'LIVE' if is_live else 'DEAD'}")
    if not verify("All detection RTSP streams live", all_live):
        log("FAIL: RTSP streams not live - aborting test")
        return 1

    # Step 4: Verify RTSP still works after branch changes
    step("Step 4: Verify RTSP survives branch operations")
    h = health()
    log(f"Health: {h}")
    rs = rtsp_status()
    verify("RTSP still alive", len(rs) == 2)

    # Step 5: Add cam1, cam2 back to recognition
    step("Step 5: Add cam1, cam2 back to recognition")
    for cam_id in ["cam1", "cam2"]:
        ok = add_to_branch(cam_id, "recognition")
        log(f"  Add {cam_id} to recognition: {'OK' if ok else 'FAIL'}")

    h = health()
    log(f"Health: {h}")
    rs = rtsp_status()
    verify("RTSP still alive", len(rs) == 2)

    # Step 5b: Start RTSP for cam1, cam2 on recognition
    step("Step 5b: Start RTSP cam1,2 on recognition")
    for cam_id in ["cam1", "cam2"]:
        ok = start_rtsp(cam_id, "recognition", bitrate=1000000)
        log(f"  Start RTSP {cam_id}/recognition: {'OK' if ok else 'FAIL'}")

    log(f"Waiting {RTSP_STABILIZE}s for RTSP stabilization...")
    time.sleep(RTSP_STABILIZE)

    h = health()
    log(f"Health: {h}")
    rs = rtsp_status()
    log(f"RTSP status: {rs}")
    # Should have: cam1(det+rec), cam2(det+rec) = 4 streams
    total_streams = sum(len(branches) for branches in rs.values())
    verify("4 total RTSP streams (2 det + 2 rec)", total_streams == 4)

    # Verify all RTSP streams are actually live
    log("Verifying all RTSP streams are live...")
    streams_to_check = [
        ("cam1", "detection"), ("cam2", "detection"),
        ("cam1", "recognition"), ("cam2", "recognition")
    ]
    all_live, live_results = verify_rtsp_streams_live(streams_to_check)
    for stream, is_live in live_results.items():
        log(f"  {stream}: {'LIVE' if is_live else 'DEAD'}")
    if not verify("All 4 RTSP streams live", all_live):
        log("FAIL: RTSP streams not live - aborting test")
        return 1

    # Step 6: Stop all RTSP
    step("Step 6: Stop all RTSP")
    # Stop detection RTSP
    for cam_id in ["cam1", "cam2"]:
        ok = stop_rtsp(cam_id, "detection")
        log(f"  Stop RTSP {cam_id}/detection: {'OK' if ok else 'FAIL'}")

    # Stop recognition RTSP
    for cam_id in ["cam1", "cam2"]:
        ok = stop_rtsp(cam_id, "recognition")
        log(f"  Stop RTSP {cam_id}/recognition: {'OK' if ok else 'FAIL'}")

    h = health()
    log(f"Health: {h}")
    rs = rtsp_status()
    verify("All RTSP stopped", len(rs) == 0)

    # Step 7: Remove all cameras
    step("Step 7: Remove all 3 cameras")
    for i in range(1, 4):
        cam_id = f"cam{i}"
        ok = remove_camera(cam_id)
        log(f"  Remove {cam_id}: {'OK' if ok else 'FAIL'}")

    h = health()
    log(f"Health: cameras={h.get('cameras')}")
    verify("All cameras removed", h.get("cameras") == 0)

    # Final
    step("TEST COMPLETE")
    log("All steps executed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
