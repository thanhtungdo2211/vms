#!/bin/bash
# Stress Test - Camera CRUD + Stream Publishing
# Usage: ./tests/stress_test.sh

BASE_URL="http://localhost:8083"
STREAM_SERVER="152.42.221.89:8554"
CAMERA_URI="rtsp://192.168.6.14:8554/testface"

echo "============================================================"
echo "  STRESS TEST: Camera CRUD + Stream Publishing"
echo "============================================================"

# Helper: Wait for operation
wait_operation() {
    local op_id="$1"
    local max_wait="${2:-60}"

    for i in $(seq 1 $max_wait); do
        OP_STATUS=$(curl -s http://localhost:8083/api/operations/$op_id)
        STATUS=$(echo "$OP_STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)

        if [ "$STATUS" = "ok" ]; then
            echo "  ✓ Success (${i}s)"
            return 0
        elif [ "$STATUS" = "error" ]; then
            echo "  ✗ Failed:"
            echo "$OP_STATUS" | python3 -m json.tool
            return 1
        fi
        sleep 1
    done
    echo "  ⏱ Timeout after ${max_wait}s"
    return 1
}

# Pre-check
echo -e "\n[Health Check]"
curl -s http://localhost:8083/api/health | python3 -m json.tool

echo -e "\n[Pipeline Status]"
curl -s http://localhost:8083/api/status | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"State: {d['pipeline']['state']}\")
print(f\"Cameras: {d['cameras']['count']}\")
print(f\"Operations queue: {d['operations']['queue_size']}\")
print(f\"Last op: {d['system']['last_operation_type']}\")
"

# Cleanup
echo -e "\n[Cleanup] Removing all cameras..."
RESP=$(curl -s -X DELETE http://localhost:8083/api/cameras/cam1)
OP_ID=$(echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin).get('operation_id',''))" 2>/dev/null)
if [ -n "$OP_ID" ]; then
    wait_operation "$OP_ID" 10
fi

# Test 1: Add camera
echo -e "\n[TEST 1] Adding cam1 to detection..."
RESP=$(curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","uri":"'$CAMERA_URI'","branch":"detection"}')

echo "$RESP" | python3 -m json.tool
OP_ID=$(echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin).get('operation_id',''))" 2>/dev/null)
if [ -n "$OP_ID" ]; then
    wait_operation "$OP_ID"
fi

# Test 2: Start stream
echo -e "\n[TEST 2] Starting stream for cam1/detection..."
RESP=$(curl -s -X POST http://localhost:8083/api/cameras/cam1/branches/detection/stream/start \
  -H "Content-Type: application/json" \
  -d '{"uri":"rtsp://'$STREAM_SERVER'/stress_cam1_detection","bitrate":4000000}')

echo "$RESP" | python3 -m json.tool
OP_ID=$(echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin).get('operation_id',''))" 2>/dev/null)
if [ -n "$OP_ID" ]; then
    wait_operation "$OP_ID"
fi

# Check streams
echo -e "\n[Status] All streams:"
curl -s http://localhost:8083/api/streams | python3 -m json.tool

# Final detailed status
echo -e "\n[Final Detailed Status]"
curl -s http://localhost:8083/api/status | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('='*60)
print(f\"Pipeline: {d['pipeline']['state']}\")
print(f\"Cameras: {d['cameras']['count']}\")
for cam_id, info in d['cameras']['details'].items():
    print(f\"  - {cam_id}: state={info['state']}, branches={info['branches']}\")
print(f\"Streams: {len(d.get('streams', {}))}\")
print(f\"Queue size: {d['operations']['queue_size']}\")
print('='*60)
"

echo -e "\n[Done] Test completed!"
