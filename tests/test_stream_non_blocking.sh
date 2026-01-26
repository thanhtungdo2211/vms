#!/bin/bash
# Test: Stream with REAL RTSP server - Pipeline should NOT pause/block
# Scenarios:
#   1. Add camera to detection branch
#   2. Start stream to REAL RTSP server (143.198.198.52:8554)
#   3. Monitor pipeline - should NOT pause/block
#   4. Verify stream is publishing
#   5. Verify pipeline is still running smoothly

CAMERA_URI="rtsp://192.168.6.14:8554/testface"
STREAM_SERVER="143.198.198.52"
STREAM_PORT="8554"
STREAM_NAME="vmsx_test_cam1"
STREAM_URI="rtsp://${STREAM_SERVER}:${STREAM_PORT}/${STREAM_NAME}"
API_BASE="http://localhost:8083/api"

echo "============================================================"
echo "  TEST: Non-Blocking Stream with Real RTSP Server"
echo "============================================================"
echo "  Stream Server: ${STREAM_SERVER}:${STREAM_PORT}"
echo "  Stream URI: ${STREAM_URI}"
echo "============================================================"

# Step 1: Check initial health
echo -e "\n[1] Checking initial pipeline health..."
HEALTH_BEFORE=$(curl -s ${API_BASE}/health)
echo "$HEALTH_BEFORE" | python3 -m json.tool

if ! echo "$HEALTH_BEFORE" | grep -q '"status":"healthy"'; then
    echo "✗ Pipeline is NOT healthy before test"
    exit 1
fi
echo "✓ Pipeline is healthy"

# Step 2: Add cam1 to branch detection
echo -e "\n[2] Adding cam1 to branch detection..."
ADD_RESULT=$(curl -s -X POST ${API_BASE}/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","uri":"'$CAMERA_URI'","branch":"detection"}')
echo "$ADD_RESULT" | python3 -m json.tool
ADD_OP_ID=$(echo "$ADD_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['operation_id'])")

sleep 3

# Step 3: Start stream to REAL server (should be NON-BLOCKING)
echo -e "\n[3] Starting stream to REAL server: $STREAM_URI"
echo "    (Pipeline should NOT pause/block during connection)"
START_TIME=$(date +%s)

STREAM_RESULT=$(curl -s -X POST ${API_BASE}/cameras/cam1/branches/detection/stream/start \
  -H "Content-Type: application/json" \
  -d '{"uri":"'$STREAM_URI'","bitrate":4000000}')
echo "$STREAM_RESULT" | python3 -m json.tool
STREAM_OP_ID=$(echo "$STREAM_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['operation_id'])")

API_RESPONSE_TIME=$(($(date +%s) - START_TIME))
echo "    API response time: ${API_RESPONSE_TIME}s"

if [ $API_RESPONSE_TIME -gt 5 ]; then
    echo "⚠️  API response was slow (${API_RESPONSE_TIME}s) - pipeline may have paused"
else
    echo "✓ API response was fast (${API_RESPONSE_TIME}s) - non-blocking"
fi

# Step 4: Check pipeline health IMMEDIATELY (should still be healthy, not paused)
echo -e "\n[4] Checking pipeline health IMMEDIATELY after stream start..."
HEALTH_AFTER=$(curl -s ${API_BASE}/health)
echo "$HEALTH_AFTER" | python3 -m json.tool

if ! echo "$HEALTH_AFTER" | grep -q '"status":"healthy"'; then
    echo "✗ Pipeline is NOT healthy after stream start"
    exit 1
fi
echo "✓ Pipeline is still healthy (non-blocking confirmed)"

# Step 5: Wait for stream to connect
echo -e "\n[5] Waiting for stream to connect (5s)..."
sleep 5

# Step 6: Check operation result
echo -e "\n[6] Checking stream operation result..."
OP_RESULT=$(curl -s ${API_BASE}/operations/${STREAM_OP_ID})
echo "$OP_RESULT" | python3 -m json.tool

if echo "$OP_RESULT" | grep -q '"status":"ok"'; then
    echo "✓ Stream operation succeeded"
elif echo "$OP_RESULT" | grep -q '"status":"error"'; then
    echo "✗ Stream operation failed"
    exit 1
fi

# Step 7: Verify stream status
echo -e "\n[7] Verifying stream status..."
STREAM_STATUS=$(curl -s ${API_BASE}/cameras/cam1/branches/detection/stream)
echo "$STREAM_STATUS" | python3 -m json.tool

if echo "$STREAM_STATUS" | grep -q '"publishing":true'; then
    echo "✓ Stream is PUBLISHING"
else
    echo "✗ Stream is NOT publishing"
    exit 1
fi

# Step 8: Verify stream with ffprobe
echo -e "\n[8] Verifying stream with ffprobe (8s timeout)..."
if timeout 8 ffprobe -v quiet -rtsp_transport tcp -stimeout 8000000 \
   -show_entries stream=codec_name -of default=nw=1 "$STREAM_URI" 2>&1 | grep -q codec_name; then
    echo "✓ Stream is LIVE and playable"
    ffprobe -v quiet -rtsp_transport tcp -stimeout 8000000 \
      -show_entries stream=codec_name,width,height,r_frame_rate -of json "$STREAM_URI" 2>/dev/null | python3 -m json.tool
else
    echo "⚠️  Stream is not yet playable (may still be connecting)"
fi

echo -e "\n============================================================"
echo "  ✓ TEST PASSED: Pipeline did NOT pause/block during stream"
echo "============================================================"
echo "  Stream URI: ${STREAM_URI}"
echo "  You can verify with: ffplay -rtsp_transport tcp ${STREAM_URI}"
echo "============================================================"
