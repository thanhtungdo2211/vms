#!/bin/bash
# Test: Stream error handling - Pipeline should continue running
# Scenarios:
#   1. Add camera to detection branch
#   2. Try to start stream to NON-EXISTENT RTSP server (should fail gracefully)
#   3. Verify pipeline is still running
#   4. Verify camera detection is still working
#   5. Check operation error details

CAMERA_URI="rtsp://192.168.6.14:8554/testface"
INVALID_SERVER="rtsp://192.168.99.99:8554/test"  # Non-existent server
API_BASE="http://localhost:8083/api"

echo "============================================================"
echo "  TEST: Stream Error Handling (Pipeline Resilience)"
echo "============================================================"

# Step 1: Add cam1 to branch detection
echo -e "\n[1] Adding cam1 to branch detection..."
ADD_RESULT=$(curl -s -X POST ${API_BASE}/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","uri":"'$CAMERA_URI'","branch":"detection"}')
echo "$ADD_RESULT" | python3 -m json.tool
ADD_OP_ID=$(echo "$ADD_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['operation_id'])")

sleep 3

# Step 2: Try to start stream to NON-EXISTENT server
echo -e "\n[2] Trying to start stream to NON-EXISTENT server: $INVALID_SERVER"
echo "    (This should FAIL, but pipeline should CONTINUE running)"
STREAM_RESULT=$(curl -s -X POST ${API_BASE}/cameras/cam1/branches/detection/stream/start \
  -H "Content-Type: application/json" \
  -d '{"uri":"'$INVALID_SERVER'","bitrate":4000000}')
echo "$STREAM_RESULT" | python3 -m json.tool
STREAM_OP_ID=$(echo "$STREAM_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['operation_id'])")

sleep 5

# Step 3: Check operation result (should show error)
echo -e "\n[3] Checking operation result (should show ConnectionError)..."
curl -s ${API_BASE}/operations/${STREAM_OP_ID} | python3 -m json.tool

# Step 4: Verify pipeline is still running
echo -e "\n[4] Verifying pipeline health..."
HEALTH=$(curl -s ${API_BASE}/health)
echo "$HEALTH" | python3 -m json.tool

if echo "$HEALTH" | grep -q '"status":"healthy"'; then
    echo "✓ Pipeline is HEALTHY"
else
    echo "✗ Pipeline is NOT healthy"
    exit 1
fi

# Step 5: Verify camera is still in pipeline
echo -e "\n[5] Verifying camera is still active..."
CAMERAS=$(curl -s ${API_BASE}/cameras)
echo "$CAMERAS" | python3 -m json.tool

if echo "$CAMERAS" | grep -q "cam1"; then
    echo "✓ Camera cam1 is ACTIVE"
else
    echo "✗ Camera cam1 is NOT active"
    exit 1
fi

# Step 6: Verify stream is NOT publishing (should show publishing: false)
echo -e "\n[6] Verifying stream is NOT publishing..."
STREAM_STATUS=$(curl -s ${API_BASE}/cameras/cam1/branches/detection/stream)
echo "$STREAM_STATUS" | python3 -m json.tool

if echo "$STREAM_STATUS" | grep -q '"publishing":false'; then
    echo "✓ Stream is correctly NOT publishing"
else
    echo "✗ Stream status unexpected"
    exit 1
fi

echo -e "\n============================================================"
echo "  ✓ TEST PASSED: Pipeline continues running after stream error"
echo "============================================================"
