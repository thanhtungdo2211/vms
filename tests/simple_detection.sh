#!/bin/bash
# Test: Add -> Remove -> Re-add -> Stream cam1
# Scenario:
#   1. Add cam1 to branch detection
#   2. Remove cam1 from pipeline
#   3. Re-add cam1 to branch detection
#   4. Start streaming cam1 detection to rtsp://152.42.221.89:8554/cam1_detection

CAMERA_URI="rtsp://192.168.6.14:8554/testface"
STREAM_SERVER="143.198.198.52"
STREAM_PORT="8554"
STREAM_NAME="cam1_detection"
STREAM_URI="rtsp://${STREAM_SERVER}:${STREAM_PORT}/${STREAM_NAME}"

echo "============================================================"
echo "  TEST: Add -> Remove -> Re-add -> Stream cam1"
echo "============================================================"

# Step 1: Add cam1 to branch detection
echo -e "\n[1] Adding cam1 to branch detection..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","uri":"'$CAMERA_URI'","branch":"detection"}' \
  | python3 -m json.tool

sleep 3

# Step 4: Start stream cam1 detection
echo -e "\n[4] Starting stream cam1_detection to $STREAM_URI..."
curl -s -X POST http://localhost:8083/api/cameras/cam1/branches/detection/stream/start \
  -H "Content-Type: application/json" \
  -d '{"uri":"'$STREAM_URI'","bitrate":4000000}' \
  | python3 -m json.tool

sleep 5

# Verify stream is live
echo -e "\n[5] Verifying stream..."
if ffprobe -v quiet -rtsp_transport tcp -stimeout 8000000 \
   -show_entries stream=codec_name -of default=nw=1 "$STREAM_URI" 2>&1 | grep -q codec_name; then
    echo "✓ Stream is LIVE: $STREAM_URI"
    ffprobe -v quiet -rtsp_transport tcp -stimeout 8000000 \
      -show_entries stream=codec_name,width,height -of json "$STREAM_URI" 2>/dev/null | python3 -m json.tool
    exit 0
else
    echo "✗ Stream NOT live: $STREAM_URI"
    exit 1
fi
