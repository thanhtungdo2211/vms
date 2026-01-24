#!/bin/bash
# Test: Add -> Remove -> Re-add -> Stream cam1
# Scenario:
#   1. Add cam1 to branch detection
#   2. Remove cam1 from pipeline
#   3. Re-add cam1 to branch detection
#   4. Start streaming cam1 detection to rtsp://152.42.221.89:8554/cam1_detection

CAMERA_URI="rtsp://192.168.6.14:8554/testface"
STREAM_SERVER="152.42.221.89"
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

# Step 2: Remove cam1 from pipeline
echo -e "\n[2] Removing cam1 from pipeline..."
curl -s -X DELETE http://localhost:8083/api/cameras/cam1 | python3 -m json.tool

sleep 3

# Step 3: Re-add cam1 to branch detection
echo -e "\n[3] Re-adding cam1 to branch detection..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","uri":"'$CAMERA_URI'","branch":"detection"}' \
  | python3 -m json.tool

sleep 8
