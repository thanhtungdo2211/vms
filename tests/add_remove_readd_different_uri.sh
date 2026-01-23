#!/bin/bash
# Test: Add -> Remove -> Re-add with DIFFERENT URI

CAMERA_URI_1="rtsp://192.168.6.14:8554/testface"
CAMERA_URI_2="rtsp://192.168.6.14:8554/testface2"  # Different URI
STREAM_SERVER="152.42.221.89"
STREAM_PORT="8554"
STREAM_NAME="cam1_detection"
STREAM_URI="rtsp://${STREAM_SERVER}:${STREAM_PORT}/${STREAM_NAME}"

echo "============================================================"
echo "  TEST: Add -> Remove -> Re-add cam1 with DIFFERENT URI"
echo "============================================================"

# Step 1: Add cam1 with URI 1
echo -e "\n[1] Adding cam1 with URI 1..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","uri":"'$CAMERA_URI_1'","branch":"detection"}' \
  | python3 -m json.tool

sleep 3

# Step 2: Remove cam1
echo -e "\n[2] Removing cam1..."
curl -s -X DELETE http://localhost:8083/api/cameras/cam1 | python3 -m json.tool

sleep 2

# Step 3: Re-add cam1 with DIFFERENT URI
echo -e "\n[3] Re-adding cam1 with DIFFERENT URI 2..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","uri":"'$CAMERA_URI_2'","branch":"detection"}' \
  | python3 -m json.tool

sleep 4

# Step 4: Check camera info
echo -e "\n[4] Checking camera info..."
curl -s http://localhost:8083/api/cameras | python3 -m json.tool

echo -e "\n✓ Test completed - verify URI was updated"
