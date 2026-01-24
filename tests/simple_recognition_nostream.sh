#!/bin/bash
# Test: Add 
# Scenario:
#   1. Add cam1 to branch recognition
#   2. Add cam2 to branch recognition

CAMERA_URI="rtsp://192.168.6.14:8554/testface"
STREAM_SERVER="152.42.221.89"
STREAM_PORT="8554"
STREAM_NAME="cam1_recognition"
STREAM_URI="rtsp://${STREAM_SERVER}:${STREAM_PORT}/${STREAM_NAME}"

echo "============================================================"
echo "  TEST: Add"
echo "============================================================"

# Step 1: Add cam1 to branch recognition
echo -e "\n[1] Adding cam1 to branch recognition..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam1\",\"uri\":\"${CAMERA_URI}\",\"branch\":\"recognition\"}" \
  | python3 -m json.tool
sleep 2
# Step 2: Add cam2 to branch recognition
echo -e "\n[2] Adding cam2 to branch recognition..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam2\",\"uri\":\"${CAMERA_URI}\",\"branch\":\"recognition\"}" \
  | python3 -m json.tool

sleep 2
# Step 2: Add cam3 to branch recognition
echo -e "\n[2] Adding cam3 to branch recognition..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam3\",\"uri\":\"${CAMERA_URI}\",\"branch\":\"recognition\"}" \
  | python3 -m json.tool

sleep 2
# Step 2: Add cam4 to branch recognition
echo -e "\n[2] Adding cam4 to branch recognition..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam4\",\"uri\":\"${CAMERA_URI}\",\"branch\":\"recognition\"}" \
  | python3 -m json.tool

sleep 2
# Step 2: Add cam5 to branch recognition
echo -e "\n[2] Adding cam5 to branch recognition..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam5\",\"uri\":\"${CAMERA_URI}\",\"branch\":\"recognition\"}" \
  | python3 -m json.tool
sleep 2
# Step 2: Add cam6 to branch recognition
echo -e "\n[2] Adding cam6 to branch recognition..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam6\",\"uri\":\"${CAMERA_URI}\",\"branch\":\"recognition\"}" \
  | python3 -m json.tool


sleep 2
# Step 2: Add cam7 to branch recognition
echo -e "\n[2] Adding cam7 to branch recognition..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam7\",\"uri\":\"${CAMERA_URI}\",\"branch\":\"recognition\"}" \
  | python3 -m json.tool


sleep 2
# Step 2: Add cam8 to branch recognition
echo -e "\n[2] Adding cam8 to branch recognition..."
curl -s -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam8\",\"uri\":\"${CAMERA_URI}\",\"branch\":\"recognition\"}" \
  | python3 -m json.tool
