#!/bin/bash
# Stress Test: Multiple cameras add -> remove -> re-add -> stream
# Scenario:
#   1. Add cam1-cam8 to branch detection
#   2. Remove cam6, cam7, cam8
#   3. Re-add cam6, cam7, cam8
#   4. Stream cam1 detection to RTSP server

CAMERA_URI="rtsp://192.168.6.14:8554/testface"
STREAM_SERVER="152.42.221.89"
STREAM_PORT="8554"
STREAM_NAME_CAM1="cam1_detection"
STREAM_NAME_CAM2="cam2_detection"
STREAM_URI_CAM1="rtsp://${STREAM_SERVER}:${STREAM_PORT}/${STREAM_NAME_CAM1}"
STREAM_URI_CAM2="rtsp://${STREAM_SERVER}:${STREAM_PORT}/${STREAM_NAME_CAM2}"

# Function to wait for operation to complete
wait_for_operation() {
  local op_id=$1
  local max_wait=${2:-30}  # Default 30 seconds timeout
  local wait_time=0

  while [ $wait_time -lt $max_wait ]; do
    result=$(curl -s http://localhost:8083/api/operations/${op_id})
    status=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)

    if [ "$status" = "ok" ] || [ "$status" = "completed" ]; then
      echo "  ✓ Operation $op_id completed"
      return 0
    elif [ "$status" = "error" ]; then
      echo "  ✗ Operation $op_id failed:"
      echo "$result" | python3 -m json.tool
      return 1
    fi

    sleep 1
    wait_time=$((wait_time + 1))
  done

  echo "  ⚠ Operation $op_id timeout after ${max_wait}s"
  return 2
}

echo "============================================================"
echo "  STRESS TEST: Multiple Cameras Add/Remove/Stream"
echo "============================================================"

# Step 1: Add cam1-cam8 to branch detection
echo -e "\n[1] Adding cam1-cam8 to branch detection..."
for i in {1..8}; do
  echo "Adding cam${i}..."
  response=$(curl -s -X POST http://localhost:8083/api/cameras \
    -H "Content-Type: application/json" \
    -d '{"camera_id":"cam'$i'","uri":"'$CAMERA_URI'","branch":"detection"}')
  echo "$response" | python3 -m json.tool

  op_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('operation_id',''))" 2>/dev/null)
  if [ -n "$op_id" ]; then
    wait_for_operation "$op_id"
  fi
done

echo -e "\n[1.1] Listing all cameras..."
curl -s http://localhost:8083/api/cameras | python3 -m json.tool

# Step 2: Remove cam6, cam7, cam8
echo -e "\n[2] Removing cam6, cam7, cam8..."
for i in {6..8}; do
  echo "Removing cam${i}..."
  response=$(curl -s -X DELETE http://localhost:8083/api/cameras/cam${i})
  echo "$response" | python3 -m json.tool

  op_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('operation_id',''))" 2>/dev/null)
  if [ -n "$op_id" ]; then
    wait_for_operation "$op_id"
  fi
done

echo -e "\n[2.1] Listing cameras after removal..."
curl -s http://localhost:8083/api/cameras | python3 -m json.tool

# Step 3: Re-add cam6, cam7, cam8
echo -e "\n[3] Re-adding cam6, cam7, cam8 to branch detection..."
for i in {6..8}; do
  echo "Re-adding cam${i}..."
  response=$(curl -s -X POST http://localhost:8083/api/cameras \
    -H "Content-Type: application/json" \
    -d '{"camera_id":"cam'$i'","uri":"'$CAMERA_URI'","branch":"detection"}')
  echo "$response" | python3 -m json.tool

  op_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('operation_id',''))" 2>/dev/null)
  if [ -n "$op_id" ]; then
    wait_for_operation "$op_id"
  fi
done

echo -e "\n[3.1] Listing all cameras after re-adding..."
curl -s http://localhost:8083/api/cameras | python3 -m json.tool

# Step 4: Stream cam1 detection
echo -e "\n[4] Starting stream cam1_detection to $STREAM_URI_CAM1..."
response=$(curl -s -X POST http://localhost:8083/api/cameras/cam1/branches/detection/stream/start \
  -H "Content-Type: application/json" \
  -d '{"uri":"'$STREAM_URI_CAM1'","bitrate":4000000}')
echo "$response" | python3 -m json.tool

op_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('operation_id',''))" 2>/dev/null)
if [ -n "$op_id" ]; then
  wait_for_operation "$op_id" 60  # Stream may take longer
fi

sleep 3  # Small buffer for stream to stabilize

# Step 5: Stream cam2 detection
echo -e "\n[5] Starting stream cam2_detection to $STREAM_URI_CAM2..."
response=$(curl -s -X POST http://localhost:8083/api/cameras/cam2/branches/detection/stream/start \
  -H "Content-Type: application/json" \
  -d '{"uri":"'$STREAM_URI_CAM2'","bitrate":4000000}')
echo "$response" | python3 -m json.tool

op_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('operation_id',''))" 2>/dev/null)
if [ -n "$op_id" ]; then
  wait_for_operation "$op_id" 60  # Stream may take longer
fi

sleep 3  # Small buffer for stream to stabilize

# Verify stream cam1
echo -e "\n[6] Verifying stream cam1_detection..."
if ffprobe -v quiet -rtsp_transport tcp -stimeout 8000000 \
   -show_entries stream=codec_name -of default=nw=1 "$STREAM_URI_CAM1" 2>&1 | grep -q codec_name; then
    echo "✓ Stream cam1 is LIVE: $STREAM_URI_CAM1"
    ffprobe -v quiet -rtsp_transport tcp -stimeout 8000000 \
      -show_entries stream=codec_name,width,height -of json "$STREAM_URI_CAM1" 2>/dev/null | python3 -m json.tool
else
    echo "✗ Stream cam1 NOT live: $STREAM_URI_CAM1"
fi

# Verify stream cam2
echo -e "\n[7] Verifying stream cam2_detection..."
if ffprobe -v quiet -rtsp_transport tcp -stimeout 8000000 \
   -show_entries stream=codec_name -of default=nw=1 "$STREAM_URI_CAM2" 2>&1 | grep -q codec_name; then
    echo "✓ Stream cam2 is LIVE: $STREAM_URI_CAM2"
    ffprobe -v quiet -rtsp_transport tcp -stimeout 8000000 \
      -show_entries stream=codec_name,width,height -of json "$STREAM_URI_CAM2" 2>/dev/null | python3 -m json.tool
    exit 0
else
    echo "✗ Stream cam2 NOT live: $STREAM_URI_CAM2"
    exit 1
fi
