#!/bin/bash
# Script to clean up CMA memory on Jetson

echo "=== CMA Memory Status BEFORE ==="
cat /proc/meminfo | grep Cma

echo ""
echo "=== Killing zombie GStreamer/Python processes ==="
# Kill all Python processes related to the pipeline
sudo pkill -9 -f 'python.*test_multi'
sudo pkill -9 -f 'python.*entry/'
# Kill any hanging GStreamer processes
sudo pkill -9 -f gst-launch
sudo pkill -9 -f nvargus

echo ""
echo "=== Stopping and restarting Docker container ==="
cd "$(dirname "$0")" || exit 1
sudo docker compose restart qv_face

echo ""
echo "=== Clearing kernel buffer cache (safe, won't lose data) ==="
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo ""
echo "=== CMA Memory Status AFTER ==="
cat /proc/meminfo | grep Cma

echo ""
echo "=== Memory usage summary ==="
free -h
