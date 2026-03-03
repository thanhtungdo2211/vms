#!/bin/bash
# Safe pipeline runner - kills old processes and port before starting
# Usage (inside container): bash entry/run_pipeline.sh

set -e

PORT="${API_PORT:-8085}"

echo "============================================================"
echo "  Starting Pipeline with Cleanup"
echo "============================================================"

# Kill old Python processes
echo "[1/3] Killing old Python processes..."
pkill -9 -f 'python.*test_multi' 2>/dev/null || true
sleep 1

# Kill processes using API port
echo "[2/3] Freeing port $PORT..."
lsof -ti :$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1

# Verify port is free
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "ERROR: Port $PORT still in use!"
    lsof -i :$PORT
    exit 1
fi

# Start pipeline (output to terminal)
echo "[3/3] Starting pipeline..."
echo "============================================================"
echo ""

exec env GST_DEBUG="${GST_DEBUG:-0}" python3 entry/test_multi_branch_video.py
