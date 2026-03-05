#!/bin/bash
# Start pipeline and camera sync worker in background
# Usage: docker exec -w /app vmsx bash entry/start_all_background.sh

set -e

LOG_DIR="data/vms_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Starting VMSx Services in Background"
echo "============================================================"

# 1. Start pipeline
echo "[1/2] Starting pipeline..."
nohup bash entry/run_pipeline.sh > "$LOG_DIR/pipeline.log" 2>&1 &
PIPELINE_PID=$!
echo "Pipeline started: PID=$PIPELINE_PID, log=$LOG_DIR/pipeline.log"

# 2. Wait for API to be ready
echo "[2/2] Waiting for API (port 8085)..."
for i in {1..30}; do
    if curl -s http://localhost:8085/api/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ API failed to start after 30s"
        exit 1
    fi
    sleep 1
done

# 3. Start camera sync worker
echo "[3/3] Starting camera DB sync worker..."
nohup bash entry/camera_db_sync_external.sh > "$LOG_DIR/camera_sync.log" 2>&1 &
SYNC_PID=$!
echo "Camera sync started: PID=$SYNC_PID, log=$LOG_DIR/camera_sync.log"

echo ""
echo "============================================================"
echo "  All Services Started Successfully"
echo "============================================================"
echo "Pipeline PID:     $PIPELINE_PID"
echo "Camera Sync PID:  $SYNC_PID"
echo ""
echo "View logs:"
echo "  Pipeline:      tail -f $LOG_DIR/pipeline.log"
echo "  Camera Sync:   tail -f $LOG_DIR/camera_sync.log"
echo ""
echo "Check status:"
echo "  ps aux | grep -E 'test_multi|camera_db_sync'"
echo ""
echo "Stop services:"
echo "  pkill -f 'test_multi'"
echo "  pkill -f 'camera_db_sync'"
echo "============================================================"