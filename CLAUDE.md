# VMSx - VMS Core Pipeline - Project Guide

## Quick Reference

| Item | Value |
|------|-------|
| Container | `qv_face` |
| Host Path | `/home/mq/disk2T/quangnv/rtc_events` |
| Container Path | `/app` |
| API Port | `8083` |
| Python | `3.10.12` |
| OS | Ubuntu 22.04 |

## Docker Execution

### Enter Container (Interactive Shell)

```bash
docker exec -it -w /app qv_face bash
```

### Run Commands Inside Container

```bash
# Single command
docker exec -w /app qv_face <command>

# With bash
docker exec -w /app qv_face bash -c "<commands>"
```

## Debug & Test

### Run Pipeline (ALWAYS Use This Method)

**IMPORTANT: Always use `entry/run_pipeline.sh` to start the pipeline. This script automatically cleans up old processes and ports.**

```bash
# From host (recommended) - Interactive mode with terminal output
docker exec -it -w /app qv_face bash entry/run_pipeline.sh

# From host - Background mode with log file
docker exec -d -w /app qv_face bash -c "bash entry/run_pipeline.sh > /tmp/pipeline.log 2>&1"
docker exec qv_face tail -f /tmp/pipeline.log

# Inside container (after docker exec -it -w /app qv_face bash)
bash entry/run_pipeline.sh
```

**Why use `run_pipeline.sh`?**
- Auto-kills old Python processes and frees port 8083
- Prevents "port already in use" errors
- Ensures clean pipeline restart
- Safer than manual Python execution

### Run Pipeline (Manual - Not Recommended)

Only use direct Python execution if you need custom configuration:

```bash
# Inside container
python3 entry/test_multi_branch_video.py

# With custom config
python3 entry/test_multi_branch_video.py --config configs/test-single-branch.yaml
```

**WARNING: Manual execution may fail if port 8083 is already in use. Use `run_pipeline.sh` instead.**

### Test Scripts

```bash
# Simple test: Add camera + Start stream
./tests/add_remove_readd_stream_cam1.sh

# Full test (from host)
./entry/full_test_video.sh
```

## API Endpoints

Base URL: `http://localhost:8083`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Quick health check |
| GET | `/api/status` | **Detailed pipeline status (debugging)** |
| GET | `/api/cameras` | List cameras |
| GET | `/api/branches` | List branches |
| GET | `/api/operations/{op_id}` | **Get operation result (with errors)** |
| POST | `/api/cameras` | Add camera |
| DELETE | `/api/cameras/{id}` | Remove camera |
| POST | `/api/cameras/{id}/branches/{branch}` | Add to branch |
| DELETE | `/api/cameras/{id}/branches/{branch}` | Remove from branch |
| POST | `/api/cameras/{id}/branches/{branch}/stream/start` | Start stream |
| POST | `/api/cameras/{id}/branches/{branch}/stream/stop` | Stop stream |
| GET | `/api/cameras/{id}/branches/{branch}/stream` | Stream status |
| GET | `/api/streams` | All streams status |
| POST | `/api/pipeline/kill` | Remove all cameras |
| POST | `/api/pipeline/stop` | Stop pipeline |

### API Examples

```bash
# Health check
curl http://localhost:8083/api/health

# Add camera (with 1 branch)
curl -X POST http://localhost:8083/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam1", "uri": "rtsp://192.168.6.14:8554/testface", "branch": "detection"}'

# Add camera to another branch
curl -X POST http://localhost:8083/api/cameras/cam1/branches/recognition

# List cameras
curl http://localhost:8083/api/cameras | python3 -m json.tool

# Start stream
curl -X POST http://localhost:8083/api/cameras/cam1/branches/detection/stream/start \
  -H "Content-Type: application/json" \
  -d '{"uri": "rtsp://192.168.6.14:8554/cam1_detection", "bitrate": 4000000}'

# Stop stream
curl -X POST http://localhost:8083/api/cameras/cam1/branches/detection/stream/stop

# List all streams
curl http://localhost:8083/api/streams
```

### Debugging APIs (NEW)

```bash
# Get detailed pipeline status (state, cameras, streams, operations)
curl http://localhost:8083/api/status | python3 -m json.tool

# Check operation result (includes error details, traceback)
curl http://localhost:8083/api/operations/{operation_id} | python3 -m json.tool

# Example operation response with error:
# {
#   "status": "error",
#   "operation_type": "stream_start",
#   "duration": 5.23,
#   "timestamp": 1234567890.12,
#   "error": "Pipeline failed to return to PLAYING state",
#   "error_type": "RuntimeError",
#   "traceback": "..."
# }
```

## Project Structure

```
/app/
├── apps/             # Processor apps (detection, face)
│   ├── detection/    # Detection processor
│   └── face/         # Face recognition processor
├── configs/          # Pipeline configurations
├── data/             # Data files, output videos
├── entry/            # Entry points
├── scripts/          # Docker management scripts
├── src/              # Core pipeline modules
│   ├── api/          # REST API server
│   ├── camera_manager.py
│   ├── common.py
│   ├── pipeline_builder.py
│   └── stream_publisher.py
└── tests/            # Test scripts
```

## Debugging Tips

### Check Container Status

```bash
docker ps -a --filter "name=qv_face"
```

### View Logs

```bash
# Container logs
docker logs qv_face --tail 100

# Pipeline logs
docker exec qv_face tail -f /tmp/pipeline.log
docker exec qv_face tail -f /tmp/full_test.log
```

### Kill Pipeline Processes

```bash
# Recommended: Use the run_pipeline.sh script (auto cleanup)
docker exec -w /app qv_face bash entry/run_pipeline.sh

# Manual kill (if needed)
docker exec qv_face pkill -9 -f 'python.*test_multi'
docker exec qv_face fuser -k 8083/tcp  # Free port 8083
```

### Check Running Processes

```bash
docker exec qv_face ps aux | grep python
```

### Container Restart

```bash
cd scripts && docker compose restart
# or
./scripts/stop.sh && ./scripts/start.sh
```

## Common Issues

1. **Container unhealthy**: Check `docker logs qv_face` for errors
2. **Port 8083 in use**: Use `entry/run_pipeline.sh` (auto cleanup) or manually kill with `docker exec qv_face fuser -k 8083/tcp`
3. **Pipeline crash**: Check `/tmp/pipeline.log` inside container
4. **RTSP timeout**: Verify camera URI is reachable from container
5. **Multiple pipeline instances**: Always use `entry/run_pipeline.sh` which auto-kills old processes before starting

## Hardware Monitoring

### Monitor GPU & System Resources

**For dGPU (T4, A100, etc.) on Ubuntu - Run in Docker:**

```bash
# From host - Monitor GPU inside container
docker exec -it qv_face bash scripts/monitor_dpu.sh

# What it monitors:
# - System RAM (used/total GB, %)
# - CPU Usage (%)
# - GPU Utilization (%)
# - GPU Memory (used/total MB, %)
# - GPU Temperature (°C)

# Warning indicators:
# ⚡ Yellow (80-90% usage)
# ⚠️  Red (>90% usage)
```

**For Jetson (Orin/Xavier) - Run directly (NO Docker):**

```bash
# Run directly on Jetson host
bash scripts/monitor_jetson.sh

# What it monitors:
# - CMA Memory (MB, %) - Important for DeepStream
# - System RAM (MB, %)
# - CPU Usage (%)
# - GPU Usage (%)
# - Temperature (°C)

# Warning indicators:
# ⚡ Yellow (80-90% usage)
# ⚠️  Red (>90% usage)
```

**Platform Differences:**

| Feature | dGPU (T4) | Jetson (Orin) |
|---------|-----------|---------------|
| Execution | Inside Docker | Direct on host (NO Docker) |
| Tool | `nvidia-smi` | `tegrastats` |
| GPU Memory | Dedicated VRAM | Shared with system RAM |
| CMA Memory | N/A | Critical for zero-copy |
| Script | `monitor_dpu.sh` | `monitor_jetson.sh` |

**Why Jetson doesn't use Docker:**
- DeepStream requires direct hardware access (tegrastats, CMA)
- GPU memory is shared with system RAM (unified memory)
- Performance overhead from containerization

## Best Practices

1. **Always start pipeline via**: `docker exec -it -w /app qv_face bash entry/run_pipeline.sh`
2. **Run commands in Docker**: Use `docker exec -w /app qv_face` for all operations
3. **Check logs**: Monitor `/tmp/pipeline.log` for debugging
4. **Clean restart**: The `run_pipeline.sh` script handles cleanup automatically
5. **Monitor resources**: Use appropriate monitoring script for your platform (see Hardware Monitoring section)
