# Face Stream Pipeline - Project Guide

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

### Run Pipeline

```bash
# Inside container
python3 entry/test_multi_branch_video.py

# With custom config
python3 entry/test_multi_branch_video.py --config configs/test-single-branch.yaml

# From host
docker exec -w /app qv_face python3 entry/test_multi_branch_video.py
```

### Background Run (with logs)

```bash
# Start in background
docker exec -d -w /app qv_face bash -c "python3 entry/test_multi_branch_video.py > /tmp/pipeline.log 2>&1"

# View logs
docker exec qv_face tail -f /tmp/pipeline.log
```

### Full Test Script

```bash
# From host (outside container)
./entry/full_test_video.sh
```

## API Endpoints

Base URL: `http://localhost:8083`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/cameras` | List cameras |
| GET | `/api/branches` | List branches |
| POST | `/api/cameras` | Add camera |
| DELETE | `/api/cameras/{id}` | Remove camera |
| POST | `/api/cameras/{id}/branches/{branch}` | Add to branch |
| DELETE | `/api/cameras/{id}/branches/{branch}` | Remove from branch |
| POST | `/api/cameras/{id}/branches/{branch}/stream/start` | Start stream |
| POST | `/api/cameras/{id}/branches/{branch}/stream/stop` | Stop stream |
| GET | `/api/cameras/{id}/branches/{branch}/stream` | Stream status |
| GET | `/api/streams` | All streams status |
| GET | `/api/operations/{op_id}` | Operation status |
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
  -d '{"uri": "srt://192.168.6.14:8890?streamid=publish:cam1_detection", "bitrate": 4000000}'

# Stop stream
curl -X POST http://localhost:8083/api/cameras/cam1/branches/detection/stream/stop

# List all streams
curl http://localhost:8083/api/streams
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
docker exec qv_face pkill -9 -f 'python.*test_multi'
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
2. **Port 8083 in use**: Run `lsof -i :8083` and kill the process
3. **Pipeline crash**: Check `/tmp/pipeline.log` inside container
4. **RTSP timeout**: Verify camera URI is reachable from container
