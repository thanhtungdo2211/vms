# VMSx - Comprehensive VMS Core

GPU-accelerated Video Management System core built on NVIDIA DeepStream 7.1.

## Overview

VMSx is a comprehensive VMS (Video Management System) core leveraging NVIDIA DeepStream SDK. It provides a flexible, multi-branch pipeline architecture for real-time video analytics including object detection, face recognition, tracking, and streaming via RTSP to MediaMTX server.

**Key Features:**
- **Multi-branch architecture**: Single decode, multiple processing pipelines (detection, recognition, recording)
- **GPU-accelerated analytics**: Real-time object detection, face detection (SCRFD 2.5G), face recognition (ArcFace ResNet-100)
- **Advanced tracking**: NvDCF visual tracker with per-camera isolation
- **Flexible streaming**: Low-latency RTSP output to MediaMTX server with H.264
- **REST API**: Dynamic camera management (add/remove/configure)
- **Scalable**: Multi-camera support with hardware-accelerated processing

## Prerequisites

- NVIDIA GPU (T4, A100, etc.) or Jetson (Orin/Xavier)
- DeepStream SDK 7.1
- Python 3.10+
- CUDA 12.6 / TensorRT 10.x
- MediaMTX server for RTSP streaming

## Quick Start

### 1. Setup Environment

```bash
# Run setup script (for dGPU, use Docker)
cd scripts
docker compose up -d

# For Jetson (direct execution)
bash setup_7_1_jetson.sh
```

### 2. Configure

Edit your YAML config file (e.g., `configs/multi-branch.yaml`) to set:
- Input camera URIs (RTSP sources)
- Output RTSP URIs (to MediaMTX server)
- Branch configurations (detection, recognition)

### 3. Run

```bash
# For dGPU (inside Docker container)
docker exec -it -w /app vmsx bash entry/run_pipeline.sh

# For Jetson (direct execution)
python3 entry/test_multi_branch_video.py
```

## Project Structure

```
VMSx/
├── src/                     # Core VMS modules
│   ├── api/                 # REST API for camera/stream management
│   │   └── api_server.py    # FastAPI/aiohttp endpoints
│   ├── camera_manager.py    # Dynamic camera CRUD operations
│   ├── pipeline_builder.py  # Multi-branch pipeline construction
│   ├── stream_publisher.py  # RTSP stream management
│   ├── common.py            # Shared utilities
│   └── sinks/               # Output adapters
│       ├── base_sink.py     # Abstract sink interface
│       ├── fakesink_adapter.py  # Testing sink
│       └── filesink_adapter.py  # MP4 recording
├── apps/                    # Analytics applications
│   ├── detection/           # Object detection processor
│   └── face/                # Face recognition application
│       ├── database.py      # Feature DB with L2 matching
│       ├── tracker.py       # Multi-object tracking with voting
│       ├── events.py        # Recognition events
│       ├── display.py       # OSD rendering
│       └── probes.py        # GStreamer buffer probes
├── entry/                   # Entry points
│   ├── test_multi_branch_video.py  # Main pipeline launcher
│   ├── run_pipeline.sh      # Startup script with cleanup
│   └── full_test_video.sh   # Integration test script
├── configs/                 # YAML configurations
│   ├── multi-branch.yaml    # Multi-branch pipeline config
│   └── test-*.yaml          # Test configurations
├── data/                    # Models and data
│   ├── face/models/         # Face detection/recognition models
│   │   ├── scrfd640/        # SCRFD face detector (PGIE)
│   │   ├── arcface/         # ArcFace embeddings (SGIE)
│   │   └── NvDCF/           # Tracker configuration
│   └── features.json        # Face database
├── scripts/                 # DevOps scripts
│   ├── docker-compose.yml   # Docker deployment
│   ├── Dockerfile           # Container image
│   ├── monitor_dpu.sh       # GPU monitoring (dGPU)
│   ├── monitor_jetson.sh    # System monitoring (Jetson)
│   ├── setup_7_1_jetson.sh  # Jetson environment setup
│   └── *.sh                 # Management scripts
├── tests/                   # Test scripts
└── docs/                    # Documentation (if exists)
```

## Architecture

**Multi-Branch Tee Fanout Pipeline** (single decode, zero-copy distribution):

```
Camera Input → nvurisrcbin (H.264/H.265 decode) → tee (fanout)
                                                    ├─→ Branch A (Face Recognition) → RTSP (MediaMTX)
                                                    ├─→ Branch B (Detection Only) → File/RTSP
                                                    └─→ Branch C (Recording) → MP4

Each Branch Pipeline:
  PGIE (Detection) → NvDCF Tracker → SGIE (Classification/Recognition) → OSD → Encoder → Sink
       ↓
  Object Metadata → Custom Probes → Events/Analytics
```

**Architecture Highlights**:

1. **Dynamic Camera Management**
   - REST API for runtime camera add/remove
   - Per-camera pipeline isolation (independent processing)
   - No pipeline restart required for camera changes

2. **Multi-Branch Processing**
   - Single video decode, multiple independent processing branches
   - Hardware-accelerated buffer copy (nvvideoconvert) prevents tearing
   - Each branch can have different: models, trackers, outputs

3. **GPU-Accelerated Analytics**
   - PGIE: Primary detection (faces, objects)
   - SGIE: Secondary classification (face recognition, attributes)
   - NvDCF Tracker: Visual tracking with per-camera isolation

4. **Flexible Output**
   - RTSP streaming to MediaMTX (rtspclientsink)
   - MP4 file recording (filesink)
   - Configurable bitrate, resolution, framerate

5. **Event System**
   - Real-time recognition events
   - Voting-based confirmation (streak threshold)
   - Per-object tracking with state machines

## Configuration

Key parameters in YAML config files (e.g., `configs/multi-branch.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `uri` | - | Input camera RTSP URI |
| `output_uri` | - | Output RTSP URI (to MediaMTX) |
| `bitrate` | 4000000 | H.264 encoding bitrate |
| `l2_threshold` | 1.20 | Max L2 distance for matching |
| `vote_threshold` | 3 | Votes needed for confirmation |
| `vote_window_frames` | 90 | Sliding window size |

## Analytics Applications

### Face Recognition

Add registered faces to `data/features.json`:

```json
{
    "PersonName": {
        "feature": [0.1, 0.2, ...],  // 512-dim ArcFace embedding
        "avatar": "<base64_encoded_image>"
    }
}
```

Features are 512-dimensional normalized L2 vectors from ArcFace ResNet-100.

### Object Detection

Configure detection models in YAML:
- Primary detector (PGIE): SCRFD, YOLO, etc.
- Secondary classifier (SGIE): ArcFace, custom models
- Tracker: NvDCF, DeepSORT, IOU

### Custom Applications

Add custom analytics by:
1. Creating processor in `apps/<name>/`
2. Implementing GStreamer probes for metadata extraction
3. Configuring in YAML branch definition

## Video Sources

Supported input formats:
- Local files: `file:///path/to/video.mp4`
- RTSP streams: `rtsp://user:pass@ip:port/path`
- HTTP streams: `http://example.com/stream`

## RTSP Output

The pipeline streams processed video to MediaMTX server via RTSP:
- Output URI format: `rtsp://<mediamtx-ip>:8554/<stream-name>`
- Supports multiple concurrent streams (multi-branch)
- H.264 encoding with configurable bitrate
- View streams using any RTSP client (VLC, ffplay, etc.)

## API Reference

VMSx provides REST API for dynamic camera and stream management. See `CLAUDE.md` for complete API documentation.

**Key Endpoints:**
- `GET /api/health` - Health check
- `GET /api/status` - Pipeline status
- `POST /api/cameras` - Add camera
- `POST /api/cameras/{id}/branches/{branch}/stream/start` - Start RTSP stream
- `DELETE /api/cameras/{id}` - Remove camera

Base URL: `http://localhost:8083`

## Hardware Monitoring

### Monitor System Resources

The project includes monitoring scripts for different platforms:

**For dGPU (T4, A100, etc.) on Ubuntu with Docker:**

```bash
# Run inside Docker container
docker exec -it vmsx bash scripts/monitor_dpu.sh

# Monitors: System RAM, CPU, GPU Util, GPU Memory, GPU Temp
# Uses: nvidia-smi
```

**For Jetson (Orin/Xavier) - Direct execution (NO Docker):**

```bash
# Run directly on Jetson host
bash scripts/monitor_jetson.sh

# Monitors: CMA Memory, System RAM, CPU, GPU, Temperature
# Uses: tegrastats
```

**Platform Comparison:**

| Platform | Script | Tool | GPU Memory | Docker |
|----------|--------|------|------------|--------|
| dGPU (T4) | `monitor_dpu.sh` | `nvidia-smi` | Dedicated VRAM | ✅ Yes |
| Jetson | `monitor_jetson.sh` | `tegrastats` | Shared/Unified | ❌ No |

**Key Differences:**
- **dGPU**: Runs in Docker, uses dedicated GPU memory
- **Jetson**: Direct execution, unified memory architecture, CMA monitoring critical for DeepStream

Both scripts show warning indicators:
- ⚡ Yellow (80-90% usage)
- ⚠️ Red (>90% usage)

## Troubleshooting

### RTSP Streaming Issues
- Verify MediaMTX server is running and accessible
- Check output URI is correctly formatted
- Ensure network allows RTSP port (default 8554)
- Test with RTSP client: `ffplay rtsp://<ip>:8554/<stream>`

### Recognition Not Working
- Check `features.json` is properly formatted
- Delete `features.cache.npz` to force cache rebuild
- Enable `debug_voting: True` in config

### Low FPS
- Reduce input resolution via `muxer_width/height`
- Increase `skip_reid` to reduce SGIE calls
- Check GPU utilization:
  - **dGPU**: `nvidia-smi` or `scripts/monitor_dpu.sh` (in Docker)
  - **Jetson**: `tegrastats` or `scripts/monitor_jetson.sh` (on host)

## License

MIT License

## Contact

**QuangVan**
- Email: vanquang.tpa@gmail.com
- Phone: 0395077199

For questions, issues, or collaboration inquiries, please reach out via email.
