# Fix GStreamer Decoder Buffer Overflow & Timestamp Issues

## Problem Analysis

### Symptoms
- Stream jitter/blur (giật hoặc nhòe)
- GStreamer warnings: "decreasing timestamp"
- GStreamer warnings: "Decoder is producing too many buffers"

### Root Cause
Located in `src/camera_manager.py:126-144` - **nvurisrcbin configuration**:

```python
source = make_element("nvurisrcbin", f"nvurisrc_{camera_id}", {
    "uri": uri,
    "drop-frame-interval": 0,  # ❌ PROBLEM: Never drops frames
    "latency": 500,             # ❌ PROBLEM: High latency (500ms) allows buffer buildup
    "num-extra-surfaces": 2,    # ⚠️  Extra decode buffers
})
```

**Why this causes issues:**
1. **`drop-frame-interval: 0`**: Decoder NEVER drops frames, causing buffer accumulation during network jitter
2. **`latency: 500ms`**: Allows up to 500ms of buffering (12-15 frames at 25fps), increasing jitter
3. **No proactive frame dropping**: Decoder produces buffers faster than pipeline can consume during RTSP network instability

### Technical Context
- **nvurisrcbin** is a DeepStream convenience element wrapping: `rtspsrc → nvv4l2decoder → appsink`
- It handles RTSP connection, decoding, and NVMM buffer management
- However, it doesn't expose fine-grained decoder controls
- Downstream queues (max-size-buffers=30, leaky=2) drop OLD buffers, but damage is already done

---

## Solution Strategy

### Option 1: Configure nvurisrcbin for Low Latency (RECOMMENDED)
**Pros**: Simple, one-line change, preserves DeepStream optimizations
**Cons**: Less granular control

**Changes to `src/camera_manager.py:126-144`:**

```python
source = make_element("nvurisrcbin", f"nvurisrc_{camera_id}", {
    "uri": uri,
    "gpu-id": self._gpu_id,
    "disable-audio": True,
    "source-id": source_id,
    "cudadec-memtype": 0,

    # FIX 1: Enable proactive frame dropping (drop 1 frame every 30 frames if late)
    "drop-frame-interval": 30,  # Changed from 0

    # FIX 2: Reduce latency buffer (lower jitter tolerance)
    "latency": 100,  # Changed from 500ms to 100ms

    # FIX 3: Minimize decode surface buffers
    "num-extra-surfaces": 0,  # Changed from 2 to 0
})
```

**Parameter explanations:**
- `drop-frame-interval: 30` - If decoder falls behind, drop 1 frame every 30 frames to catch up
- `latency: 100` - Reduce buffering window from 500ms (12 frames) to 100ms (2-3 frames)
- `num-extra-surfaces: 0` - Don't pre-allocate extra decode buffers

### Option 2: Switch to Raw Elements (MORE CONTROL)
**Pros**: Full control over rtspsrc, decoder, buffer management
**Cons**: More complex, requires significant refactoring

Replace nvurisrcbin with:
```
rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvvideoconvert
```

This allows setting:
- `rtspsrc` properties: `latency=100`, `drop-on-latency=true`, `do-timestamp=true`
- `nvv4l2decoder` properties: `enable-max-performance=true`, `bufapi-version=true`

**Recommendation**: Try Option 1 first, only implement Option 2 if issues persist.

---

## Implementation Plan

### Step 1: Update nvurisrcbin Configuration
**File**: `src/camera_manager.py`

**Location**: Lines 126-144 in `add_camera()` method

**Changes**:
```python
# BEFORE:
source = make_element("nvurisrcbin", f"nvurisrc_{camera_id}", {
    "uri": uri,
    "gpu-id": self._gpu_id,
    "disable-audio": True,
    "source-id": source_id,
    "cudadec-memtype": 0,
    "num-extra-surfaces": 2,
    "latency": 500,
    "drop-frame-interval": 0
})

# AFTER:
source = make_element("nvurisrcbin", f"nvurisrc_{camera_id}", {
    "uri": uri,
    "gpu-id": self._gpu_id,
    "disable-audio": True,
    "source-id": source_id,
    "cudadec-memtype": 0,
    "num-extra-surfaces": 0,      # Reduce decode buffer surfaces
    "latency": 100,                # Reduce latency from 500ms to 100ms
    "drop-frame-interval": 30      # Drop 1 frame per 30 if late
})
```

**Why these values:**
- `drop-frame-interval: 30` - Aggressive enough to prevent buildup, gentle enough to maintain quality
- `latency: 100ms` - Low latency for real-time streaming (2-3 frame buffer at 25fps)
- `num-extra-surfaces: 0` - Minimize memory/decode overhead

### Step 2: Add Logging for Monitoring
Add debug logging after decoder creation to track drop statistics:

```python
logger.info(f"[Camera] nvurisrcbin configured: latency=100ms, drop-interval=30, surfaces=0")
```

### Step 3: Optional - Add Configuration File Support
Create `configs/decoder_tuning.yaml` for easy adjustment:

```yaml
decoder:
  latency_ms: 100
  drop_frame_interval: 30
  num_extra_surfaces: 0

  # Tuning notes:
  # - Increase latency for unstable networks (200-300ms)
  # - Decrease drop_interval for aggressive dropping (10-20)
  # - Increase num_extra_surfaces for high-bitrate streams (1-2)
```

Update `camera_manager.py` to load from config if available.

---

## Verification Plan

### Test 1: Verify No GStreamer Warnings
**Steps:**
1. Restart pipeline: `docker exec -it -w /app vmsx bash entry/run_pipeline.sh`
2. Add camera: `curl -X POST http://localhost:8083/api/cameras -d '{"camera_id":"cam1","uri":"rtsp://192.168.6.14:8554/testface","branch":"detection"}'`
3. Monitor logs: `docker exec vmsx tail -f /tmp/pipeline.log | grep -E "decreasing timestamp|too many buffers|nvv4l2decoder"`
4. **Expected**: No warnings about timestamps or buffer overflow

### Test 2: Verify Stream Quality
**Steps:**
1. Start RTSP stream publish: `curl -X POST http://localhost:8083/api/cameras/cam1/branches/detection/stream/start -d '{"uri":"rtsp://143.198.198.52:8554/test","bitrate":4000000}'`
2. Play stream: `ffplay -rtsp_transport tcp rtsp://143.198.198.52:8554/test`
3. Observe for 2-3 minutes
4. **Expected**: No jitter, smooth playback, crisp image (not blurry)

### Test 3: Stress Test with Network Jitter
**Steps:**
1. Use `tc` (traffic control) to simulate network jitter:
   ```bash
   docker exec vmsx bash -c "tc qdisc add dev eth0 root netem delay 50ms 20ms"
   ```
2. Monitor pipeline for 5 minutes
3. Check logs for frame drops: `grep "drop" /tmp/pipeline.log`
4. **Expected**: Pipeline drops frames gracefully, no buffer overflow warnings
5. Clean up: `docker exec vmsx bash -c "tc qdisc del dev eth0 root"`

### Test 4: Long-Running Stability
**Steps:**
1. Run pipeline with stream for 30+ minutes
2. Monitor memory usage: `docker stats vmsx`
3. Check for memory leaks or CMA exhaustion
4. **Expected**: Stable memory usage, no crashes

### Success Criteria
- ✅ No "decreasing timestamp" warnings
- ✅ No "too many buffers" warnings
- ✅ Stream playback is smooth (no jitter)
- ✅ Image is crisp (not blurry)
- ✅ Pipeline handles network jitter gracefully
- ✅ Memory usage remains stable

---

## Rollback Plan

If the changes cause issues:

1. **Revert to original settings** in `camera_manager.py`:
   ```python
   "latency": 500,
   "drop-frame-interval": 0,
   "num-extra-surfaces": 2
   ```

2. **If stream quality degrades** (too aggressive dropping):
   - Increase `drop-frame-interval` to 60 or 90
   - Increase `latency` to 200-300ms
   - Increase `num-extra-surfaces` to 1

3. **If Option 1 doesn't work**, implement Option 2 (raw elements) for fine-grained control

---

## Critical Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/camera_manager.py` | 126-144 | nvurisrcbin configuration (MAIN FIX) |
| `configs/decoder_tuning.yaml` | (new) | Optional: Tunable decoder params |
| `tests/test_decoder_quality.sh` | (new) | Optional: Automated quality test |

---

## Additional Notes

### Why Not Modify Queues?
Queues with `leaky=2` already drop old buffers, but this is **reactive** (after damage). The fix needs to be **proactive** at the decoder source.

### Why Not Modify rtspclientsink?
The jitter/blur happens on the **input** side (RTSP source → decoder), not the **output** side (publisher → rtspclientsink).

### Alternative: Use Jetson Hardware Decoder
If running on Jetson, consider using hardware-accelerated decoder properties:
```python
"enable-max-performance": 1  # Jetson nvv4l2decoder only
```

### Tuning Guidelines
- **Stable network**: `latency=100, drop-interval=30`
- **Unstable network**: `latency=200, drop-interval=20` (more aggressive dropping)
- **High-bitrate 4K**: `latency=300, num-extra-surfaces=1` (more buffering)
