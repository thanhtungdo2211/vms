#!/bin/bash
# RTSP Publishing Test Script
# Tests start/stop RTSP publishing to MediaMTX server
# Run outside Docker - manages container pipeline

DOCKER_CONTAINER="${DOCKER_CONTAINER:-qv_face}"
PROJECT_DIR="/app"
PORT="${API_PORT:-8084}"
MEDIAMTX_HOST="${MEDIAMTX_HOST:-192.168.6.14}"
MEDIAMTX_PORT="${MEDIAMTX_PORT:-8554}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

API_URL="http://localhost:$PORT"
RTSP_BASE="rtsp://$MEDIAMTX_HOST:$MEDIAMTX_PORT"

# Check if container is running
check_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
        log_error "Container '$DOCKER_CONTAINER' is not running!"
        exit 1
    fi
    log_info "Container '$DOCKER_CONTAINER' is running"
}

# Check API health
check_health() {
    curl -s --connect-timeout 3 "$API_URL/api/health" 2>/dev/null | grep -q "healthy"
}

# Wait for API to be ready
wait_for_api() {
    log_info "Waiting for API at $API_URL..."
    for i in $(seq 1 30); do
        if check_health; then
            log_info "API ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    log_error "API not ready after 30s"
    return 1
}

# Start pipeline with demux config
start_pipeline_demux() {
    log_step "Starting Pipeline (Demux Mode)"

    # Kill existing pipeline
    docker exec -w "$PROJECT_DIR" "$DOCKER_CONTAINER" bash -c "pkill -9 -f 'python.*test_multi' 2>/dev/null || true"
    sleep 2

    # Start new pipeline with demux config
    docker exec -d -w "$PROJECT_DIR" "$DOCKER_CONTAINER" bash -c \
        "API_PORT=$PORT python3 entry/test_multi_branch_video.py --config configs/multi-branch-demux.yaml > /tmp/publish_test.log 2>&1"

    wait_for_api || exit 1
}

# Start pipeline with normal config
start_pipeline() {
    log_step "Starting Pipeline"

    # Kill existing pipeline
    docker exec -w "$PROJECT_DIR" "$DOCKER_CONTAINER" bash -c "pkill -9 -f 'python.*test_multi' 2>/dev/null || true"
    sleep 2

    # Start new pipeline
    docker exec -d -w "$PROJECT_DIR" "$DOCKER_CONTAINER" bash -c \
        "API_PORT=$PORT python3 entry/test_multi_branch_video.py > /tmp/publish_test.log 2>&1"

    wait_for_api || exit 1
}

# Add camera
add_camera() {
    local cam_id=$1
    local uri=$2
    local branches=$3

    log_info "Adding camera: $cam_id -> $uri"
    result=$(curl -s -X POST "$API_URL/api/cameras" \
        -H "Content-Type: application/json" \
        -d "{\"camera_id\": \"$cam_id\", \"uri\": \"$uri\", \"branches\": $branches}")
    echo "  Response: $result"
    sleep 3
}

# Start RTSP publishing (full branch)
start_rtsp() {
    local branch=$1
    local stream_name=$2
    local bitrate=${3:-4000000}

    log_info "Starting RTSP publish: $branch -> $RTSP_BASE/$stream_name"
    result=$(curl -s -X POST "$API_URL/api/branches/$branch/rtsp/start" \
        -H "Content-Type: application/json" \
        -d "{\"location\": \"$RTSP_BASE/$stream_name\", \"bitrate\": $bitrate}")
    echo "  Response: $result"
    sleep 2
}

# Stop RTSP publishing
stop_rtsp() {
    local branch=$1

    log_info "Stopping RTSP publish: $branch"
    result=$(curl -s -X POST "$API_URL/api/branches/$branch/rtsp/stop")
    echo "  Response: $result"
    sleep 2
}

# Check RTSP status
check_rtsp_status() {
    log_info "Branch RTSP Status:"
    curl -s "$API_URL/api/rtsp/status" | python3 -m json.tool 2>/dev/null || \
        curl -s "$API_URL/api/rtsp/status"
}

# Start per-camera RTSP publishing
start_camera_rtsp() {
    local cam_id=$1
    local branch=$2
    local stream_name=$3
    local bitrate=${4:-4000000}

    log_info "Starting per-camera RTSP: $cam_id/$branch -> $RTSP_BASE/$stream_name"
    result=$(curl -s -X POST "$API_URL/api/cameras/$cam_id/branches/$branch/rtsp/start" \
        -H "Content-Type: application/json" \
        -d "{\"location\": \"$RTSP_BASE/$stream_name\", \"bitrate\": $bitrate}")
    echo "  Response: $result"
    sleep 3
}

# Stop per-camera RTSP publishing
stop_camera_rtsp() {
    local cam_id=$1
    local branch=$2

    log_info "Stopping per-camera RTSP: $cam_id/$branch"
    result=$(curl -s -X POST "$API_URL/api/cameras/$cam_id/branches/$branch/rtsp/stop")
    echo "  Response: $result"
    sleep 2
}

# Check per-camera RTSP status
check_camera_rtsp_status() {
    log_info "Per-Camera RTSP Status:"
    curl -s "$API_URL/api/cameras/rtsp/status" | python3 -m json.tool 2>/dev/null || \
        curl -s "$API_URL/api/cameras/rtsp/status"
}

# Verify stream with ffprobe
verify_stream() {
    local stream_url=$1
    local timeout=${2:-8}

    log_info "Verifying stream: $stream_url"
    if timeout $timeout ffprobe -v quiet -print_format json -show_streams "$stream_url" 2>/dev/null | grep -q "codec_name"; then
        log_info "Stream verified OK"
        return 0
    else
        log_warn "Stream not accessible or timeout"
        return 1
    fi
}

# Show logs
show_logs() {
    local lines=${1:-20}
    log_step "Pipeline Logs (last $lines lines)"
    docker exec "$DOCKER_CONTAINER" tail -$lines /tmp/publish_test.log 2>/dev/null | \
        grep -E "RTSP|rtsp|publish|FPS|ERROR|error|DemuxRTSP|Started|Stopped" || \
        docker exec "$DOCKER_CONTAINER" tail -$lines /tmp/publish_test.log 2>/dev/null
}

# Cleanup
cleanup() {
    log_step "Cleanup"
    curl -s -X POST "$API_URL/api/pipeline/kill" > /dev/null 2>&1
    curl -s -X POST "$API_URL/api/pipeline/stop" > /dev/null 2>&1
    log_info "Pipeline stopped"
}

# Full test with demux - per-camera annotated streams
# Each camera gets its own OSD (via demux RTSP chain)
test_demux() {
    echo "============================================"
    echo "  Per-Camera RTSP Publishing Test (Demux)"
    echo "  Each camera gets individual OSD annotations"
    echo "  Container: $DOCKER_CONTAINER"
    echo "  API: $API_URL"
    echo "  MediaMTX: $RTSP_BASE"
    echo "============================================"

    check_container

    # Start pipeline with demux config
    start_pipeline_demux

    sleep 3
    log_step "Step 1: Add Cameras to detection + recognition branches"
    add_camera "cam1" "rtsp://$MEDIAMTX_HOST:$MEDIAMTX_PORT/testface1" '["recognition", "detection"]'
    add_camera "cam2" "rtsp://$MEDIAMTX_HOST:$MEDIAMTX_PORT/testface2" '["recognition", "detection"]'

    log_step "Step 2: Check Cameras"
    curl -s "$API_URL/api/cameras" | python3 -m json.tool

    log_step "Step 3: Start per-camera RTSP - cam1/detection (with OSD)"
    start_camera_rtsp "cam1" "detection" "cam1_detection"
    check_camera_rtsp_status

    log_step "Step 4: Start per-camera RTSP - cam2/recognition (with OSD)"
    start_camera_rtsp "cam2" "recognition" "cam2_recognition"
    check_camera_rtsp_status

    log_step "Step 5: Verify Streams"
    verify_stream "$RTSP_BASE/cam1_detection" 10
    verify_stream "$RTSP_BASE/cam2_recognition" 10

    log_step "Step 6: Let streams run for 15s"
    sleep 15
    check_camera_rtsp_status

    show_logs 30

    log_step "Test Complete"
    echo ""
    echo "Summary:"
    echo "  - Cameras added: cam1, cam2 (both to detection + recognition)"
    echo "  - cam1/detection -> $RTSP_BASE/cam1_detection (with per-camera OSD)"
    echo "  - cam2/recognition -> $RTSP_BASE/cam2_recognition (with per-camera OSD)"
    echo ""
    echo "Architecture:"
    echo "  mux -> PGIE -> Tracker -> SGIE -> nvstreamdemux -> [per-cam: OSD -> enc -> RTSP]"
    echo ""
    echo "View streams:"
    echo "  ffplay $RTSP_BASE/cam1_detection"
    echo "  ffplay $RTSP_BASE/cam2_recognition"
    echo ""

    # Ask to cleanup
    read -p "Stop pipeline? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi
}

# Main test sequence (legacy - branch RTSP)
main() {
    echo "============================================"
    echo "  RTSP Publishing Test (Branch Mode)"
    echo "  Container: $DOCKER_CONTAINER"
    echo "  API: $API_URL"
    echo "  MediaMTX: $RTSP_BASE"
    echo "============================================"

    check_container

    # Check if pipeline already running
    if check_health; then
        log_info "Pipeline already running"
    else
        start_pipeline
    fi

    sleep 5
    log_step "Step 1: Add Camera"
    add_camera "cam1" "rtsp://$MEDIAMTX_HOST:$MEDIAMTX_PORT/testface" '["recognition", "detection"]'

    log_step "Step 2: Start RTSP Publishing - Recognition Branch"
    start_rtsp "recognition" "output_recognition" 4000000
    check_rtsp_status

    log_step "Step 3: Verify Recognition Stream"
    verify_stream "$RTSP_BASE/output_recognition" 5

    log_step "Step 4: Start RTSP Publishing - Detection Branch"
    start_rtsp "detection" "output_detection" 4000000
    check_rtsp_status

    log_step "Step 5: Verify Detection Stream"
    verify_stream "$RTSP_BASE/output_detection" 5

    log_step "Step 6: Let streams run for 10s"
    sleep 10
    check_rtsp_status

    show_logs 30

    log_step "Test Complete"
    echo ""
    echo "Summary:"
    echo "  - Camera added: OK"
    echo "  - RTSP start: OK"
    echo ""

    # Ask to cleanup
    read -p "Stop pipeline? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi
}

# Show help
show_help() {
    echo "Usage: $0 [command] [args...]"
    echo ""
    echo "Commands:"
    echo "  (no args)        Run legacy branch RTSP test"
    echo "  demux            Run per-camera demux RTSP test (annotated streams)"
    echo "  start            Start pipeline with normal config"
    echo "  start-demux      Start pipeline with demux config"
    echo "  publish          Start branch RTSP: publish <branch> [stream_name]"
    echo "  stop             Stop branch RTSP: stop [branch]"
    echo "  status           Show branch RTSP status"
    echo "  camera-publish   Start per-camera RTSP: camera-publish <cam_id> <branch> [stream_name]"
    echo "  camera-stop      Stop per-camera RTSP: camera-stop <cam_id> <branch>"
    echo "  camera-status    Show per-camera RTSP status"
    echo "  logs             Show pipeline logs: logs [lines]"
    echo "  cleanup          Stop pipeline"
    echo "  help             Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 demux                              # Full demux test"
    echo "  $0 camera-publish cam1 detection out  # Publish cam1/detection -> rtsp://.../out"
    echo "  $0 camera-status                      # Check per-camera status"
}

# Handle args
case "${1:-}" in
    demux)
        test_demux
        ;;
    start)
        check_container
        start_pipeline
        add_camera "cam1" "rtsp://$MEDIAMTX_HOST:$MEDIAMTX_PORT/testface" '["recognition", "detection"]'
        ;;
    start-demux)
        check_container
        start_pipeline_demux
        ;;
    publish)
        branch="${2:-recognition}"
        stream="${3:-output_$branch}"
        start_rtsp "$branch" "$stream"
        check_rtsp_status
        ;;
    stop)
        branch="${2:-}"
        if [ -n "$branch" ]; then
            stop_rtsp "$branch"
        else
            stop_rtsp "recognition"
            stop_rtsp "detection"
        fi
        check_rtsp_status
        ;;
    status)
        check_rtsp_status
        ;;
    camera-publish)
        # Per-camera RTSP: ./publish_mtx.sh camera-publish cam1 recognition cam1_recog
        cam_id="${2:-cam1}"
        branch="${3:-recognition}"
        stream="${4:-${cam_id}_${branch}}"
        start_camera_rtsp "$cam_id" "$branch" "$stream"
        check_camera_rtsp_status
        ;;
    camera-stop)
        # Stop per-camera RTSP: ./publish_mtx.sh camera-stop cam1 recognition
        cam_id="${2:-cam1}"
        branch="${3:-recognition}"
        stop_camera_rtsp "$cam_id" "$branch"
        check_camera_rtsp_status
        ;;
    camera-status)
        check_camera_rtsp_status
        ;;
    logs)
        show_logs "${2:-50}"
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        main
        ;;
esac
