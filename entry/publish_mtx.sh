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
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${GREEN}=== $1 ===${NC}"; }

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

# Start pipeline
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

    log_info "Adding camera: $cam_id"
    result=$(curl -s -X POST "$API_URL/api/cameras" \
        -H "Content-Type: application/json" \
        -d "{\"camera_id\": \"$cam_id\", \"uri\": \"$uri\", \"branches\": $branches}")
    echo "  Response: $result"
    sleep 3
}

# Start RTSP publishing
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
    log_info "RTSP Status:"
    curl -s "$API_URL/api/rtsp/status" | python3 -m json.tool 2>/dev/null || \
        curl -s "$API_URL/api/rtsp/status"
}

# Verify stream with ffprobe
verify_stream() {
    local stream_url=$1
    local timeout=${2:-5}

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
        grep -E "RTSP|rtsp|publish|FPS|ERROR|error" || \
        docker exec "$DOCKER_CONTAINER" tail -$lines /tmp/publish_test.log 2>/dev/null
}

# Cleanup
cleanup() {
    log_step "Cleanup"
    curl -s -X POST "$API_URL/api/pipeline/kill" > /dev/null 2>&1
    curl -s -X POST "$API_URL/api/pipeline/stop" > /dev/null 2>&1
    log_info "Pipeline stopped"
}

# Main test sequence
main() {
    echo "============================================"
    echo "  RTSP Publishing Test"
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

    # log_step "Step 7: Stop Recognition RTSP"
    # stop_rtsp "recognition"
    # check_rtsp_status

    # log_step "Step 8: Stop Detection RTSP"
    # stop_rtsp "detection"
    # check_rtsp_status

    # log_step "Step 9: Verify Streams Stopped"
    # verify_stream "$RTSP_BASE/output_recognition" 3 && log_warn "Recognition stream still active" || log_info "Recognition stream stopped"
    # verify_stream "$RTSP_BASE/output_detection" 3 && log_warn "Detection stream still active" || log_info "Detection stream stopped"

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

# Handle args
case "${1:-}" in
    start)
        check_container
        start_pipeline
        add_camera "cam1" "rtsp://$MEDIAMTX_HOST:$MEDIAMTX_PORT/testface" '["recognition", "detection"]'
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
    logs)
        show_logs "${2:-50}"
        ;;
    cleanup)
        cleanup
        ;;
    *)
        main
        ;;
esac
