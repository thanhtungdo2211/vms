#!/usr/bin/env bash

set -euo pipefail

SYNC_SCRIPT="/app/entry/camera_db_sync_external.sh"

if [[ ! -x "$SYNC_SCRIPT" ]]; then
  echo "Missing executable sync script: $SYNC_SCRIPT"
  exit 1
fi

cleanup() {
  if [[ -n "${SYNC_PID:-}" ]] && kill -0 "$SYNC_PID" 2>/dev/null; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Start external DB sync in background
"$SYNC_SCRIPT" &
SYNC_PID=$!
echo "[runner] external camera sync started pid=$SYNC_PID"

# Start pipeline in foreground (existing flow)
exec /app/entry/run_pipeline.sh

