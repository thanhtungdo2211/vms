#!/usr/bin/env bash
# External camera sync worker (no changes in base pipeline required).
# Polls DB every INTERVAL_SEC and reconciles cameras via REST API.

set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8085}"
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:admin123@192.168.6.229:5432/iotvms_new}"
DB_QUERY="${DB_QUERY:-SELECT id, url FROM stream}"
BRANCH="${BRANCH:-recognition}"
INTERVAL_SEC="${INTERVAL_SEC:-10}"
STARTUP_DELAY_SEC="${STARTUP_DELAY_SEC:-10}"
PENDING_TTL_SEC="${PENDING_TTL_SEC:-45}"
MAX_OPS_PER_CYCLE="${MAX_OPS_PER_CYCLE:-3}"
CURL_TIMEOUT_SEC="${CURL_TIMEOUT_SEC:-5}"

LOG_PREFIX="[camera-db-sync-ext]"

declare -A PENDING_UNTIL=()

log() {
  printf '%s %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$LOG_PREFIX" "$*"
}

fetch_db_json() {
  DATABASE_URL="$DATABASE_URL" DB_QUERY="$DB_QUERY" python3 - <<'PY'
import json
import os
from sqlalchemy import create_engine, text

url = os.environ["DATABASE_URL"]
query = os.environ["DB_QUERY"]
engine = create_engine(url, pool_pre_ping=True)

rows = []
with engine.connect() as conn:
    rows = conn.execute(text(query)).fetchall()

data = {}
for row in rows:
    cam_id = str(row[0]).strip() if row[0] is not None else ""
    cam_url = str(row[1]).strip() if row[1] is not None else ""
    if cam_id and cam_url:
        data[cam_id] = cam_url

print(json.dumps(data, ensure_ascii=True))
PY
}

fetch_api_json() {
  local response
  response="$(curl -sS --max-time "$CURL_TIMEOUT_SEC" "$API_BASE_URL/api/cameras" || true)"
  if [[ -z "$response" ]]; then
    echo "{}"
    return 0
  fi

  echo "$response" | jq -c '.cameras // {} | with_entries(.value = (.value.uri // ""))' 2>/dev/null || echo "{}"
}

api_remove_camera() {
  local camera_id="$1"
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time "$CURL_TIMEOUT_SEC" \
    -X DELETE "$API_BASE_URL/api/cameras/$camera_id" || echo "000")"

  if [[ "$code" == "200" || "$code" == "202" || "$code" == "404" ]]; then
    log "remove accepted camera_id=$camera_id http=$code"
    return 0
  fi
  log "remove failed camera_id=$camera_id http=$code"
  return 1
}

api_add_camera() {
  local camera_id="$1"
  local camera_url="$2"
  local payload
  payload="$(jq -nc \
    --arg cid "$camera_id" \
    --arg uri "$camera_url" \
    --arg branch "$BRANCH" \
    '{camera_id:$cid, uri:$uri, branch:$branch}')"

  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time "$CURL_TIMEOUT_SEC" \
    -X POST "$API_BASE_URL/api/cameras" \
    -H "Content-Type: application/json" \
    -d "$payload" || echo "000")"

  if [[ "$code" == "200" || "$code" == "201" || "$code" == "202" ]]; then
    log "add accepted camera_id=$camera_id http=$code"
    return 0
  fi
  log "add failed camera_id=$camera_id http=$code"
  return 1
}

is_pending() {
  local camera_id="$1"
  local now="$2"
  local until="${PENDING_UNTIL[$camera_id]:-0}"
  [[ "$until" -gt "$now" ]]
}

mark_pending() {
  local camera_id="$1"
  local now="$2"
  PENDING_UNTIL["$camera_id"]=$((now + PENDING_TTL_SEC))
}

clear_pending_if_converged() {
  local db_json="$1"
  local api_json="$2"
  local now="$3"

  local camera_id desired actual until
  for camera_id in "${!PENDING_UNTIL[@]}"; do
    until="${PENDING_UNTIL[$camera_id]}"
    desired="$(jq -r --arg id "$camera_id" '.[$id] // empty' <<<"$db_json")"
    actual="$(jq -r --arg id "$camera_id" '.[$id] // empty' <<<"$api_json")"

    if { [[ -z "$desired" ]] && [[ -z "$actual" ]]; } || \
       { [[ -n "$desired" ]] && [[ "$desired" == "$actual" ]]; } || \
       (( until <= now )); then
      unset "PENDING_UNTIL[$camera_id]"
    fi
  done
}

build_actions() {
  local db_json="$1"
  local api_json="$2"

  # Output format:
  # remove|camera_id
  # restart|camera_id|new_url
  # add|camera_id|url
  jq -nr \
    --argjson db "$db_json" \
    --argjson api "$api_json" '
      # 1) remove in API but not in DB
      ($api | keys[] | select($db[.] == null) | "remove|\(.)"),
      # 2) restart when URL changed
      ($db | keys[] | select($api[.] != null and $api[.] != $db[.]) | "restart|\(.)|\($db[.])"),
      # 3) add in DB but not in API
      ($db | keys[] | select($api[.] == null) | "add|\(.)|\($db[.])")
    '
}

main_loop() {
  log "starting worker api=$API_BASE_URL branch=$BRANCH interval=${INTERVAL_SEC}s startup_delay=${STARTUP_DELAY_SEC}s"
  sleep "$STARTUP_DELAY_SEC"

  while true; do
    local db_json api_json now op_count action kind camera_id camera_url
    db_json="$(fetch_db_json || echo '{}')"
    api_json="$(fetch_api_json || echo '{}')"
    now="$(date +%s)"
    op_count=0

    clear_pending_if_converged "$db_json" "$api_json" "$now"

    while IFS= read -r action; do
      [[ -z "$action" ]] && continue
      IFS='|' read -r kind camera_id camera_url <<<"$action"

      if is_pending "$camera_id" "$now"; then
        continue
      fi

      case "$kind" in
        remove)
          if api_remove_camera "$camera_id"; then
            mark_pending "$camera_id" "$now"
            ((op_count += 1))
          fi
          ;;
        restart)
          if api_remove_camera "$camera_id" && api_add_camera "$camera_id" "$camera_url"; then
            mark_pending "$camera_id" "$now"
            log "restart accepted camera_id=$camera_id"
            ((op_count += 1))
          fi
          ;;
        add)
          if api_add_camera "$camera_id" "$camera_url"; then
            mark_pending "$camera_id" "$now"
            ((op_count += 1))
          fi
          ;;
      esac

      if (( op_count >= MAX_OPS_PER_CYCLE )); then
        break
      fi
    done < <(build_actions "$db_json" "$api_json")

    sleep "$INTERVAL_SEC"
  done
}

main_loop

