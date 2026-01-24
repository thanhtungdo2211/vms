#!/bin/bash
# Rebuild and restart the VMSx VMS Core containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Rebuilding and restarting VMSx containers..."

# Stop existing containers
docker compose down

docker compose build --no-cache

# Start containers
docker compose up -d

echo "Rebuild complete!"
echo "Use ./log.sh to view logs"
