#!/bin/bash
# Stop the VMSx VMS Core containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping VMSx containers..."

docker compose down

echo "Containers stopped successfully!"
