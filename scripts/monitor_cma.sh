#!/bin/bash
# Monitor CMA memory usage in real-time

echo "=== CMA Memory Monitor (Ctrl+C to stop) ==="
echo "Timestamp            | CMA Total | CMA Free  | Free %"
echo "---------------------+-----------+-----------+--------"

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    cma_total=$(grep "CmaTotal:" /proc/meminfo | awk '{print $2}')
    cma_free=$(grep "CmaFree:" /proc/meminfo | awk '{print $2}')

    if [ -n "$cma_total" ] && [ -n "$cma_free" ]; then
        free_percent=$((cma_free * 100 / cma_total))
        printf "%s | %8d kB | %8d kB | %5d%%\n" \
            "$timestamp" "$cma_total" "$cma_free" "$free_percent"

        # Warn if CMA free < 10%
        if [ "$free_percent" -lt 10 ]; then
            echo "⚠️  WARNING: CMA memory critically low! ($free_percent%)"
        fi
    fi

    sleep 2
done
