#!/bin/bash
# monitor.sh - Monitor CMA, RAM, CPU, GPU, Temperature for Jetson

# Temp file for tegrastats output
TEGRA_STATS_FILE="/tmp/tegrastats_monitor_$$"

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping monitor..."
    pkill -P $$ tegrastats 2>/dev/null
    pkill tegrastats 2>/dev/null
    rm -f "$TEGRA_STATS_FILE"
    exit 0
}
trap cleanup EXIT INT TERM QUIT

echo "=== JETSON HARDWARE MONITOR ==="
echo "Device: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo 'Unknown')"
echo "Started: $(date)"
echo ""

# Display initial info
TOTAL_RAM=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
CMA_TOTAL=$(awk '/CmaTotal/ {printf "%.0f", $2/1024}' /proc/meminfo)
CPU_CORES=$(nproc)

echo "System Info:"
echo "  Total RAM   : ${TOTAL_RAM} GB"
echo "  CMA Total   : ${CMA_TOTAL} MB"
echo "  CPU Cores   : ${CPU_CORES}"
echo ""

# Start tegrastats in background (writes to file continuously)
tegrastats --interval 500 > "$TEGRA_STATS_FILE" 2>/dev/null &
TEGRA_PID=$!
sleep 1  # Wait for first output

# Column headers
printf "%-8s | %-15s | %-15s | %-12s | %-12s | %-10s\n" \
    "Time" "CMA Memory" "System RAM" "CPU Usage" "GPU Usage" "Temp"
printf "%-8s-+-%-15s-+-%-15s-+-%-12s-+-%-12s-+-%-10s\n" \
    "--------" "---------------" "---------------" "------------" "------------" "----------"

while true; do
    # Timestamp
    TIME=$(date '+%H:%M:%S')

    # CMA Memory
    CMA_TOTAL_KB=$(awk '/CmaTotal/ {print $2}' /proc/meminfo)
    CMA_FREE_KB=$(awk '/CmaFree/ {print $2}' /proc/meminfo)
    CMA_USED_KB=$((CMA_TOTAL_KB - CMA_FREE_KB))
    CMA_PERCENT=$((CMA_USED_KB * 100 / CMA_TOTAL_KB))
    CMA_USED_MB=$((CMA_USED_KB / 1024))
    CMA_TOTAL_MB=$((CMA_TOTAL_KB / 1024))

    # System RAM
    RAM_TOTAL_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    RAM_AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    RAM_USED_KB=$((RAM_TOTAL_KB - RAM_AVAIL_KB))
    RAM_PERCENT=$((RAM_USED_KB * 100 / RAM_TOTAL_KB))
    RAM_USED_MB=$((RAM_USED_KB / 1024))
    RAM_TOTAL_MB=$((RAM_TOTAL_KB / 1024))

    # CPU Usage (per-core average)
    CPU_USAGE=$(grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage}')
    CPU_USAGE_INT=$(printf "%.0f" "$CPU_USAGE" 2>/dev/null || echo "0")

    # GPU Usage - Parse from tegrastats file (last line)
    GPU_USAGE=0
    if [ -f "$TEGRA_STATS_FILE" ]; then
        LAST_TEGRA=$(tail -1 "$TEGRA_STATS_FILE" 2>/dev/null)
        GPU_USAGE=$(echo "$LAST_TEGRA" | grep -oP 'GR3D_FREQ\s+\K[0-9]+(?=%)' || echo "0")
    fi
    GPU_USAGE_INT=$(printf "%.0f" "$GPU_USAGE" 2>/dev/null || echo "0")

    # Temperature - Parse from tegrastats (cpu@XX.XC or tj@XX.XC)
    TEMP_CPU=0
    if [ -f "$TEGRA_STATS_FILE" ]; then
        LAST_TEGRA=$(tail -1 "$TEGRA_STATS_FILE" 2>/dev/null)
        # Extract temperature from "cpu@42.5C" or "tj@42.343C"
        TEMP_STR=$(echo "$LAST_TEGRA" | grep -oP 'cpu@\K[0-9]+(\.[0-9]+)?(?=C)' | head -1)
        if [ -n "$TEMP_STR" ]; then
            # Convert to integer (42.5 -> 42, not 420)
            TEMP_CPU=$(echo "$TEMP_STR" | awk '{printf "%.0f", $1}')
        fi
    fi

    # Fallback temperature from sysfs
    if [ "$TEMP_CPU" = "0" ] && [ -f /sys/devices/virtual/thermal/thermal_zone0/temp ]; then
        TEMP_CPU=$(cat /sys/devices/virtual/thermal/thermal_zone0/temp 2>/dev/null)
        TEMP_CPU=$((TEMP_CPU / 1000))
    fi

    # Format output strings
    CMA_STR=$(printf "%3d/%3dMB %2d%%" "$CMA_USED_MB" "$CMA_TOTAL_MB" "$CMA_PERCENT")
    RAM_STR=$(printf "%4d/%4dMB %2d%%" "$RAM_USED_MB" "$RAM_TOTAL_MB" "$RAM_PERCENT")
    CPU_STR=$(printf "%3d%%" "$CPU_USAGE_INT")
    GPU_STR=$(printf "%3d%%" "$GPU_USAGE_INT")
    TEMP_STR=$(printf "%2d°C" "$TEMP_CPU")

    # Warning indicators
    CMA_WARN=" "
    [ $CMA_PERCENT -gt 90 ] && CMA_WARN="⚠️"
    [ $CMA_PERCENT -gt 80 ] && [ $CMA_PERCENT -le 90 ] && CMA_WARN="⚡"

    RAM_WARN=" "
    [ $RAM_PERCENT -gt 90 ] && RAM_WARN="⚠️"
    [ $RAM_PERCENT -gt 80 ] && [ $RAM_PERCENT -le 90 ] && RAM_WARN="⚡"

    CPU_WARN=" "
    [ $CPU_USAGE_INT -gt 90 ] && CPU_WARN="⚠️"
    [ $CPU_USAGE_INT -gt 80 ] && [ $CPU_USAGE_INT -le 90 ] && CPU_WARN="⚡"

    GPU_WARN=" "
    [ $GPU_USAGE_INT -gt 90 ] && GPU_WARN="⚠️"
    [ $GPU_USAGE_INT -gt 80 ] && [ $GPU_USAGE_INT -le 90 ] && GPU_WARN="⚡"

    TEMP_WARN=" "
    [ $TEMP_CPU -gt 85 ] && TEMP_WARN="⚠️"
    [ $TEMP_CPU -gt 75 ] && [ $TEMP_CPU -le 85 ] && TEMP_WARN="⚡"

    printf "%s | %s %s | %s %s | %s %s | %s %s | %s %s\n" \
        "$TIME" \
        "$CMA_STR" "$CMA_WARN" \
        "$RAM_STR" "$RAM_WARN" \
        "$CPU_STR" "$CPU_WARN" \
        "$GPU_STR" "$GPU_WARN" \
        "$TEMP_STR" "$TEMP_WARN"

    sleep 2
done
