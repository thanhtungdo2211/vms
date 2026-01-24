#!/bin/bash
# monitor_dpu.sh - Monitor RAM, CPU, GPU (NVIDIA dGPU like T4, A100, etc.) for Ubuntu/x86_64

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping monitor..."
    exit 0
}
trap cleanup EXIT INT TERM QUIT

echo "=== NVIDIA GPU HARDWARE MONITOR ==="
echo "Hostname: $(hostname)"
echo "Started: $(date)"
echo ""

# Display initial info
TOTAL_RAM=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
CPU_CORES=$(nproc)

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "System Info:"
echo "  Total RAM   : ${TOTAL_RAM} GB"
echo "  CPU Cores   : ${CPU_CORES}"
echo "  GPU         : ${GPU_NAME}"
echo "  GPU Memory  : ${GPU_MEMORY} MB"
echo ""

# Column headers
printf "%-8s | %-15s | %-15s | %-12s | %-15s | %-10s\n" \
    "Time" "System RAM" "CPU Usage" "GPU Usage" "GPU Memory" "GPU Temp"
printf "%-8s-+-%-15s-+-%-15s-+-%-12s-+-%-15s-+-%-10s\n" \
    "--------" "---------------" "---------------" "------------" "---------------" "----------"

# Store previous CPU stats for accurate calculation
prev_idle=0
prev_total=0

while true; do
    # Timestamp
    TIME=$(date '+%H:%M:%S')

    # System RAM
    RAM_TOTAL_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    RAM_AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    RAM_USED_KB=$((RAM_TOTAL_KB - RAM_AVAIL_KB))
    RAM_PERCENT=$((RAM_USED_KB * 100 / RAM_TOTAL_KB))
    RAM_USED_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_USED_KB/1024/1024}")
    RAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_TOTAL_KB/1024/1024}")

    # CPU Usage (accurate calculation based on delta)
    read cpu user nice system idle iowait irq softirq steal guest guest_nice < <(grep '^cpu ' /proc/stat)
    curr_idle=$((idle + iowait))
    curr_total=$((user + nice + system + idle + iowait + irq + softirq + steal))

    if [ $prev_total -ne 0 ]; then
        total_delta=$((curr_total - prev_total))
        idle_delta=$((curr_idle - prev_idle))
        if [ $total_delta -gt 0 ]; then
            CPU_USAGE=$(awk "BEGIN {printf \"%.0f\", 100 * (1 - $idle_delta / $total_delta)}")
        else
            CPU_USAGE=0
        fi
    else
        CPU_USAGE=0
    fi

    prev_idle=$curr_idle
    prev_total=$curr_total
    CPU_USAGE_INT=$(printf "%.0f" "$CPU_USAGE" 2>/dev/null || echo "0")

    # GPU Utilization (%)
    GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    GPU_USAGE_INT=$(printf "%.0f" "$GPU_USAGE" 2>/dev/null || echo "0")

    # GPU Memory
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_MEM_PERCENT=$((GPU_MEM_USED * 100 / GPU_MEM_TOTAL))

    # GPU Temperature
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    GPU_TEMP_INT=$(printf "%.0f" "$GPU_TEMP" 2>/dev/null || echo "0")

    # Format output strings
    RAM_STR=$(printf "%4.1f/%4.1fGB %2d%%" "$RAM_USED_GB" "$RAM_TOTAL_GB" "$RAM_PERCENT")
    CPU_STR=$(printf "%3d%%" "$CPU_USAGE_INT")
    GPU_STR=$(printf "%3d%%" "$GPU_USAGE_INT")
    GPU_MEM_STR=$(printf "%5d/%5dMB %2d%%" "$GPU_MEM_USED" "$GPU_MEM_TOTAL" "$GPU_MEM_PERCENT")
    GPU_TEMP_STR=$(printf "%2d°C" "$GPU_TEMP_INT")

    # Warning indicators
    RAM_WARN=" "
    [ $RAM_PERCENT -gt 90 ] && RAM_WARN="⚠️"
    [ $RAM_PERCENT -gt 80 ] && [ $RAM_PERCENT -le 90 ] && RAM_WARN="⚡"

    CPU_WARN=" "
    [ $CPU_USAGE_INT -gt 90 ] && CPU_WARN="⚠️"
    [ $CPU_USAGE_INT -gt 80 ] && [ $CPU_USAGE_INT -le 90 ] && CPU_WARN="⚡"

    GPU_WARN=" "
    [ $GPU_USAGE_INT -gt 90 ] && GPU_WARN="⚠️"
    [ $GPU_USAGE_INT -gt 80 ] && [ $GPU_USAGE_INT -le 90 ] && GPU_WARN="⚡"

    GPU_MEM_WARN=" "
    [ $GPU_MEM_PERCENT -gt 90 ] && GPU_MEM_WARN="⚠️"
    [ $GPU_MEM_PERCENT -gt 80 ] && [ $GPU_MEM_PERCENT -le 90 ] && GPU_MEM_WARN="⚡"

    GPU_TEMP_WARN=" "
    [ $GPU_TEMP_INT -gt 85 ] && GPU_TEMP_WARN="⚠️"
    [ $GPU_TEMP_INT -gt 75 ] && [ $GPU_TEMP_INT -le 85 ] && GPU_TEMP_WARN="⚡"

    printf "%s | %s %s | %s %s | %s %s | %s %s | %s %s\n" \
        "$TIME" \
        "$RAM_STR" "$RAM_WARN" \
        "$CPU_STR" "$CPU_WARN" \
        "$GPU_STR" "$GPU_WARN" \
        "$GPU_MEM_STR" "$GPU_MEM_WARN" \
        "$GPU_TEMP_STR" "$GPU_TEMP_WARN"

    sleep 2
done
