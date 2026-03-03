#!/bin/bash
# ==========================================
# Setup NVIDIA Container Toolkit for WSL2
# Run this script on the WSL2 HOST (not inside Docker)
# Tested on Ubuntu 24.04 WSL2 + RTX 5070
# ==========================================

set -e

echo "=== Installing NVIDIA Container Toolkit for WSL2 ==="

# Distribution
distribution=ubuntu24.04

# Step 1: Add NVIDIA GPG key
echo "[1/4] Adding NVIDIA GPG key..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Step 2: Add NVIDIA container toolkit repository
echo "[2/4] Adding NVIDIA container toolkit repository..."
curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Step 3: Install nvidia-container-toolkit
echo "[3/4] Installing nvidia-container-toolkit..."
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Step 4: Configure Docker runtime and restart
echo "[4/4] Configuring Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo ""
echo "=== Setup Complete ==="
echo "Verify with: docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi"
echo ""
echo "Then build & start VMSx:"
echo "  cd scripts && docker compose up -d --build"
