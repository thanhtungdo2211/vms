# ==========================================
# VMSx - DeepStream 8.0 VMS Core (WSL2 + RTX 5070)
# ==========================================
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# ----------------
# Environment
# ----------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GST_DEBUG=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH \
    PYTHONPATH=/opt/nvidia/deepstream/deepstream/lib:$PYTHONPATH

# Set working directory
WORKDIR /app

# Step 1: Install gstreamer1.0-libav
RUN apt-get update && apt-get install -y gstreamer1.0-libav

# Step 2: Install core libraries and gstreamer plugins (Ubuntu 24.04 / DS 8.0)
RUN apt-get install --reinstall -y \
    libssl3 \
    libssl-dev \
    libgles2-mesa-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libyaml-cpp-dev \
    libjsoncpp-dev \
    protobuf-compiler \
    libmosquitto1 \
    gcc \
    make \
    git \
    python3

# Step 3: Install gstreamer and python dependencies
RUN apt-get install -y \
    gstreamer1.0 \
    python3-pip \
    python3-gst-1.0 \
    gir1.2-gst-plugins-bad-1.0 \
    gstreamer1.0-nice \
    pkg-config \
    libcairo2-dev \
    gcc \
    python3-dev \
    libgirepository1.0-dev

# Step 4: Install opencv
RUN apt-get install -y libopencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache/gstreamer-1.0/

# Install Python packages (Python 3.12 in DS 8.0 / Ubuntu 24.04 requires --break-system-packages)
RUN pip3 install --no-cache-dir --break-system-packages \
    websockets \
    numpy \
    PyGObject \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl

# Install additional requirements
COPY scripts/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt && rm /tmp/requirements.txt
