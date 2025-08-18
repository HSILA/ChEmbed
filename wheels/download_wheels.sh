#!/bin/bash
set -euo pipefail

WHEELS_DIR="wheels"

echo "Creating wheel directory structure..."
mkdir -p "${WHEELS_DIR}/deepspeed"
mkdir -p "${WHEELS_DIR}/s3fs" 
mkdir -p "${WHEELS_DIR}/pymongo"
mkdir -p "${WHEELS_DIR}/flash-attention"

echo "Downloading deepspeed and dependencies..."
pip download -d "${WHEELS_DIR}/deepspeed" \
    deepspeed==0.14.2 \
    filelock==3.16.1 \
    hjson==3.1.0 \
    jinja2==3.1.4 \
    mpmath==1.3.0 \
    networkx==3.4.2 \
    ninja==1.11.1.1 \
    py-cpuinfo==9.0.0 \
    pynvml==11.5.3 \
    tqdm==4.67.0 \
    typing-extensions==4.12.2

echo "Downloading s3fs and dependencies..."
pip download -d "${WHEELS_DIR}/s3fs" \
    aiobotocore==2.7.0 \
    botocore==1.31.64 \
    fsspec==2023.10.0 \
    s3fs==2023.10.0 \
    urllib3==2.0.7

echo "Downloading pymongo..."
pip download -d "${WHEELS_DIR}/pymongo" \
    pymongo==3.13.0

echo "Building main flash-attention..."
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git --no-build-isolation -w "${WHEELS_DIR}/flash-attention/"

echo "Building flash-attention components..."
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary --no-build-isolation -w "${WHEELS_DIR}/flash-attention/"
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/layer_norm --no-build-isolation -w "${WHEELS_DIR}/flash-attention/"
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/fused_dense_lib --no-build-isolation -w "${WHEELS_DIR}/flash-attention"
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/xentropy --no-build-isolation -w "${WHEELS_DIR}/flash-attention"

echo "Download and build complete!"
