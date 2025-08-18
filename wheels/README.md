# Wheels Directory

Pre-built wheels for Compute Canada offline installation.

## Contents

See `wheels_inventory.txt` for complete list of 22 wheels organized by subdirectory:
- `deepspeed/` - DeepSpeed and dependencies (11 wheels)
- `s3fs/` - S3 filesystem support (5 wheels)  
- `pymongo/` - MongoDB driver (1 wheel)
- `flash-attention/` - FlashAttention components (5 wheels)

## Download All Wheels

Run the script to recreate this directory structure:

```bash
./wheels/download_wheels.sh
```

## FlashAttention

FlashAttention wheels should be built from the GitHub repository and saved for offline installation:

```bash
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git --no-build-isolation -w wheels/flash-attention/
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary --no-build-isolation -w wheels/flash-attention/
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/layer_norm --no-build-isolation -w wheels/flash-attention/
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/fused_dense_lib --no-build-isolation -w wheels/flash-attention/
MAX_JOBS=4 pip wheel git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/xentropy --no-build-isolation -w wheels/flash-attention/
```

