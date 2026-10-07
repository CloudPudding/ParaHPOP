#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$root/build}"
nvcc_path="${CUDACXX:-}"
if [[ -z "$nvcc_path" ]]; then
    # Prefer the actual toolkit. A shell wrapper on PATH (e.g.
    # /usr/local/bin/nvcc) can hide the toolkit root from CMake.
    if [[ -x /usr/local/cuda-12.5/bin/nvcc ]]; then
        nvcc_path=/usr/local/cuda-12.5/bin/nvcc
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        nvcc_path="$(readlink -f /usr/local/cuda/bin/nvcc)"
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc_path="$(readlink -f "$(command -v nvcc)")"
    else
        echo 'Set CUDACXX to the nvcc executable of CUDA 12.5 or a compatible toolkit.' >&2
        exit 1
    fi
fi
cmake -S "$root" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$nvcc_path" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75}"
cmake --build "$build_dir" --parallel "${BUILD_JOBS:-2}"
