#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${CUTLASS_ROOT:?Set CUTLASS_ROOT to your existing CUTLASS 4.4.2 tree}"
project=$(pwd -P)
cutlass=$(cd -- "$CUTLASS_ROOT" && pwd -P)
[[ -f "$cutlass/include/cutlass/cutlass.h" ]] || { echo 'CUTLASS headers are missing' >&2; exit 1; }
nvcc=$(realpath -- "$(command -v -- "${NVCC:-nvcc}")")
cxx=$(realpath -- "$(command -v -- "${CXX:-c++}")")
runtime_lib="$(dirname -- "$nvcc")/../lib64"
[[ -f "$runtime_lib/libcudart_static.a" ]] || { echo 'CUDA static runtime is missing' >&2; exit 1; }
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
# CUDA device assertions can keep header names even when host prefix mapping is enabled.
# Compile from relative copies so callers' source and dependency paths stay out of the binaries.
mkdir -p "$stage/cutlass/tools/util"
cp -R -- src "$stage/src"
cp -R -- "$cutlass/include" "$stage/cutlass/include"
cp -R -- "$cutlass/tools/util/include" "$stage/cutlass/tools/util/include"
cd -- "$stage"
flags=(-std=c++17 -O3 --cudart=static --cudadevrt=none --keep --keep-dir .
       -gencode arch=compute_120a,code=sm_120a -Isrc
       -Xcompiler "-ffile-prefix-map=$stage=.")
# Each CUDA executable is a single translation unit with its own device image.
# Link its host object directly to avoid a second, empty device-link image that
# records the build machine's CUDA library search path in a compiler note.
compile_cuda() {
    local input=$1 output=$2
    shift 2
    "$nvcc" "${flags[@]}" --frandom-seed="$output" \
        "-Xcudafe=--orig_src_path_name=fp8miner/$input" "-Xcicc=--orig_src_path_name=fp8miner/$input" \
        "$@" -c "$input" -o "$output.o"
    "$cxx" "$output.o" "$runtime_lib/libcudart_static.a" -lcrypto -ldl -lrt -pthread -o "$output"
}
compile_cuda src/scan.cu pearl-fp8-scan --expt-relaxed-constexpr -DPEARL_WITH_CUTLASS \
    -Icutlass/include -Icutlass/tools/util/include
compile_cuda src/gpu_proof.cu pearl-fp8-open
"$cxx" -std=c++17 -O3 -fPIC -shared -fopenmp -ffp-contract=off \
    "-ffile-prefix-map=$stage=." src/fast_prepare.cpp -o fast_prepare.native.so
strip --strip-all --remove-section=.comment -- pearl-fp8-scan pearl-fp8-open fast_prepare.native.so
mkdir -p "$project/build"
install -m 755 -- pearl-fp8-scan pearl-fp8-open fast_prepare.native.so "$project/build/"
