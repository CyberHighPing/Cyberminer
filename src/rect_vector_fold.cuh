#pragma once
#include "rect_vector_layout.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace pearl_rect_vector {

__device__ __forceinline__ void store_shared(uint4* ptr, uint4 v) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ uint4 load_shared(const uint4* ptr) {
    uint4 v;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ unsigned update(unsigned x, unsigned value) {
    x = x * 0x9e3779b1u + value;
    return __funnelshift_l(x, x, 13);
}

__global__ __launch_bounds__(128) void fold(const std::uint32_t* __restrict__ matrix, unsigned n,
                                            std::uint32_t* __restrict__ out, unsigned fold_cols) {
    __shared__ uint4 data[16u * 64u];
    const unsigned tid = threadIdx.x;
    const unsigned vector = tid & 63u;
    const unsigned first_row = blockIdx.y * 16u;
    const unsigned first_col = blockIdx.x * 256u;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned row = (tid >> 6) + 2u * i;
        const auto* src = matrix + std::size_t(first_row + row) * n + first_col + 4u * vector;
        store_shared(data + shared_vector(row, vector), *reinterpret_cast<const uint4*>(src));
    }
    __syncthreads();
    const unsigned row = tid & 15u, ticket = tid >> 4;
    unsigned x = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < 8; ++chunk) {
        const uint4 v = load_shared(data + shared_vector(row, 8u * ticket + chunk));
        x = update(x, v.x);
        x = update(x, v.y);
        x = update(x, v.z);
        x = update(x, v.w);
    }
    out[(std::size_t(blockIdx.y) * fold_cols + blockIdx.x * 8u + ticket) * 16u + row] = x;
}

inline bool enabled() {
    static const bool on=[] {
        const char* value=std::getenv("PEARL_RECT_FOLD");
        return value && std::strcmp(value,"vector")==0;
    }();
    return on;
}

// Both invocation sites are reached only after the native column-stream geometry
// checks. Keep the kernel bounds explicit if it is reused on other panels.
inline cudaError_t launch(const std::uint32_t* matrix,int m,int n,std::uint32_t* out,
                          int fold_cols,cudaStream_t stream) {
    if (!matrix || !out || m<=0 || m%16 || n<=0 || n%256 || fold_cols<n/32 ||
        m/16>65535 || (reinterpret_cast<std::uintptr_t>(matrix)&15u) ||
        (reinterpret_cast<std::uintptr_t>(out)&3u)) return cudaErrorInvalidValue;
    fold<<<dim3(unsigned(n)/256u,unsigned(m)/16u),128,0,stream>>>(matrix,unsigned(n),out,unsigned(fold_cols));
    return cudaGetLastError();
}
}  // namespace pearl_rect_vector
