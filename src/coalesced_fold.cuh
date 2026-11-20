#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Identity lane order only: one 16x32 ticket produces sixteen words, one
// serial fold per matrix row. The caller supplies a row-major uint32 panel
// and may offset out to a window inside a wider full-job fold buffer.
namespace pearl_fold {
namespace detail {

constexpr unsigned kHeight = 16;
constexpr unsigned kWidth = 32;
constexpr unsigned kTickets = 8;
constexpr unsigned kThreads = kHeight * kTickets;

template <bool Swizzled>
__device__ __forceinline__ unsigned shared_index(unsigned ticket, unsigned row,
                                                  unsigned col) {
    const unsigned lane = ticket * kHeight + row;
    if constexpr (Swizzled) {
        // For vector loaders, four successive tickets select four different
        // bank residues mod 4. For readers, one warp covers two whole tickets;
        // their sixteen rows select the even and odd banks respectively.
        return lane * kWidth + ((col + 2 * row + ticket) & 31u);
    }
    return lane * (kWidth + 1) + col;
}

template <bool Vectorized>
__device__ __forceinline__ uint4 load_four(const std::uint32_t* ptr) {
    if constexpr (Vectorized) {
        return *reinterpret_cast<const uint4*>(ptr);
    }
    return make_uint4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

template <bool Vectorized, bool Swizzled>
__global__ __launch_bounds__(kThreads) void fold16x32(
        const std::uint32_t* __restrict__ matrix, unsigned n,
        unsigned ticket_rows, unsigned ticket_cols,
        std::uint32_t* __restrict__ out, unsigned fold_cols) {
    constexpr unsigned stride = Swizzled ? kWidth : kWidth + 1;
    __shared__ std::uint32_t tile[kThreads * stride];

    const unsigned tid = threadIdx.x;
    const unsigned base_ticket = blockIdx.x * kTickets;
    const unsigned vector_in_row = tid & 63u;       // 64 uint4s span 8 tickets
    const unsigned load_ticket = vector_in_row >> 3;
    const unsigned load_col = (vector_in_row & 7u) * 4;
    const unsigned fold_ticket = tid >> 4;
    const unsigned fold_row = tid & 15u;

    for (unsigned tr = blockIdx.y; tr < ticket_rows; tr += gridDim.y) {
        if (base_ticket + load_ticket < ticket_cols) {
            // Keep fewer swizzled addresses live at once on SM120.
#pragma unroll (Swizzled ? 2 : 8)
            for (unsigned pair = 0; pair < kHeight / 2; ++pair) {
                const unsigned row = 2 * pair + (tid >> 6);
                const std::size_t global_row = std::size_t(tr) * kHeight + row;
                const std::size_t global_col = std::size_t(base_ticket) * kWidth + 4 * vector_in_row;
                const uint4 v = load_four<Vectorized>(matrix + global_row * n + global_col);
                tile[shared_index<Swizzled>(load_ticket, row, load_col)] = v.x;
                tile[shared_index<Swizzled>(load_ticket, row, load_col + 1)] = v.y;
                tile[shared_index<Swizzled>(load_ticket, row, load_col + 2)] = v.z;
                tile[shared_index<Swizzled>(load_ticket, row, load_col + 3)] = v.w;
            }
        }
        __syncthreads();

        if (base_ticket + fold_ticket < ticket_cols) {
            std::uint32_t x = 0;
#pragma unroll (Swizzled ? 8 : 32)
            for (unsigned col = 0; col < kWidth; ++col) {
                x = x * 0x9e3779b1u + tile[shared_index<Swizzled>(fold_ticket, fold_row, col)];
                x = __funnelshift_l(x, x, 13);
            }
            out[(std::size_t(tr) * fold_cols + base_ticket) * kHeight + tid] = x;
        }
        // Only very tall panels reuse this block's shared buffer. All threads,
        // including inactive tail-ticket threads, reach the same barriers.
        if (tr + gridDim.y >= ticket_rows) break;
        __syncthreads();
    }
}

template <bool Swizzled>
inline cudaError_t launch(const std::uint32_t* matrix, int m, int n, std::uint32_t* out,
                          int fold_cols, cudaStream_t stream) {
    if (m < 0 || n < 0 || fold_cols < 0) return cudaErrorInvalidValue;
    const unsigned ticket_rows = static_cast<unsigned>(m) / kHeight;
    const unsigned ticket_cols = static_cast<unsigned>(n) / kWidth;
    if (!ticket_rows || !ticket_cols) return cudaSuccess;
    if (!matrix || !out || static_cast<unsigned>(fold_cols) < ticket_cols)
        return cudaErrorInvalidValue;
    const auto base = reinterpret_cast<std::uintptr_t>(matrix);
    if ((base & 3u) || (reinterpret_cast<std::uintptr_t>(out) & 3u))
        return cudaErrorInvalidValue;
    const dim3 grid((ticket_cols + kTickets - 1) / kTickets,
                    ticket_rows < 65535u ? ticket_rows : 65535u);
    if (!(base & 15u) && !(n & 3)) {
        fold16x32<true, Swizzled><<<grid, kThreads, 0, stream>>>(
            matrix, n, ticket_rows, ticket_cols, out, fold_cols);
    } else {
        fold16x32<false, Swizzled><<<grid, kThreads, 0, stream>>>(
            matrix, n, ticket_rows, ticket_cols, out, fold_cols);
    }
    return cudaGetLastError();
}

}  // namespace detail

// fold_cols is the destination pitch measured in 16-word tickets.
// Both launches are asynchronous in stream. The caller owns fencing and errors
// after execution. An empty grid has no input/output storage requirement.
inline cudaError_t launch16x32(const std::uint32_t* matrix, int m, int n,
                              std::uint32_t* out, int fold_cols, cudaStream_t stream = nullptr) {
    return detail::launch<false>(matrix, m, n, out, fold_cols, stream);
}

inline cudaError_t launch16x32_swizzled(const std::uint32_t* matrix, int m, int n,
                                       std::uint32_t* out, int fold_cols, cudaStream_t stream = nullptr) {
    return detail::launch<true>(matrix, m, n, out, fold_cols, stream);
}

}  // namespace pearl_fold
