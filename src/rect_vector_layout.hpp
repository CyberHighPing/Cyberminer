#pragma once

#ifdef __CUDACC__
#define PEARL_RECT_LAYOUT __host__ __device__ __forceinline__
#else
#define PEARL_RECT_LAYOUT inline
#endif

namespace pearl_rect_vector {

// Entries are sixteen-byte vectors. Adjacent eight-lane shared-memory
// transaction groups touch every bank exactly once for both loads and stores.
PEARL_RECT_LAYOUT constexpr unsigned shared_vector(unsigned row, unsigned vector_in_row) {
    return row * 64u + (vector_in_row ^ (row & 7u));
}

}  // namespace pearl_rect_vector
#undef PEARL_RECT_LAYOUT
