#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pearl_square {
// At 16x16 the fixed sixteen extractor lanes are exactly the sixteen rows.
// A CTA loads 8 neighbouring tickets coalesced; the padded shared pitch
// assigns distinct banks to the 16 row lanes of each adjacent ticket.
constexpr unsigned rows=16, width=16, group=8, threads=rows*group, slab_width=width*group;
__global__ __launch_bounds__(threads) void fold(const std::uint32_t* __restrict__ input,unsigned n,
                                                 std::uint32_t* __restrict__ out,unsigned fold_cols) {
    __shared__ std::uint32_t block[rows][slab_width+1];
    const unsigned tid=threadIdx.x;
    const unsigned first_col=blockIdx.x*slab_width;
    const unsigned first_row=blockIdx.y*rows;
#pragma unroll
    for(unsigned i=0;i<4;++i) {
        const unsigned vector=tid+i*threads,r=vector/(slab_width/4),c=(vector%(slab_width/4))*4;
        const auto v=*reinterpret_cast<const uint4*>(input+std::size_t(first_row+r)*n+first_col+c);
        block[r][c]=v.x;block[r][c+1]=v.y;block[r][c+2]=v.z;block[r][c+3]=v.w;
    }
    __syncthreads();
    const unsigned ticket=tid/rows,row=tid%rows;
    std::uint32_t x=0;
#pragma unroll
    for(unsigned col=0;col<width;++col) {
        x=x*0x9e3779b1u+block[row][ticket*width+col];
        x=__funnelshift_l(x,x,13);
    }
    out[(std::size_t(blockIdx.y)*fold_cols+blockIdx.x*group+ticket)*rows+row]=x;
}
inline cudaError_t launch(const std::uint32_t* matrix,int m,int n,std::uint32_t* output,int fold_cols,cudaStream_t stream) {
    if(!matrix || !output || m<=0 || m%int(rows) || n<=0 || n%int(slab_width) || fold_cols<n/int(width) ||
       m/int(rows)>65535 || (reinterpret_cast<std::uintptr_t>(matrix)&15u) || (reinterpret_cast<std::uintptr_t>(output)&3u))
        return cudaErrorInvalidValue;
    fold<<<dim3(unsigned(n)/slab_width,unsigned(m)/rows),threads,0,stream>>>(matrix,unsigned(n),output,unsigned(fold_cols));
    return cudaGetLastError();
}
}  // namespace pearl_square
