#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace pearl_prep {
struct Row {
    float alpha, beta;
    uint8_t strip, threshold;
    uint16_t reserved;
};
static_assert(sizeof(Row) == 12, "row diagnostic wire layout");

__device__ __forceinline__ float bf(unsigned code) { return __uint_as_float(code << 16); }
__device__ __forceinline__ float bf_round(float x) {
    const unsigned v = __float_as_uint(x);
    return bf((v + 0x7fffu + ((v >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint8_t e4m3(float x) {
    const unsigned bits = __float_as_uint(x), mag = bits & 0x7fffffffu;
    unsigned code;
    if (mag < 0x3c800000u)
        code = __float_as_uint(__fadd_rn(__uint_as_float(mag), 16384.f)) - 0x46800000u;
    else {
        const unsigned rounded = mag + 0x7ffffu + ((mag >> 20) & 1u);
        code = min(126u,(rounded >> 20) - 960u);
    }
    return uint8_t(code | ((bits >> 24) & 128u));
}

__device__ __forceinline__ Row row_coefficients(unsigned sum, unsigned peak, int k, unsigned* invalid) {
    // Compare the exact integer sum against adjacent BF16 squares. No order-
    // dependent floating sum participates in the row norm.
    const double root = __dsqrt_rn(__ddiv_rn(double(sum),double(k)));
    unsigned lo = __float_as_uint(__double2float_rn(root)) >> 16;
    const double low = double(bf(lo));
    if (low * low * k > double(sum)) --lo;
    const double mid = (double(bf(lo)) + double(bf(lo + 1))) * .5;
    const double mid_sq = mid * mid * k;
    lo += (sum > mid_sq || (sum == mid_sq && (lo & 1u)));
    const float rms = fmaxf(bf((lo + 2u) & 0xfffcu), 0x1p-32f);
    const float maximum = fmaxf(float(peak), 0x1p-32f);
    // BF16(sqrt(32)*0.5) and that same rounded factor divided by 65536.
    const float weighted = __double2float_rn(double(bf(0x4035)) * rms + maximum);
    const float alpha = bf_round(__fdiv_rn(448.f,bf_round(weighted)));
    const float scaled = __fmul_rn(alpha,rms);
    const float beta = bf_round(__fmul_rn(bf_round(scaled),bf(0x3835)));
    if (scaled < 4.f) { atomicOr(invalid, 1u); return {}; }
    const unsigned exponent = (__float_as_uint(scaled) >> 23) & 255u;
    const uint8_t strip = uint8_t(((exponent - 129u) << 3) & 120u);
    return {alpha, beta, strip, e4m3(__fmul_rn(scaled, scaled < 8.f ? 0.1f : 0.08f)), 0};
}

__global__ void row_stats(const int8_t* __restrict__ codes, int k, Row* __restrict__ rows, unsigned* invalid) {
    constexpr int Warps = 8;
    __shared__ unsigned sums[Warps], peaks[Warps];
    const auto* src = reinterpret_cast<const char4*>(codes + size_t(blockIdx.x)*k);
    unsigned sum=0, peak=0;
    for (int i=threadIdx.x; i<k/4; i+=blockDim.x) {
        const char4 v=src[i];
        const int a=v.x,b=v.y,c=v.z,d=v.w;
        sum += a*a+b*b+c*c+d*d;
        peak=max(peak, unsigned(max(max(abs(a),abs(b)),max(abs(c),abs(d)))));
    }
    for (int delta=16; delta; delta>>=1) {
        sum += __shfl_down_sync(0xffffffffu,sum,delta);
        peak = max(peak,__shfl_down_sync(0xffffffffu,peak,delta));
    }
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    if (!lane) { sums[warp]=sum; peaks[warp]=peak; }
    __syncthreads();
    if (warp==0) {
        sum=lane<Warps?sums[lane]:0;
        peak=lane<Warps?peaks[lane]:0;
        for (int delta=4; delta; delta>>=1) {
            sum += __shfl_down_sync(0xffffffffu,sum,delta);
            peak = max(peak,__shfl_down_sync(0xffffffffu,peak,delta));
        }
        if (!lane) rows[blockIdx.x]=row_coefficients(sum,peak,k,invalid);
    }
}

__device__ __forceinline__ unsigned packed4(const uint8_t* plane, int row, int rank_offset) {
    return *reinterpret_cast<const unsigned*>(plane+size_t(row)*32+rank_offset);
}
__device__ __forceinline__ unsigned final_code(float mixed, bool change, Row p) {
    unsigned raw=e4m3(mixed);
    if (change && (raw&127u)<p.threshold)
        raw=(raw&128u)|p.strip|(raw&7u)|1u;
    return raw;
}

template<bool InPlace>
__global__ void noise_and_mix(const int8_t* __restrict__ codes, const uint8_t* __restrict__ e,
                              const uint8_t* __restrict__ f, const Row* __restrict__ params,
                              uint8_t* __restrict__ out, int m, int k) {
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    // The in-place specialization reads through the same pointer it writes,
    // after the statistics kernel has read the complete original slab.
    const int8_t* input=InPlace?reinterpret_cast<const int8_t*>(out):codes;
    const int col0=int(blockIdx.x)*128+warp*32, row0=int(blockIdx.y)*16+lane/4;
    if (col0>=k) return; // all lanes of this warp agree; k is a multiple of 32
    const int pos=(lane&3)*4;
    unsigned a0=0,a1=0,a2=0,a3=0;
    if (row0<m) { a0=packed4(e,row0,pos); a2=packed4(e,row0,pos+16); }
    if (row0+8<m) { a1=packed4(e,row0+8,pos); a3=packed4(e,row0+8,pos+16); }
    Row p[2]={};
    if (row0<m) p[0]=params[row0];
    if (row0+8<m) p[1]=params[row0+8];
    #pragma unroll
    for (int j=0;j<4;++j) {
        const int frow=col0+j*8+lane/4;
        unsigned b0=packed4(f,frow,pos),b1=packed4(f,frow,pos+16);
        float d[4]={};
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
                     : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        const int col=col0+j*8+(lane&3)*2;
        #pragma unroll
        for (int part=0;part<2;++part) {
            const int row=row0+part*8;
            const size_t idx=size_t(row)*k+col;
            char2 inputs={0,0};
            if(row<m) inputs=*reinterpret_cast<const char2*>(input+idx);
            const float lo=__fmaf_rn(d[part*2],p[part].beta,__fmul_rn(p[part].alpha,float(inputs.x)));
            const float hi=__fmaf_rn(d[part*2+1],p[part].beta,__fmul_rn(p[part].alpha,float(inputs.y)));
            // Adjacent lanes own pairs of cells in the row; every four cells
            // share the decision from the first mixed FP32 word.
            const unsigned first=__shfl_sync(0xffffffffu,__float_as_uint(lo),lane & ~1);
            const bool change=((first>>8)&127u)<112u;
            const uint16_t pair=uint16_t(final_code(lo,change,p[part])|(final_code(hi,change,p[part])<<8));
            if(row<m) *reinterpret_cast<uint16_t*>(out+idx)=pair;
        }
    }
}

// Inputs and outputs are ordinary row-major device buffers. The caller owns
// their lifetime and supplies an initially zero invalid flag. No allocation,
// synchronization or host/device copies occur in these launch wrappers.
inline cudaError_t statistics(const int8_t* codes, int rows, int k, Row* params, unsigned* invalid, cudaStream_t stream) {
    row_stats<<<rows,256,0,stream>>>(codes,k,params,invalid);
    return cudaGetLastError();
}
inline cudaError_t quantize(const int8_t* codes, const uint8_t* e, const uint8_t* f,
                            const Row* params, uint8_t* out, int rows, int k, cudaStream_t stream) {
    dim3 grid((k+127)/128,(rows+15)/16);
    if (reinterpret_cast<const void*>(codes)==static_cast<const void*>(out))
        noise_and_mix<true><<<grid,128,0,stream>>>(nullptr,e,f,params,out,rows,k);
    else
        noise_and_mix<false><<<grid,128,0,stream>>>(codes,e,f,params,out,rows,k);
    return cudaGetLastError();
}
}
