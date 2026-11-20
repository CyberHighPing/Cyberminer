#pragma once
#include "ticket_filter.cuh"
#include <cstdint>
#include <cstring>
#ifdef __CUDACC__
#include "prep_gpu.cuh"
#define PRL_NOISE_FN __host__ __device__ __forceinline__
#else
#define PRL_NOISE_FN inline
#endif

namespace pearl_noise {
struct Key { uint32_t word[8]; };
static_assert(sizeof(Key)==32,"noise-line key must contain exactly 32 bytes");
inline Key key_from_bytes(const uint8_t* p) {
    Key k{};
    for(int i=0;i<8;++i)
        k.word[i]=uint32_t(p[4*i])|(uint32_t(p[4*i+1])<<8)|
                  (uint32_t(p[4*i+2])<<16)|(uint32_t(p[4*i+3])<<24);
    return k;
}

// Restoring integer square root. For the rank-32 generator the input is
// <= 32*128^2*1024 = 2^29. The implementation also covers all uint32 values.
PRL_NOISE_FN uint32_t floor_sqrt(uint32_t value) {
    uint32_t result=0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for(int step=15;step>=0;--step) {
        const uint32_t bit=uint32_t(1)<<(2*step);
        if(value>=result+bit) {
            value-=result+bit;
            result=(result>>1)+bit;
        } else result>>=1;
    }
    return result;
}

PRL_NOISE_FN float round_bf16(float value) {
#if defined(__CUDA_ARCH__)
    return pearl_prep::bf_round(value);
#else
    uint32_t u;
    std::memcpy(&u,&value,4);
    u=(u+0x7fffu+((u>>16)&1u))&0xffff0000u;
    float result;std::memcpy(&result,&u,4);
    return result;
#endif
}

PRL_NOISE_FN uint8_t fp8_rne(float value) {
#if defined(__CUDA_ARCH__)
    return pearl_prep::e4m3(value);
#else
    uint32_t bits;std::memcpy(&bits,&value,4);
    uint32_t mag=bits&0x7fffffffu,code;
    if(mag<0x3c800000u) {
        float f;std::memcpy(&f,&mag,4);
        f+=16384.f;std::memcpy(&code,&f,4);
        code-=0x46800000u;
    } else {
        code=((mag+0x7ffffu+((mag>>20)&1u))>>20)-960u;
        if(code>126u)code=126u;
    }
    return uint8_t(code|((bits>>24)&128u));
#endif
}

PRL_NOISE_FN void normalize_digest(const uint32_t digest[8],uint32_t out[8]) {
    uint32_t sum=0;
#ifdef __CUDACC__
#pragma unroll
#endif
    for(int j=0;j<8;++j) {
#ifdef __CUDACC__
#pragma unroll
#endif
        for(int i=0;i<4;++i) {
            // Absolute coefficient lies in [1,128], even for byte 0x80.
            const uint32_t magnitude=((digest[j]>>(8*i))&127u)+1u;
            sum+=magnitude*magnitude;
        }
    }
    const float denom=round_bf16(float(floor_sqrt(sum<<10)));
#if defined(__CUDA_ARCH__)
    const float gain=round_bf16(__fdiv_rn(8192.f,denom));
#else
    const float gain=round_bf16(8192.f/denom);
#endif
#ifdef __CUDACC__
#pragma unroll
#endif
    for(int j=0;j<8;++j) {
        uint32_t packed=0;
#ifdef __CUDACC__
#pragma unroll
#endif
        for(int i=0;i<4;++i) {
            const uint32_t u=(digest[j]>>(8*i))&255u;
            const int abs=int((u&127u)+1u);
            const float x=float((u&128u)?-abs:abs);
            // A signed 8-bit magnitude times a BF16 gain fits exactly in FP32.
            packed |= uint32_t(fp8_rne(round_bf16(x*gain)))<<(8*i);
        }
        out[j]=packed;
    }
}

// For <BBI58x> the index straddles the first two little-endian message words.
// digest_fold is ordinary keyed BLAKE3 on one complete 64-byte root block;
// key is the pre-derived noise-line key, without any further role hashing.
PRL_NOISE_FN void generate_line(const Key& key,uint32_t side,uint32_t factor,uint32_t index,uint32_t out[8]) {
    uint32_t address[16]={side|(factor<<8)|(index<<16), index>>16};
    uint32_t digest[8];
    pearl_ticket::digest_fold(key.word,address,digest);
    normalize_digest(digest,out);
}
}
#undef PRL_NOISE_FN
