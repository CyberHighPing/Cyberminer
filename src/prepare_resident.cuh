#pragma once
#include "noise_math.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace pearl_noise {
static __global__ void generate_kernel(Key key,uint32_t side,uint32_t factor,uint32_t first,
                                       uint32_t count,uint32_t* output) {
    const uint32_t line=blockIdx.x*blockDim.x+threadIdx.x;
    if(line>=count)return;
    uint32_t words[8];
    generate_line(key,side,factor,first+line,words);
    uint32_t* dest=output+size_t(line)*8;
    // Both the base and every complete line have 16-byte alignment in this API.
    reinterpret_cast<uint4*>(dest)[0]=make_uint4(words[0],words[1],words[2],words[3]);
    reinterpret_cast<uint4*>(dest)[1]=make_uint4(words[4],words[5],words[6],words[7]);
}

// Generate arbitrary, consecutive global indices for either table. The
// low-level entry point also serves the independent GPU table checks.
inline cudaError_t generate_table(const Key& key,int side,int factor,uint32_t first,
                                  uint32_t count,void* d_table,cudaStream_t stream) {
    if(side<0 || side>1 || factor<0 || factor>1 || count>(1u<<24) ||
       (count && (uint64_t(first)+count-1>UINT32_MAX || !d_table ||
                  (reinterpret_cast<uintptr_t>(d_table)&15u))))
        return cudaErrorInvalidValue;
    if(!count)return cudaSuccess;
    generate_kernel<<<(count+127)/128,128,0,stream>>>(key,uint32_t(side),uint32_t(factor),first,count,
                                                     static_cast<uint32_t*>(d_table));
    return cudaGetLastError();
}

inline size_t scratch_bytes(int k,int batch_rows) {
    if(k<=0 || k>65536 || (k%32) || batch_rows<=0 || batch_rows>(1<<24)-1)return 0;
    // F[K,32], E[batch,32], one row-parameter record per active row.
    return size_t(k)*32 + size_t(batch_rows)*(32+sizeof(pearl_prep::Row));
}

// All device buffers are caller-owned. The host key bytes are copied into
// launch arguments before this function returns. d_invalid must already
// contain zero (or earlier failures in the same job). All work is enqueued
// on stream. It is safe to reuse the same scratch for A and B on this stream.
inline cudaError_t prepare_resident(int8_t* d_codes,int rows,int k,int side,
                                    const uint8_t* host_e_line_key32,const uint8_t* host_f_line_key32,
                                    int batch_rows,void* d_scratch,size_t scratch_size,
                                    unsigned* d_invalid,cudaStream_t stream,uint32_t first_row=0) {
    const size_t need=scratch_bytes(k,batch_rows);
    if(!need || scratch_size<need || rows<1 || rows>(1<<24)-1 || side<0 || side>1 ||
       uint64_t(first_row)+rows-1>UINT32_MAX || !host_e_line_key32 || !host_f_line_key32 ||
       !d_codes || (reinterpret_cast<uintptr_t>(d_codes)&3u) || !d_scratch ||
       (reinterpret_cast<uintptr_t>(d_scratch)&15u) || !d_invalid ||
       (reinterpret_cast<uintptr_t>(d_invalid)&3u))return cudaErrorInvalidValue;
    const Key e_key=key_from_bytes(host_e_line_key32), f_key=key_from_bytes(host_f_line_key32);
    auto* f=static_cast<uint8_t*>(d_scratch);
    auto* e=f+size_t(k)*32;
    auto* params=reinterpret_cast<pearl_prep::Row*>(e+size_t(batch_rows)*32);
    cudaError_t s=generate_table(f_key,side,1,0,uint32_t(k),f,stream);
    if(s!=cudaSuccess)return s;
    for(int local=0;local<rows;local+=batch_rows) {
        const int n=rows-local<batch_rows ? rows-local : batch_rows;
        auto* part=d_codes+size_t(local)*k;
        s=generate_table(e_key,side,0,first_row+uint32_t(local),uint32_t(n),e,stream);
        if(s!=cudaSuccess)return s;
        s=pearl_prep::statistics(part,n,k,params,d_invalid,stream);
        if(s!=cudaSuccess)return s;
        s=pearl_prep::quantize(part,e,f,params,reinterpret_cast<uint8_t*>(part),n,k,stream);
        if(s!=cudaSuccess)return s;
    }
    return cudaSuccess;
}
}
