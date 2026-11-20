#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fp8_sm120 {
using Word = std::uint32_t;

__device__ __forceinline__ Word shared_address(const void* p){return static_cast<Word>(__cvta_generic_to_shared(p));}

__device__ __forceinline__ void initialize_barrier(std::uint64_t* p,int count){
    asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(shared_address(p)),"r"(count):"memory");
}

__device__ __forceinline__ void wait_barrier(std::uint64_t* p,int phase){
    const Word address=shared_address(p);Word done;
    do{
        asm volatile("{.reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p,[%1],%2,0x100000;"
                     "selp.u32 %0,1,0,p;}":"=r"(done):"r"(address),"r"(phase):"memory");
    }while(!done);
}

__device__ __forceinline__ void release_stage(std::uint64_t* p){
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"::"r"(shared_address(p)):"memory");
}

__device__ __forceinline__ void multiply32(float (&d)[4],const Word (&a)[4],const Word (&b)[2]){
    constexpr Word ones=0x7f7f7f7fu; // UE8M0: each selected scale byte is +1.
    constexpr unsigned short selector=0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%15},{%14},{%15,%15};"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
         "f"(d[0]),"f"(d[1]),"f"(d[2]),"f"(d[3]),"r"(ones),"h"(selector));
}

__device__ __forceinline__ Word canonical_bits(float x){
    const Word bits=__float_as_uint(x);
    return (bits&0x7fffffffu)>0x7f800000u?0x7fc00000u:bits;
}

template<bool Finite> __device__ __forceinline__ Word output_bits(float x){
    if constexpr(Finite)return __float_as_uint(x);
    return canonical_bits(x);
}

__device__ __forceinline__ Word fold_step(Word x,Word bits){
    x=x*0x9e3779b1u+bits;return __funnelshift_l(x,x,13);
}

__device__ __forceinline__ void exchange(Word& low,Word& high,int step,int lane){
    const Word send=(lane&step)?low:high;
    const Word received=__shfl_xor_sync(0xffffffffu,send,step,4);
    if(lane&step)low=received;else high=received;
}

__device__ __forceinline__ void transpose(Word (&v)[4],int lane){
    exchange(v[0],v[1],1,lane);exchange(v[2],v[3],1,lane);
    exchange(v[0],v[2],2,lane);exchange(v[1],v[3],2,lane);
}

inline void check_cuda(cudaError_t e,const char* op){
    if(e!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));
}

inline void check_driver(CUresult e,const char* op){
    if(e!=CUDA_SUCCESS)throw std::runtime_error(std::string(op)+" failed with CUDA driver status "+std::to_string(int(e)));
}

template<class Function> Function driver_function(const char* name){
    void* address=nullptr;cudaDriverEntryPointQueryResult status{};
    check_cuda(cudaGetDriverEntryPointByVersion(name,&address,12000,cudaEnableDefault,&status),name);
    if(!address||status!=cudaDriverEntryPointSuccess)throw std::runtime_error(std::string("required CUDA driver entry point unavailable: ")+name);
    return reinterpret_cast<Function>(address);
}

struct Driver {
    using Encode=decltype(&::cuTensorMapEncodeTiled);
    using Replace=decltype(&::cuTensorMapReplaceAddress);
    Encode encode=driver_function<Encode>("cuTensorMapEncodeTiled");
    Replace replace=driver_function<Replace>("cuTensorMapReplaceAddress");
};

}  // namespace fp8_sm120
