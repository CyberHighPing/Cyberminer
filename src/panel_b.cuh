#pragma once
#include "job_generator.cuh"
#ifdef __CUDACC__
#include "prep_inputs.cuh"
#endif
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace pearl_panel_b {
using pearl_jobgen::Digest;
using pearl_jobgen::Plan;
using pearl_jobgen::Roots;
using pearl_jobgen::Size;
using pearl_jobgen::Word;
using Byte=std::uint8_t;
constexpr Size chunk=1024,panel_limit=48ull<<20,ceiling=7168ull<<20;
enum class Mode{Automatic,Whole,Panel};
inline Mode selected(){
    const char* v=std::getenv("PEARL_B_COMMIT");
    if(!v || !*v || !std::strcmp(v,"auto"))return Mode::Automatic;
    if(!std::strcmp(v,"whole"))return Mode::Whole;
    if(!std::strcmp(v,"panel"))return Mode::Panel;
    throw std::invalid_argument("PEARL_B_COMMIT must be auto, whole or panel");
}
inline bool enabled(Mode mode,Size whole_bytes){return mode==Mode::Panel || (mode==Mode::Automatic && whole_bytes>ceiling);}
inline Size panel_capacity(const Plan& p) {
    if (!p.k || !p.value_bytes) throw std::invalid_argument("empty panel operand");
    if (p.value_bytes <= panel_limit) return p.value_bytes;
    // Interior boundaries must start both a source row and a 1024-byte Merkle leaf.
    // K is 32-bit, so this least common multiple fits in Size.
    const Size unit = std::lcm(Size(p.k), chunk);
    if (unit > panel_limit) throw std::invalid_argument("aligned panel unit exceeds the panel limit");
    return panel_limit - panel_limit % unit;
}
inline Size cv_offset(const Plan& p){return (panel_capacity(p)+31u)&~Size(31);}
inline Size storage(const Plan& p){
    const Size head=cv_offset(p);if(p.scratch_bytes>std::numeric_limits<Size>::max()-head)throw std::overflow_error("panel Merkle workspace overflow");
    return head+p.scratch_bytes;
}

#ifdef __CUDACC__
#define PANEL_INLINE __host__ __device__ __forceinline__
#else
#define PANEL_INLINE inline
#endif
// Raw addressing is relative to this panel. The keyed compression counter,
// both terminal flags and all sixteen blocks have the original global meaning.
PANEL_INLINE Digest value_leaf(const Byte* panel,Size local_bytes,Size local_leaf,Size global_leaf,const Digest& key,bool single){
    Digest state=key;
    for(unsigned block=0;block<16;++block){
        Word msg[16];pearl_jobgen::load_plane_block<false>(panel,local_leaf*chunk+block*64,local_bytes,msg);
        Word flags=16u|(block==0?1u:0u)|(block==15?(2u|(single?8u:0u)):0u);
        pearl_jobgen::detail::compress<false>(state.words,msg,global_leaf,64,flags,state.words);
    }
    return state;
}
#undef PANEL_INLINE

#ifdef __CUDACC__
namespace detail=pearl_jobgen::detail;
static __global__ void value_leaves_kernel(const Byte* raw,Size size,Size count,Size first,Digest key,bool single,Digest* global){
    const Size step=Size(gridDim.x)*blockDim.x;
    for(Size i=Size(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=step)global[first+i]=value_leaf(raw,size,i,first+i,key,single);
}
inline cudaError_t generate_bytes(Byte* raw,Size capacity,Size size,Size first_byte,const Digest& seed,cudaStream_t stream,pearl_jobgen::Side side=pearl_jobgen::Side::B){
    if(!raw || !size || size>capacity || first_byte%32)return cudaErrorInvalidValue;
    detail::generate_kernel<<<detail::blocks(1+(size-1)/32),detail::threads,0,stream>>>(raw,size,side,seed,first_byte/32);
    return cudaGetLastError();
}
inline cudaError_t value_leaves(Byte* panel,Size capacity,const Plan& p,const Digest& seed,const Digest& key,Digest* global,cudaStream_t stream,prep_inputs::Row* row_params=nullptr,Word* invalid=nullptr,pearl_jobgen::Side side=pearl_jobgen::Side::B){
    if(!pearl_jobgen::valid_plan(p) || !global || !panel || capacity<panel_capacity(p))return cudaErrorInvalidValue;
    const Size panel_bytes=panel_capacity(p);
    if(row_params&&(!invalid||p.k%32||panel_bytes%p.k))return cudaErrorInvalidValue;
    for(Size start=0;start<p.value_bytes;){
        const Size used=std::min(panel_bytes,p.value_bytes-start),count=1+(used-1)/chunk;
        auto err=row_params?prep_inputs::generate_and_measure(panel,capacity,p,side,seed,start/p.k,used/p.k,row_params+start/p.k,invalid,stream):
            generate_bytes(panel,capacity,used,start,seed,stream,side);if(err!=cudaSuccess)return err;
        value_leaves_kernel<<<detail::blocks(count),detail::threads,0,stream>>>(panel,used,count,start/chunk,key,p.value_chunks==1,global);
        err=cudaGetLastError();if(err!=cudaSuccess)return err;start+=used;
    }
    return cudaSuccess;
}
inline cudaError_t commit(Byte* arena,Size capacity,const Plan& p,const Digest& seed,const Digest& key,Roots& host,cudaStream_t stream,prep_inputs::Row* row_params=nullptr,Word* invalid=nullptr,bool inline_leaves=false,pearl_jobgen::Side side=pearl_jobgen::Side::B){
    if(!pearl_jobgen::valid_plan(p) || !arena || capacity<storage(p))return cudaErrorInvalidValue;
    auto* level=reinterpret_cast<Digest*>(arena+cv_offset(p));auto* next=level+p.value_chunks;
    auto* roots=pearl_jobgen::device_roots(level,p);
    cudaError_t err;
    if(!inline_leaves)err=value_leaves(arena,panel_capacity(p),p,seed,key,level,stream,row_params,invalid,side);
    else if(side==pearl_jobgen::Side::A)err=prep_inputs::direct_b_leaves<pearl_jobgen::Side::A>(p,seed,key,level,row_params,invalid,stream);
    else err=prep_inputs::direct_b_leaves<pearl_jobgen::Side::B>(p,seed,key,level,row_params,invalid,stream);if(err!=cudaSuccess)return err;
    auto count=p.value_chunks;Digest* from=level;Digest* to=next;
    while(count>1){const Size n=(count+1)/2;
        detail::parents_kernel<<<detail::blocks(n),detail::threads,0,stream>>>(from,count,key,to);
        err=cudaGetLastError();if(err!=cudaSuccess)return err;std::swap(from,to);count=n;
    }
    err=cudaMemcpyAsync(&roots->values,from,sizeof(Digest),cudaMemcpyDeviceToDevice,stream);if(err!=cudaSuccess)return err;
    err=detail::tree<true>(nullptr,p.scale_bytes,p.scale_chunks,key,level,next,&roots->scales,stream);if(err!=cudaSuccess)return err;
    detail::operand_kernel<<<1,1,0,stream>>>(roots,key);err=cudaGetLastError();if(err!=cudaSuccess)return err;
    err=cudaMemcpyAsync(&host,roots,sizeof(Roots),cudaMemcpyDeviceToHost,stream);
    return err==cudaSuccess?cudaStreamSynchronize(stream):err;
}
#endif
} // namespace pearl_panel_b
