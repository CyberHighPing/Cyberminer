#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define PJ_INLINE __host__ __device__ __forceinline__
#else
#define PJ_INLINE inline
#endif

namespace pearl_jobgen {
using Word = std::uint32_t;
using Size = std::uint64_t;
struct Digest { Word words[8]; };
struct Roots { Digest values,scales,combined; };
static_assert(sizeof(Digest)==32 && sizeof(Roots)==96,"digest wire layout");
enum class Side : unsigned char {A='A', B='B'};
PJ_INLINE Digest from_bytes(const std::uint8_t* p) {
    Digest d{};
    for(unsigned i=0;i<8;++i)
        d.words[i]=Word(p[4*i])|(Word(p[4*i+1])<<8)|(Word(p[4*i+2])<<16)|(Word(p[4*i+3])<<24);
    return d;
}
PJ_INLINE void to_bytes(const Digest& d,std::uint8_t* p) {
    for(unsigned i=0;i<32;++i)p[i]=std::uint8_t(d.words[i/4]>>(8*(i%4)));
}
struct Plan {
    Size rows=0;
    Word k=0;
    Size value_bytes=0, scale_bytes=0, value_chunks=0, scale_chunks=0;
    std::size_t scratch_bytes=0;
};

inline bool make_plan(Size rows, Word k, Plan& output) {
    if (!rows || !k || k%8 || rows>(std::numeric_limits<Size>::max)()/k) return false;
    Plan p{};
    p.rows=rows; p.k=k; p.value_bytes=rows*k; p.scale_bytes=p.value_bytes/4;
    if(p.value_bytes>(std::numeric_limits<std::size_t>::max)()) return false;
    p.value_chunks=1+(p.value_bytes-1)/1024;
    p.scale_chunks=1+(p.scale_bytes-1)/1024;
    const Size cvs=p.value_chunks+(p.value_chunks+1)/2;
    if(cvs>((std::numeric_limits<std::size_t>::max)()-sizeof(Roots))/sizeof(Digest)) return false;
    p.scratch_bytes=std::size_t(cvs)*sizeof(Digest)+sizeof(Roots);
    output=p;
    return true;
}

namespace detail {
template<unsigned N>
PJ_INLINE Word rotr(Word x) {
#ifdef __CUDA_ARCH__
    return __funnelshift_r(x,x,N);
#else
    return (x>>N)|(x<<(32-N));
#endif
}
PJ_INLINE void g(Word& a, Word& b, Word& c, Word& d, Word x, Word y) {
    a+=b+x;d=rotr<16>(d^a);c+=d;b=rotr<12>(b^c);
    a+=b+y;d=rotr<8>(d^a);c+=d;b=rotr<7>(b^c);
}
PJ_INLINE constexpr unsigned perm(unsigned i) {
    constexpr unsigned order[16]={2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};
    return order[i];
}
PJ_INLINE constexpr unsigned at(unsigned r,unsigned i) {
    return r?at(r-1,perm(i)):i;
}
template<unsigned R>
PJ_INLINE void round(Word* v,const Word* m) {
    g(v[0],v[4],v[8],v[12],m[at(R,0)],m[at(R,1)]);
    g(v[1],v[5],v[9],v[13],m[at(R,2)],m[at(R,3)]);
    g(v[2],v[6],v[10],v[14],m[at(R,4)],m[at(R,5)]);
    g(v[3],v[7],v[11],v[15],m[at(R,6)],m[at(R,7)]);
    g(v[0],v[5],v[10],v[15],m[at(R,8)],m[at(R,9)]);
    g(v[1],v[6],v[11],v[12],m[at(R,10)],m[at(R,11)]);
    g(v[2],v[7],v[8],v[13],m[at(R,12)],m[at(R,13)]);
    g(v[3],v[4],v[9],v[14],m[at(R,14)],m[at(R,15)]);
}
// The input chaining value may alias out in the 32-byte mode.
template<bool Xof>
PJ_INLINE void compress(const Word* cv,const Word* msg,Size counter,Word len,Word flags,Word* out) {
    Word v[16]={cv[0],cv[1],cv[2],cv[3],cv[4],cv[5],cv[6],cv[7],
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        Word(counter),Word(counter>>32),len,flags};
    round<0>(v,msg);round<1>(v,msg);round<2>(v,msg);round<3>(v,msg);
    round<4>(v,msg);round<5>(v,msg);round<6>(v,msg);
#ifdef __CUDACC__
#pragma unroll
#endif
    for(unsigned i=0;i<8;++i) {
        if constexpr (Xof) out[i+8]=v[i+8]^cv[i];
        out[i]=v[i]^v[i+8];
    }
}
PJ_INLINE constexpr Word le4(char a,char b,char c,char d) {
    return Word(a)|(Word(b)<<8)|(Word(c)<<16)|(Word(d)<<24);
}
PJ_INLINE Word triangular_pair(Word random) {
    return (((random&127)+((random>>8)&127)-127)&255) |
           (((((random>>16)&127)+((random>>24)&127)-127)&255)<<8);
}
} // namespace detail

#ifdef PEARL_SCAN_INPUTS
// Each resulting byte is (two seven-bit samples + 1) xor 0x80.
// The largest pre-XOR value is 255, so adjacent byte sums cannot carry.
PJ_INLINE Word pack_samples(Word a, Word b) {
#ifdef __CUDA_ARCH__
    const Word even=__byte_perm(a,b,0x6420), odd=__byte_perm(a,b,0x7531);
#else
    const Word even=(a&255u)|((a>>8)&0xff00u)|((b&255u)<<16)|((b&0xff0000u)<<8);
    const Word odd=((a>>8)&255u)|((a>>16)&0xff00u)|((b&0xff00u)<<8)|(b&0xff000000u);
#endif
    return ((even&0x7f7f7f7fu)+(odd&0x7f7f7f7fu)+0x01010101u)^0x80808080u;
}

#endif  // PEARL_SCAN_INPUTS
// One XOF output block covers precisely 32 dense values. No generator state
// is shared between blocks or between independent A/B labels.
PJ_INLINE void raw_block(const Digest& seed, Side side,Size index,Word* out8) {
    const Word msg[16]={detail::le4('p','e','a','r'),detail::le4('l','-','5','0'),
        detail::le4('6','0','/','d'),detail::le4('e','n','s','e'),
        detail::le4('-','x','o','f'),detail::le4('/','v','2','/'),
        Word(static_cast<unsigned char>(side)),0,0,0,0,0,0,0,0,0};
    Word x[16];
    detail::compress<true>(seed.words,msg,index,25,27,x);
#ifdef __CUDACC__
#pragma unroll
#endif
    for(unsigned i=0;i<8;++i)
#ifdef PEARL_SCAN_INPUTS
        out8[i]=pack_samples(x[2*i],x[2*i+1]);
#else  // PEARL_SCAN_INPUTS
        out8[i]=detail::triangular_pair(x[2*i])|(detail::triangular_pair(x[2*i+1])<<16);
#endif  // PEARL_SCAN_INPUTS
}

#ifdef PEARL_SCAN_INPUTS
// For one generator key, side and 32 high counter bits, columns 1..3 of
// the first BLAKE3 round are identical at every output block. Two diagonal
// half-round prefixes also have no dependency on the varying low counter.
struct RawPrefix { Word state[16]; Word counter_high; };
PJ_INLINE RawPrefix raw_prefix(const Digest& seed,Side side,Word counter_high=0) {
    RawPrefix p{{seed.words[0],seed.words[1],seed.words[2],seed.words[3],
                 seed.words[4],seed.words[5],seed.words[6],seed.words[7],
                 0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                 0,counter_high,25,27},counter_high};
    auto* v=p.state;
    detail::g(v[1],v[5],v[9],v[13],detail::le4('6','0','/','d'),detail::le4('e','n','s','e'));
    detail::g(v[2],v[6],v[10],v[14],detail::le4('-','x','o','f'),detail::le4('/','v','2','/'));
    detail::g(v[3],v[7],v[11],v[15],Word(static_cast<unsigned char>(side)),0);
    v[0]+=v[4]+detail::le4('p','e','a','r');
    v[1]+=v[6];
    v[2]+=v[7];
    v[13]=detail::rotr<16>(v[13]^v[2]);
    return p;
}

PJ_INLINE void raw_after_rotation(Word& a,Word& b,Word& c,Word& d,Word y) {
    c+=d;b=detail::rotr<12>(b^c);
    a+=b+y;d=detail::rotr<8>(d^a);c+=d;b=detail::rotr<7>(b^c);
}

PJ_INLINE void raw_from_matching_prefix(const Digest& seed,Side side,const RawPrefix& prefix,Size counter,Word* out) {
    const Word msg[16]={detail::le4('p','e','a','r'),detail::le4('l','-','5','0'),
        detail::le4('6','0','/','d'),detail::le4('e','n','s','e'),detail::le4('-','x','o','f'),
        detail::le4('/','v','2','/'),Word(static_cast<unsigned char>(side)),0,0,0,0,0,0,0,0,0};
    Word v[16];
#ifdef __CUDACC__
#pragma unroll
#endif
    for(unsigned j=0;j<16;++j)v[j]=prefix.state[j];
    v[12]=detail::rotr<16>(Word(counter)^v[0]);
    raw_after_rotation(v[0],v[4],v[8],v[12],msg[1]);
    detail::g(v[0],v[5],v[10],v[15],0,0);
    v[12]=detail::rotr<16>(v[12]^v[1]);
    raw_after_rotation(v[1],v[6],v[11],v[12],0);
    raw_after_rotation(v[2],v[7],v[8],v[13],0);
    detail::g(v[3],v[4],v[9],v[14],0,0);
    detail::round<1>(v,msg);detail::round<2>(v,msg);detail::round<3>(v,msg);
    detail::round<4>(v,msg);detail::round<5>(v,msg);detail::round<6>(v,msg);
#ifdef __CUDACC__
#pragma unroll
#endif
    for(unsigned i=0;i<4;++i) {
        const unsigned j=2*i;
        out[i]=pack_samples(v[j]^v[j+8],v[j+1]^v[j+9]);
        out[4+i]=pack_samples(v[j+8]^seed.words[j],v[j+9]^seed.words[j+1]);
    }
}

#endif  // PEARL_SCAN_INPUTS
template<bool Scales>
PJ_INLINE void load_plane_block(const std::uint8_t* raw,Size offset,Size size,Word* msg) {
    const Size available=offset<size?size-offset:0;
    if(available>=64) {
        if constexpr (Scales) {
#ifdef __CUDACC__
#pragma unroll
#endif
            for(unsigned i=0;i<16;++i) msg[i]=0x3f803f80u;
            return;
        }
#ifdef __CUDA_ARCH__
        if((reinterpret_cast<std::uintptr_t>(raw)&15)==0) {
#pragma unroll
            for(unsigned i=0;i<4;++i) {
                const uint4 x=reinterpret_cast<const uint4*>(raw+offset)[i];
                msg[4*i]=x.x;msg[4*i+1]=x.y;msg[4*i+2]=x.z;msg[4*i+3]=x.w;
            }
            return;
        }
#endif
    }
#ifdef __CUDACC__
#pragma unroll
#endif
    for(unsigned i=0;i<16;++i) {
        Word v=0;
#ifdef __CUDACC__
#pragma unroll
#endif
        for(unsigned j=0;j<4;++j) {
            const unsigned pos=4*i+j;
            if(pos<available) {
                Word b;
                if constexpr(Scales) b=(pos&1)?0x3f:0x80;
                else b=raw[offset+pos];
                v|=b<<(8*j);
            }
        }
        msg[i]=v;
    }
}

template<bool Scales>
PJ_INLINE Digest leaf(const std::uint8_t* raw, Size size,Size index,const Digest& key,bool only_leaf) {
    Digest state=key;
    for(unsigned block=0;block<16;++block) {
        Word m[16];
        load_plane_block<Scales>(raw,index*1024+block*64,size,m);
        const Word flags=16u|(block==0?1u:0u)|(block==15?(2u|(only_leaf?8u:0u)):0u);
        detail::compress<false>(state.words,m,index,64,flags,state.words);
    }
    return state;
}

PJ_INLINE Digest fold_pair(const Digest& left,const Digest& right,const Digest& key,Word flags) {
    Word msg[16];
#ifdef __CUDACC__
#pragma unroll
#endif
    for(unsigned i=0;i<8;++i){msg[i]=left.words[i];msg[i+8]=right.words[i];}
    Digest out;
    detail::compress<false>(key.words,msg,0,64,flags,out.words);
    return out;
}

#ifdef __CUDACC__
namespace detail {
constexpr unsigned threads=128;
inline unsigned blocks(Size n) {
    const Size grid=1+(n-1)/threads;
    return unsigned(grid>65535?65535:grid);
}
static __global__ void generate_kernel(std::uint8_t* raw,Size bytes,Side side,Digest seed,Size first_block=0) {
    const Size count=1+(bytes-1)/32,step=Size(gridDim.x)*blockDim.x;
    for(Size i=Size(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=step) {
        Word out[8]; raw_block(seed,side,first_block+i,out);
        const Size p=32*i;
        if(bytes-p>=32 && !(reinterpret_cast<std::uintptr_t>(raw+p)&15)) {
            uint4* dst=reinterpret_cast<uint4*>(raw+p);
            dst[0]=make_uint4(out[0],out[1],out[2],out[3]);
            dst[1]=make_uint4(out[4],out[5],out[6],out[7]);
        } else {
            for(unsigned j=0;j<32 && j<bytes-p;++j)
                raw[p+j]=std::uint8_t(out[j/4]>>(8*(j%4)));
        }
    }
}
template<bool Scales>
static __global__ void leaf_kernel(const std::uint8_t* raw,Size size,Size count,Digest key,Digest* out) {
    const Size step=Size(gridDim.x)*blockDim.x;
    for(Size i=Size(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=step)
        out[i]=leaf<Scales>(raw,size,i,key,count==1);
}
static __global__ void parents_kernel(const Digest* cur,Size count,Digest key,Digest* out) {
    const Size n=(count+1)/2,step=Size(gridDim.x)*blockDim.x;
    for(Size i=Size(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=step) {
        const Size child=i*2;
        out[i]=child+1<count?fold_pair(cur[child],cur[child+1],key,count==2?28:20):cur[child];
    }
}
static __global__ void operand_kernel(Roots* out,Digest key) {
    out->combined=fold_pair(out->values,out->scales,key,27);
}
template<bool Scales>
inline cudaError_t tree(const std::uint8_t* raw,Size size,Size count,const Digest& key,
                        Digest* a,Digest* b,Digest* root,cudaStream_t stream,
                        Digest* host_cut=nullptr,Size cut_count=0) {
    leaf_kernel<Scales><<<blocks(count),threads,0,stream>>>(raw,size,count,key,count==1?root:a);
    auto err=cudaGetLastError();
    if(err==cudaSuccess && host_cut && count==cut_count)
        err=cudaMemcpyAsync(host_cut,count==1?root:a,count*sizeof(Digest),cudaMemcpyDeviceToHost,stream);
    if(err!=cudaSuccess || count==1) return err;
    while(count>1) {
        const Size next=(count+1)/2;
        Digest* dst=next==1?root:b;
        parents_kernel<<<blocks(next),threads,0,stream>>>(a,count,key,dst);
        err=cudaGetLastError();
        if(err!=cudaSuccess) return err;
        b=a;a=dst;count=next;
        if(host_cut && count==cut_count) {
            err=cudaMemcpyAsync(host_cut,a,count*sizeof(Digest),cudaMemcpyDeviceToHost,stream);
            if(err!=cudaSuccess)return err;
        }
    }
    return cudaSuccess;
}
} // namespace detail

inline bool valid_plan(const Plan& p) {
    Plan fresh;
    return make_plan(p.rows,p.k,fresh) && p.value_bytes==fresh.value_bytes &&
        p.scale_bytes==fresh.scale_bytes && p.value_chunks==fresh.value_chunks &&
        p.scale_chunks==fresh.scale_chunks && p.scratch_bytes==fresh.scratch_bytes;
}

inline cudaError_t launch_generate(std::uint8_t* d_raw,std::size_t raw_capacity,const Plan& p,
                                   Side side,const Digest& job_seed,cudaStream_t stream) {
    if(!valid_plan(p) || raw_capacity<p.value_bytes || !d_raw || (side!=Side::A && side!=Side::B))
        return cudaErrorInvalidValue;
    detail::generate_kernel<<<detail::blocks(1+(p.value_bytes-1)/32),detail::threads,0,stream>>>(
        d_raw,p.value_bytes,side,job_seed);
    return cudaGetLastError();
}

inline cudaError_t launch_generate_rows(std::uint8_t* out,std::size_t capacity,const Plan& p,
                                        Side side,const Digest& seed,Size first,Size rows,cudaStream_t stream) {
    if(!valid_plan(p) || !out || !rows || first>p.rows || rows>p.rows-first || p.k%32 ||
       (side!=Side::A && side!=Side::B) || rows*p.k>capacity) return cudaErrorInvalidValue;
    const Size bytes=rows*p.k,block0=first*(p.k/32);
    detail::generate_kernel<<<detail::blocks(bytes/32),detail::threads,0,stream>>>(out,bytes,side,seed,block0);
    return cudaGetLastError();
}

inline Roots* device_roots(void* d_scratch,const Plan& p) {
    return reinterpret_cast<Roots*>(static_cast<unsigned char*>(d_scratch)+p.scratch_bytes-sizeof(Roots));
}

inline bool buffers_ok(const std::uint8_t* raw,std::size_t raw_capacity,const Plan& p,
                       void* scratch,std::size_t scratch_capacity) {
    if(!valid_plan(p) || !raw || raw_capacity<p.value_bytes || !scratch ||
       scratch_capacity<p.scratch_bytes) return false;
    const auto a=reinterpret_cast<std::uintptr_t>(raw), b=reinterpret_cast<std::uintptr_t>(scratch);
    if(b%alignof(Roots))return false;
    return a<=b ? b-a>=p.value_bytes : a-b>=p.scratch_bytes;
}

inline cudaError_t launch_commit(const std::uint8_t* d_raw,std::size_t raw_capacity,const Plan& p,
                                 const Digest& tree_key,void* d_scratch,std::size_t scratch_capacity,
                                 cudaStream_t stream) {
    if(!buffers_ok(d_raw,raw_capacity,p,d_scratch,scratch_capacity)) return cudaErrorInvalidValue;
    auto* a=static_cast<Digest*>(d_scratch);
    auto* b=a+p.value_chunks;
    auto* roots=device_roots(d_scratch,p);
    auto e=detail::tree<false>(d_raw,p.value_bytes,p.value_chunks,tree_key,a,b,&roots->values,stream);
    if(e!=cudaSuccess) return e;
    e=detail::tree<true>(nullptr,p.scale_bytes,p.scale_chunks,tree_key,a,b,&roots->scales,stream);
    if(e!=cudaSuccess) return e;
    detail::operand_kernel<<<1,1,0,stream>>>(roots,tree_key);
    return cudaGetLastError();
}

inline cudaError_t generate_and_commit(std::uint8_t* d_raw,std::size_t raw_capacity,const Plan& p,
                                       Side side,const Digest& job_seed,const Digest& tree_key,
                                       void* d_scratch,std::size_t scratch_capacity,Roots* host_roots,
                                       cudaStream_t stream) {
    if(!host_roots || !buffers_ok(d_raw,raw_capacity,p,d_scratch,scratch_capacity))
        return cudaErrorInvalidValue;
    auto e=launch_generate(d_raw,raw_capacity,p,side,job_seed,stream);
    if(e!=cudaSuccess) return e;
    e=launch_commit(d_raw,raw_capacity,p,tree_key,d_scratch,scratch_capacity,stream);
    if(e!=cudaSuccess) return e;
    e=cudaMemcpyAsync(host_roots,device_roots(d_scratch,p),sizeof(Roots),cudaMemcpyDeviceToHost,stream);
    return e!=cudaSuccess?e:cudaStreamSynchronize(stream);
}
#endif
} // namespace pearl_jobgen
#undef PJ_INLINE
