#pragma once
#include "scan_depth.hpp"
#include "job_generator.cuh"
#include "prepare_resident.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace prep_inputs {
using Byte=std::uint8_t;
using Word=std::uint32_t;
using Size=pearl_jobgen::Size;
using Row=pearl_prep::Row;
using Side=pearl_jobgen::Side;
using Digest=pearl_jobgen::Digest;
enum class Mode { off,packed,stats,shared,shuffle,vector,sharedvec,shufflevec,leaf,leafvec };
inline Mode selected(){
    const char* v=std::getenv("PEARL_PREP_INPUTS");
    if(!v||!std::strcmp(v,"off"))return Mode::off;
    if(!std::strcmp(v,"packed"))return Mode::packed;
    if(!std::strcmp(v,"stats"))return Mode::stats;
    if(!std::strcmp(v,"shared"))return Mode::shared;
    if(!std::strcmp(v,"shuffle"))return Mode::shuffle;
    if(!std::strcmp(v,"vector"))return Mode::vector;
    if(!std::strcmp(v,"sharedvec"))return Mode::sharedvec;
    if(!std::strcmp(v,"shufflevec"))return Mode::shufflevec;
    if(!std::strcmp(v,"leaf"))return Mode::leaf;
    if(!std::strcmp(v,"leafvec"))return Mode::leafvec;
    throw std::invalid_argument("unsupported input preparation option");
}
inline const char* name(Mode m){
    switch(m){
        case Mode::off:return "off";case Mode::packed:return "packed";case Mode::stats:return "stats";
        case Mode::shared:return "shared";case Mode::shuffle:return "shuffle";
        case Mode::vector:return "vector";
        case Mode::sharedvec:return "sharedvec";
        case Mode::shufflevec:return "shufflevec";
        case Mode::leaf:return "leaf";
        case Mode::leafvec:return "leafvec";

    }
    return "invalid";
}
inline bool saved_rows(Mode m){return m==Mode::stats||m==Mode::shared||m==Mode::shuffle||m==Mode::sharedvec||m==Mode::shufflevec||m==Mode::leaf||m==Mode::leafvec;}
inline bool direct_b(Mode m){return saved_rows(m)&&m!=Mode::stats;}

__device__ __forceinline__ void collect_word(Word x,int& sum,Word& lane_peaks){
    sum=__dp4a(static_cast<int>(x),static_cast<int>(x),sum);
    lane_peaks=__vmaxu4(lane_peaks,__vabs4(x));
}
template<int Threads>
__device__ __forceinline__ void finish_row(int sum,Word lane_peaks,int k,Row* row,Word* invalid){
    constexpr int warps=Threads/32;
    __shared__ Word warp_sums[warps],warp_peaks[warps];
    Word peak=max(max(lane_peaks&255,(lane_peaks>>8)&255),max((lane_peaks>>16)&255,lane_peaks>>24));
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
#pragma unroll
    for(int shift=16;shift;shift/=2){
        sum+=__shfl_down_sync(0xffffffffu,sum,shift);
        peak=max(peak,__shfl_down_sync(0xffffffffu,peak,shift));
    }
    if(!lane){warp_sums[warp]=sum;warp_peaks[warp]=peak;}
    __syncthreads();
    if(warp==0){
        Word total=lane<warps?warp_sums[lane]:0;
        peak=lane<warps?warp_peaks[lane]:0;
#pragma unroll
        for(int shift=warps/2;shift;shift/=2){
            total+=__shfl_down_sync(0xffffffffu,total,shift);
            peak=max(peak,__shfl_down_sync(0xffffffffu,peak,shift));
        }
        if(!lane)*row=pearl_prep::row_coefficients(total,peak,k,invalid);
    }
}
template<bool Generate>
__global__ void generate_and_measure_kernel(Byte* raw,int k,Side side,Digest seed,Size first,Row* rows,Word* invalid){
    constexpr int threads=256;
    const int row=blockIdx.x;const int units=k/32;
    int sum=0;Word peak=0;
    uint4* dest=reinterpret_cast<uint4*>(raw+size_t(row)*k);
    for(int i=threadIdx.x;i<units;i+=threads){
        Word v[8];
        if constexpr(Generate){
            pearl_jobgen::raw_block(seed,side,(first+row)*units+i,v);
            dest[2*i]=make_uint4(v[0],v[1],v[2],v[3]);
            dest[2*i+1]=make_uint4(v[4],v[5],v[6],v[7]);
        }else{
            const uint4 lo=dest[2*i],hi=dest[2*i+1];
            v[0]=lo.x;v[1]=lo.y;v[2]=lo.z;v[3]=lo.w;v[4]=hi.x;v[5]=hi.y;v[6]=hi.z;v[7]=hi.w;
        }
#pragma unroll
        for(int j=0;j<8;j++)collect_word(v[j],sum,peak);
    }
    finish_row<threads>(sum,peak,k,rows+row,invalid);
}
template<bool Generate=true> inline cudaError_t generate_and_measure(Byte* data,size_t cap,const pearl_jobgen::Plan& p,
                 Side side,const Digest& seed,Size first,Size rows,Row* meta,Word* invalid,cudaStream_t stream){
    if(!pearl_jobgen::valid_plan(p)||!data||!meta||!invalid||!rows||rows>0x7fffffffu||p.k>65536||p.k%32||
       first>p.rows||rows>p.rows-first||rows*p.k>cap||(side!=Side::A&&side!=Side::B)||
       (reinterpret_cast<uintptr_t>(data)&15)||(reinterpret_cast<uintptr_t>(meta)&3)||(reinterpret_cast<uintptr_t>(invalid)&3))return cudaErrorInvalidValue;
    generate_and_measure_kernel<Generate><<<unsigned(rows),256,0,stream>>>(data,p.k,side,seed,first,meta,invalid);
    return cudaGetLastError();
}

// Each lane computes every raw byte and all sixteen keyed compressions of a
// 1024-byte leaf. No temporary global-memory plane is required for the first B pass.
template<int LeafsPerRow,Side operand=Side::B>
__global__ void direct_b_leaves_kernel(Size leaves,Digest seed,Digest key,Digest* values,Row* meta,Word* invalid){
    static_assert(LeafsPerRow==8||LeafsPerRow==16||LeafsPerRow==32);
    const Size index=Size(blockIdx.x)*blockDim.x+threadIdx.x;
    const unsigned active=__ballot_sync(0xffffffffu,index<leaves);
    if(index>=leaves)return;
    Digest state=key;int sum=0;Word lane_peaks=0;
#pragma unroll 1
    for(int b=0;b<16;b++){
        Word msg[16];const Size first=index*32+2*b;
        pearl_jobgen::raw_block(seed,operand,first,msg);
        pearl_jobgen::raw_block(seed,operand,first+1,msg+8);
#pragma unroll
        for(int j=0;j<16;j++)collect_word(msg[j],sum,lane_peaks);
        const Word flags=16u|(b==0?1u:0u)|(b==15?2u:0u);
        pearl_jobgen::detail::compress<false>(state.words,msg,index,64,flags,state.words);
    }
    values[index]=state;
    Word peak=max(max(lane_peaks&255,(lane_peaks>>8)&255),max((lane_peaks>>16)&255,lane_peaks>>24));
#pragma unroll
    for(int d=LeafsPerRow/2;d;d/=2){
        sum+=__shfl_down_sync(active,sum,d,LeafsPerRow);
        peak=max(peak,__shfl_down_sync(active,peak,d,LeafsPerRow));
    }
    if(!(threadIdx.x&(LeafsPerRow-1)))meta[index/LeafsPerRow]=pearl_prep::row_coefficients(sum,peak,LeafsPerRow*1024,invalid);
}
// One lane owns an entire BLAKE3 value leaf. Whole rows stay in one warp even
// when their number of leaves is not a power of two. Adjacent rows never share
// an addition or a maximum in the row policy reduction.
template<int Leaves,Side operand=Side::B> __global__ void grouped_b_leaves(Size rows,Digest seed,Digest key,Digest* values,Row* meta,Word* invalid){
    static_assert(Leaves>0&&Leaves<32);
    constexpr int groups=32/Leaves;
    const int lane=threadIdx.x%32,member=lane%Leaves;
    const Size warp=(Size(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const Size row=warp*groups+lane/Leaves;
    const bool used=lane<groups*Leaves&&row<rows;
    const auto active=__ballot_sync(0xffffffffu,used);
    if(!used)return;
    const Size index=row*Leaves+member;
    Digest state=key;int sum=0;Word lane_peaks=0;
#pragma unroll 1
    for(int b=0;b<16;b++){
        Word msg[16];const Size first=index*32+2*b;
        pearl_jobgen::raw_block(seed,operand,first,msg);
        pearl_jobgen::raw_block(seed,operand,first+1,msg+8);
#pragma unroll
        for(int j=0;j<16;j++)collect_word(msg[j],sum,lane_peaks);
        const Word flags=16u|(b==0?1u:0u)|(b==15?2u:0u);
        pearl_jobgen::detail::compress<false>(state.words,msg,index,64,flags,state.words);
    }
    values[index]=state;
    Word peak=max(max(lane_peaks&255,(lane_peaks>>8)&255),max((lane_peaks>>16)&255,lane_peaks>>24));
#pragma unroll
    for(int offset=16;offset;offset/=2){
        const int next_sum=__shfl_down_sync(active,sum,offset);
        const Word next_peak=__shfl_down_sync(active,peak,offset);
        if(member+offset<Leaves){sum+=next_sum;peak=max(peak,next_peak);}
    }
    if(member==0)meta[row]=pearl_prep::row_coefficients(sum,peak,Leaves*1024,invalid);
}
template<int Leaves,Side operand=Side::B> inline void grouped_b(const pearl_jobgen::Plan& p,Digest seed,Digest key,Digest* cvs,Row* meta,Word* invalid,cudaStream_t stream){
    constexpr int rows_per_block=4*(32/Leaves);
    const int blocks=int((p.rows+rows_per_block-1)/rows_per_block);
    grouped_b_leaves<Leaves,operand><<<blocks,128,0,stream>>>(p.rows,seed,key,cvs,meta,invalid);
}
template<Side operand=Side::B> inline cudaError_t direct_b_leaves(const pearl_jobgen::Plan& p,const Digest& seed,const Digest& key,Digest* cvs,Row* meta,Word* invalid,cudaStream_t stream){
    if(!pearl_jobgen::valid_plan(p)||!cvs||!meta||!invalid||!pearl_geometry::supported_scan_depth(p.k)||p.rows>=(1u<<24)||
       (reinterpret_cast<uintptr_t>(cvs)&3)||(reinterpret_cast<uintptr_t>(meta)&3)||(reinterpret_cast<uintptr_t>(invalid)&3)||
       p.value_bytes!=p.value_chunks*1024)return cudaErrorInvalidValue;
    const unsigned blocks=unsigned((p.value_chunks+127)/128);
    if(p.k==32768)direct_b_leaves_kernel<32,operand><<<blocks,128,0,stream>>>(p.value_chunks,seed,key,cvs,meta,invalid);
    else if(p.k==8192)direct_b_leaves_kernel<8,operand><<<blocks,128,0,stream>>>(p.value_chunks,seed,key,cvs,meta,invalid);
    else if(p.k==10240)grouped_b<10,operand>(p,seed,key,cvs,meta,invalid,stream);
    else if(p.k==12288)grouped_b<12,operand>(p,seed,key,cvs,meta,invalid,stream);
    else if(p.k==13312)grouped_b<13,operand>(p,seed,key,cvs,meta,invalid,stream);
    else if(p.k==14336)grouped_b<14,operand>(p,seed,key,cvs,meta,invalid,stream);
    else if(p.k==20480)grouped_b<20,operand>(p,seed,key,cvs,meta,invalid,stream);
    else if(p.k==24576)grouped_b<24,operand>(p,seed,key,cvs,meta,invalid,stream);
    else direct_b_leaves_kernel<16,operand><<<blocks,128,0,stream>>>(p.value_chunks,seed,key,cvs,meta,invalid);
    return cudaGetLastError();
}

// CUDA packs the first FP32 into the high byte. Clamp an exceptional NaN
// encoding to the scalar encoder's saturated magnitude and preserve sign.
__device__ __forceinline__ std::uint16_t encode_pair(float lo,float hi){
    unsigned short bits;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;":"=h"(bits):"f"(hi),"f"(lo));
    const Word mag=__vminu4(Word(bits)&0x7f7fu,0x7e7eu);
    const Word sign=((__float_as_uint(lo)>>24)&0x80u)|((__float_as_uint(hi)>>16)&0x8000u);
    return std::uint16_t(mag|sign);
}
__device__ __forceinline__ std::uint16_t finish_pair(float lo,float hi,bool strip,Row p){
    const Word pair=encode_pair(lo,hi);
    const Word mag=pair&0x7f7fu;
    const Word threshold=Word(p.threshold)*0x101u;
    const Word replace=strip?__vcmpltu4(mag,threshold):0u;
    const Word changed=(pair&0x8787u)|Word(p.strip|1u)*0x101u;
    return std::uint16_t((pair&~replace)|(changed&replace));
}

__device__ __forceinline__ void shuffle_exchange(Word& low,Word& high,int delta,int lane){
    const Word value=(lane&delta)?low:high;
    const Word other=__shfl_xor_sync(0xffffffffu,value,delta,4);
    if(lane&delta)low=other;else high=other;
}
__device__ __forceinline__ void matrix_transpose(Word (&v)[4],int lane){
    shuffle_exchange(v[0],v[1],1,lane);shuffle_exchange(v[2],v[3],1,lane);
    shuffle_exchange(v[0],v[2],2,lane);shuffle_exchange(v[1],v[3],2,lane);
}
__device__ __forceinline__ void load_fragment(const Byte* base,int row,int total,int k,int col,int lane,Word (&reg)[4]){
    uint4 words{};
    const int at=col+(lane&3)*16;
    if(row<total&&at<k)words=*reinterpret_cast<const uint4*>(base+size_t(row)*k+at);
    reg[0]=__byte_perm(words.x,words.z,0x5410);reg[1]=__byte_perm(words.x,words.z,0x7632);
    reg[2]=__byte_perm(words.y,words.w,0x5410);reg[3]=__byte_perm(words.y,words.w,0x7632);
    matrix_transpose(reg,lane);
}
__device__ __forceinline__ void store_fragment(Byte* base,int row,int total,int k,int col,int lane,Word (&reg)[4]){
    matrix_transpose(reg,lane);
    const uint4 words=make_uint4(__byte_perm(reg[0],reg[1],0x5410),__byte_perm(reg[2],reg[3],0x5410),
                                 __byte_perm(reg[0],reg[1],0x7632),__byte_perm(reg[2],reg[3],0x7632));
    const int at=col+(lane&3)*16;
    if(row<total&&at<k)*reinterpret_cast<uint4*>(base+size_t(row)*k+at)=words;
}

enum class Source { raw,shared,shuffle };
// Each warp owns 16 rows x 64 columns. Every one of its lanes computes one
// independent 32-byte raw XOF block when generation and mixing are fused.
template<Source S,bool Vector=false> __global__ void mix_kernel(Byte* output,int rows,int k,Size first,Digest seed,
                                             const Byte* e,const Byte* f,const Row* params){
    constexpr int warps=4;
    __shared__ Word exchange[S==Source::shared?warps*8*32:1];
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const int base_col=blockIdx.x*(warps*64)+warp*64,row_base=blockIdx.y*16;
    if(base_col>=k)return;
    Word words[8]{};
    if constexpr(S!=Source::raw){
        const int rr=row_base+(lane&15),cc=base_col+(lane/16)*32;
        if(rr<rows&&cc<k)pearl_jobgen::raw_block(seed,Side::B,(first+rr)*(k/32)+cc/32,words);
        if constexpr(S==Source::shared){
#pragma unroll
            for(int j=0;j<8;j++)exchange[(warp*8+j)*32+lane]=words[j];
            __syncwarp();
        }
    }
    const int row=row_base+lane/4,pos=(lane&3)*4;
    const Word a0=row<rows?pearl_prep::packed4(e,row,pos):0;
    const Word a2=row<rows?pearl_prep::packed4(e,row,pos+16):0;
    const Word a1=row+8<rows?pearl_prep::packed4(e,row+8,pos):0;
    const Word a3=row+8<rows?pearl_prep::packed4(e,row+8,pos+16):0;
    Word incoming[2][4]{},outgoing[2][4]{};
    if constexpr(Vector && S==Source::raw){
#pragma unroll
        for(int part=0;part<2;part++)load_fragment(output,row+8*part,rows,k,base_col,lane,incoming[part]);
    }
    Row p[2]{};
    if(row<rows)p[0]=params[row];
    if(row+8<rows)p[1]=params[row+8];
#pragma unroll
    for(int j=0;j<8;j++){
        const int frow=base_col+8*j+lane/4;
        if(base_col+8*j>=k)break;
        const Word b0=pearl_prep::packed4(f,frow,pos),b1=pearl_prep::packed4(f,frow,pos+16);
        float v[4]={};
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     :"+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        const int col=base_col+8*j+(lane&3)*2;
#pragma unroll
        for(int part=0;part<2;part++){
            const int rr=row+8*part;size_t offset=size_t(rr)*k+col;char2 raw{};
            if constexpr(S==Source::raw){
                if constexpr(Vector){
                    const Word bits=incoming[part][j/2]>>(16*(j&1));
                    raw.x=static_cast<signed char>(bits);raw.y=static_cast<signed char>(bits>>8);
                }else if(rr<rows)raw=*reinterpret_cast<const char2*>(output+offset);
            }else{
                const int source_lane=(j/4)*16+8*part+lane/4;
                const int w=2*(j%4);Word bits;
                if constexpr(S==Source::shared){
                    bits=exchange[(warp*8+w+(lane&3)/2)*32+source_lane];
                }else{
                    const Word left=__shfl_sync(0xffffffffu,words[w],source_lane);
                    const Word right=__shfl_sync(0xffffffffu,words[w+1],source_lane);
                    bits=(lane&2)?right:left;
                }
                bits>>=(lane&1)*16;
                raw.x=static_cast<signed char>(bits);raw.y=static_cast<signed char>(bits>>8);
            }
            const float low=__fmaf_rn(v[2*part],p[part].beta,__fmul_rn(p[part].alpha,float(raw.x)));
            const float high=__fmaf_rn(v[2*part+1],p[part].beta,__fmul_rn(p[part].alpha,float(raw.y)));
            const Word first_bits=__shfl_sync(0xffffffffu,__float_as_uint(low),lane&~1);
            const std::uint16_t pair=finish_pair(low,high,((first_bits>>8)&127u)<112,p[part]);
            if constexpr(Vector)outgoing[part][j/2]|=Word(pair)<<(16*(j&1));
            else if(rr<rows)*reinterpret_cast<std::uint16_t*>(output+offset)=pair;
        }
    }
    if constexpr(Vector){
#pragma unroll
        for(int part=0;part<2;part++)store_fragment(output,row+8*part,rows,k,base_col,lane,outgoing[part]);
    }
}
inline cudaError_t prepare(Mode mode,Byte* data,int rows,int k,int side,std::uint32_t first,
        pearl_noise::Key ek,pearl_noise::Key fk,const Digest& seed,int batch,void* scratch,size_t capacity,
        const Row* meta,Word* invalid,bool build_f,cudaStream_t stream){
    const size_t need=pearl_noise::scratch_bytes(k,batch);
    if(mode==Mode::off||!need||rows<1||rows>=(1<<24)||Size(first)+rows-1>UINT32_MAX||side<0||side>1||
       !data||(reinterpret_cast<uintptr_t>(data)&3u)||!scratch||(reinterpret_cast<uintptr_t>(scratch)&15u)||
       capacity<need||!invalid||(reinterpret_cast<uintptr_t>(invalid)&3u)||(saved_rows(mode)&&(!meta||(reinterpret_cast<uintptr_t>(meta)&3u))))return cudaErrorInvalidValue;
    // An aligned allocation can be sliced on any four-byte boundary in this
    // API. Keep the scalar path for slices without uint4 alignment.
    if(reinterpret_cast<uintptr_t>(data)&15u){
        switch(mode){
            case Mode::vector:mode=Mode::packed;break;
            case Mode::sharedvec:mode=Mode::shared;break;
            case Mode::shufflevec:mode=Mode::shuffle;break;
            case Mode::leafvec:mode=Mode::leaf;break;
            default:break;
        }
    }
    Byte* f=static_cast<Byte*>(scratch),*e=f+size_t(k)*32;
    auto* temp=reinterpret_cast<Row*>(e+size_t(batch)*32);
    if(build_f){
        const auto st=pearl_noise::generate_table(fk,side,1,0,k,f,stream);if(st!=cudaSuccess)return st;
    }
    for(int local=0;local<rows;local+=batch){
        const int n=std::min(batch,rows-local);Byte* part=data+size_t(local)*k;
        auto st=pearl_noise::generate_table(ek,side,0,first+local,n,e,stream);if(st!=cudaSuccess)return st;
        const Row* r=meta?meta+local:temp;
        if(!meta){st=pearl_prep::statistics(reinterpret_cast<const std::int8_t*>(part),n,k,temp,invalid,stream);if(st!=cudaSuccess)return st;}
        const dim3 grid((k+255)/256,(n+15)/16);
        if(side==1&&(mode==Mode::sharedvec||mode==Mode::leafvec))mix_kernel<Source::shared,true><<<grid,128,0,stream>>>(part,n,k,Size(first)+local,seed,e,f,r);
        else if(side==1&&mode==Mode::shufflevec)mix_kernel<Source::shuffle,true><<<grid,128,0,stream>>>(part,n,k,Size(first)+local,seed,e,f,r);
        else if(mode==Mode::vector||mode==Mode::sharedvec||mode==Mode::shufflevec||mode==Mode::leafvec)mix_kernel<Source::raw,true><<<grid,128,0,stream>>>(part,n,k,Size(first)+local,seed,e,f,r);
        else if(side==1&&(mode==Mode::shared||mode==Mode::leaf))mix_kernel<Source::shared><<<grid,128,0,stream>>>(part,n,k,Size(first)+local,seed,e,f,r);
        else if(side==1&&mode==Mode::shuffle)mix_kernel<Source::shuffle><<<grid,128,0,stream>>>(part,n,k,Size(first)+local,seed,e,f,r);
        else mix_kernel<Source::raw><<<grid,128,0,stream>>>(part,n,k,Size(first)+local,seed,e,f,r);
        st=cudaGetLastError();if(st!=cudaSuccess)return st;
    }
    return cudaSuccess;
}
}
