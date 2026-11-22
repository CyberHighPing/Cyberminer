#pragma once
#include "finite_mix.cuh"
#include "directpack_layout.hpp"
namespace leaf_packed {
using prep_inputs::Byte;using prep_inputs::Word;using prep_inputs::Row;using prep_inputs::Size;
using prep_inputs::Digest;using prep_inputs::Side;using prep_inputs::Source;
using prep_inputs::matrix_transpose;using prep_inputs::load_fragment;using finite_mix::finish_pair;
inline bool selected(){
    const char* x=std::getenv("PEARL_LEAF_B_LAYOUT");
    if(!x||std::strcmp(x,"row")==0)return false;
    if(std::strcmp(x,"tma256")==0)return true;
    throw std::invalid_argument("PEARL_LEAF_B_LAYOUT must be row or tma256");
}
__device__ __forceinline__ void store_packed_fragment(Byte* base,int row,int total,int k,int col,int lane,Word (&reg)[4],int panel_rows,int local,int split){
    matrix_transpose(reg,lane);
    const uint4 words=make_uint4(__byte_perm(reg[0],reg[1],0x5410),__byte_perm(reg[2],reg[3],0x5410),
                                 __byte_perm(reg[0],reg[1],0x7632),__byte_perm(reg[2],reg[3],0x7632));
    const int at=col+(lane&3)*16;
    if(row<total&&at<k)*reinterpret_cast<uint4*>(base+directpack::offset<directpack::Mode::tma256>(local+row,at,panel_rows,k,split))=words;
}

#ifdef PEARL_SCAN_INPUTS
template<bool Vector,bool Prefix> __device__ __forceinline__ void quantize(Byte* output,int rows,int k,Size first,Digest seed,const pearl_jobgen::RawPrefix& prefix,
                                             const Byte* e,const Byte* f,const Row* params,int panel_rows,int local,int split){
    constexpr Source S=Source::shared;
    constexpr int warps=4;
    __shared__ Word exchange[S==Source::shared?warps*8*32:1];
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const int base_col=blockIdx.x*(warps*64)+warp*64,row_base=blockIdx.y*16;
    if(base_col>=k)return;
    Word words[8]{};
    if constexpr(S!=Source::raw){
        const int rr=row_base+(lane&15),cc=base_col+(lane/16)*32;
        if(rr<rows&&cc<k){
            const Size index=(first+rr)*(k/32)+cc/32;
            if constexpr(Prefix)pearl_jobgen::raw_from_matching_prefix(seed,Side::B,prefix,index,words);
            else pearl_jobgen::raw_block(seed,Side::B,index,words);
        }
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
            else if(rr<rows)*reinterpret_cast<std::uint16_t*>(output+directpack::offset<directpack::Mode::tma256>(local+rr,col,panel_rows,k,split))=pair;
        }
    }
    if constexpr(Vector){
#pragma unroll
        for(int part=0;part<2;part++)store_packed_fragment(output,row+8*part,rows,k,base_col,lane,outgoing[part],panel_rows,local,split);
    }
}
template<bool Vector> __global__ void mix(Byte* output,int rows,int k,Size first,Digest seed,
                                             const Byte* e,const Byte* f,const Row* params,int panel_rows,int local,int split){
    quantize<Vector,false>(output,rows,k,first,seed,{},e,f,params,panel_rows,local,split);
}
__global__ void prefix_mix(Byte* output,int rows,int k,Size first,Digest seed,pearl_jobgen::RawPrefix prefix,
                                             const Byte* e,const Byte* f,const Row* params,int panel_rows,int local,int split){
    quantize<true,true>(output,rows,k,first,seed,prefix,e,f,params,panel_rows,local,split);
}
inline bool prefix_enabled(int k){return k==13312 || k==16384;}

inline bool one_counter_high(Size first,int rows,int k){
    const Size blocks=unsigned(k)/32;
    return ((first*blocks)>>32)==(((first+unsigned(rows))*blocks-1)>>32);
}
inline int prefix_batches(int first,int rows,int k,int batch){
    int out=0;
    for(int i=0;i<rows;i+=batch)out+=one_counter_high(Size(first)+i,std::min(batch,rows-i),k);
    return out;
}
#else
template<bool Vector> __global__ void mix(Byte* output,int rows,int k,Size first,Digest seed,
                                             const Byte* e,const Byte* f,const Row* params,int panel_rows,int local,int split){
    constexpr Source S=Source::shared;
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
            else if(rr<rows)*reinterpret_cast<std::uint16_t*>(output+directpack::offset<directpack::Mode::tma256>(local+rr,col,panel_rows,k,split))=pair;
        }
    }
    if constexpr(Vector){
#pragma unroll
        for(int part=0;part<2;part++)store_packed_fragment(output,row+8*part,rows,k,base_col,lane,outgoing[part],panel_rows,local,split);
    }
}
#endif
inline cudaError_t prepare(Byte* output,int rows,int k, std::uint32_t first,int split,
        pearl_noise::Key ek,pearl_noise::Key fk,const Digest& seed,int batch,void* scratch,size_t capacity,
        const Row* meta,Word* invalid,bool build_f,cudaStream_t stream){
    const size_t need=pearl_noise::scratch_bytes(k,batch);
    if(!need||rows<1||rows>=(1<<24)||split<0||split>rows||Size(first)+rows-1>UINT32_MAX||
       !output||(reinterpret_cast<uintptr_t>(output)&3u)||!scratch||(reinterpret_cast<uintptr_t>(scratch)&15u)||
       capacity<need||!meta||(reinterpret_cast<uintptr_t>(meta)&3u)||!invalid||(reinterpret_cast<uintptr_t>(invalid)&3u))
        return cudaErrorInvalidValue;
    const bool wide=(reinterpret_cast<uintptr_t>(output)&15u)==0;
#ifdef PEARL_SCAN_INPUTS
    const bool use_prefix=wide&&prefix_enabled(k);
    const auto prefix=use_prefix?pearl_jobgen::raw_prefix(seed,Side::B):pearl_jobgen::RawPrefix{};
#endif
    Byte* f=static_cast<Byte*>(scratch),*e=f+size_t(k)*32;
    if(build_f){
        const auto s=pearl_noise::generate_table(fk,1,1,0,k,f,stream);
        if(s!=cudaSuccess)return s;
    }
    for(int local=0;local<rows;local+=batch){
        const int count=std::min(batch,rows-local);
        auto s=pearl_noise::generate_table(ek,1,0,first+local,count,e,stream);
        if(s!=cudaSuccess)return s;
        const dim3 grid((k+255)/256,(count+15)/16);
#ifdef PEARL_SCAN_INPUTS
        if(use_prefix && one_counter_high(Size(first)+local,count,k)){
            const Word high=Word(((Size(first)+local)*(unsigned(k)/32))>>32);
            const auto selected=high==prefix.counter_high?prefix:pearl_jobgen::raw_prefix(seed,Side::B,high);
            prefix_mix<<<grid,128,0,stream>>>(output,count,k,Size(first)+local,seed,selected,e,f,meta+local,rows,local,split);
        }else
#endif
        if(wide)mix<true><<<grid,128,0,stream>>>(output,count,k,Size(first)+local,seed,e,f,meta+local,rows,local,split);
        else mix<false><<<grid,128,0,stream>>>(output,count,k,Size(first)+local,seed,e,f,meta+local,rows,local,split);
        s=cudaGetLastError();if(s!=cudaSuccess)return s;
    }
    return cudaSuccess;
}
}
