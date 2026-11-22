#pragma once
#include "finite_mix.cuh"
#include <type_traits>
namespace left_pack {
using prep_inputs::Byte;using prep_inputs::Word;using prep_inputs::Row;using prep_inputs::Size;
using prep_inputs::Digest;using prep_inputs::Side;using prep_inputs::Source;
using prep_inputs::matrix_transpose;using prep_inputs::load_fragment;using finite_mix::finish_pair;
inline int selected(){
    const char* x=std::getenv("PEARL_LEFT_WIDTH");
    if(!x||!std::strcmp(x,"row"))return 0;
    for(int w:{128,256,512,1024})if(std::to_string(w)==x)return w;
    throw std::invalid_argument("invalid left operand physical width");
}
inline bool staged(){
    const char* x=std::getenv("PEARL_LEFT_SOURCE");
    if(!x||!std::strcmp(x,"stage"))return true;
    if(!std::strcmp(x,"xof"))return false;
    throw std::invalid_argument("invalid left operand source");
}
template<int W>__host__ __device__ inline size_t offset(int row,int c,int rows,int k){
    const int first=row&~255,local=row&255,start=c&~(W-1);
    const int active_rows=rows-first<256?rows-first:256;
    const int active_k=k-start<W?k-start:W;
    return size_t(first)*k+size_t(start)*active_rows+size_t(local)*active_k+(c&(W-1));
}
inline bool disjoint(const void* a,size_t an,const void* b,size_t bn){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x>=an:x-y>=bn;
}

template<int W>__device__ __forceinline__ void store_packed_fragment(Byte* base,int row,int total,int k,int col,int lane,Word (&reg)[4],int panel_rows,int local){
    matrix_transpose(reg,lane);
    const uint4 words=make_uint4(__byte_perm(reg[0],reg[1],0x5410),__byte_perm(reg[2],reg[3],0x5410),
                                 __byte_perm(reg[0],reg[1],0x7632),__byte_perm(reg[2],reg[3],0x7632));
    const int at=col+(lane&3)*16;
    if(row<total&&at<k)*reinterpret_cast<uint4*>(base+offset<W>(local+row,at,panel_rows,k))=words;
}

template<int W,bool Staged,bool Vector> __global__ void mix(Byte* output,const Byte* raw_source,int rows,int k,Size first,Digest seed,
                                             const Byte* e,const Byte* f,const Row* params,int panel_rows,int local){
    constexpr Source S=Staged?Source::raw:Source::shared;
    constexpr int warps=4;
    __shared__ Word exchange[S==Source::shared?warps*8*32:1];
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const int base_col=blockIdx.x*(warps*64)+warp*64,row_base=blockIdx.y*16;
    if(base_col>=k)return;
    Word words[8]{};
    if constexpr(S!=Source::raw){
        const int rr=row_base+(lane&15),cc=base_col+(lane/16)*32;
        if(rr<rows&&cc<k)pearl_jobgen::raw_block(seed,Side::A,(first+rr)*(k/32)+cc/32,words);
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
        for(int part=0;part<2;part++)load_fragment(raw_source,row+8*part,rows,k,base_col,lane,incoming[part]);
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
                }else if(rr<rows)raw=*reinterpret_cast<const char2*>(raw_source+offset);
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
            else if(rr<rows)*reinterpret_cast<std::uint16_t*>(output+left_pack::offset<W>(local+rr,col,panel_rows,k))=pair;
        }
    }
    if constexpr(Vector){
#pragma unroll
        for(int part=0;part<2;part++)store_packed_fragment<W>(output,row+8*part,rows,k,base_col,lane,outgoing[part],panel_rows,local);
    }
}
template<int W,bool Staged>inline cudaError_t prepare_impl(Byte* output,int rows,int k,std::uint32_t first,
        const pearl_noise::Key& ek,const pearl_noise::Key& fk,const Digest& seed,int batch,
        Byte* source,size_t source_bytes,Byte* scratch,const Row* meta,Word* invalid,cudaStream_t stream,bool build_f,bool source_ready=false){
    auto* f=scratch;auto* e=f+size_t(k)*32;
    cudaError_t err=cudaSuccess;
    if(build_f){err=pearl_noise::generate_table(fk,0,1,0,k,f,stream);if(err!=cudaSuccess)return err;}
    const bool wide=(reinterpret_cast<uintptr_t>(output)&15u)==0&&(!Staged||(reinterpret_cast<uintptr_t>(source)&15u)==0);
    int slab_max=rows;
    if constexpr(Staged){
        slab_max=std::min<size_t>(4096,source_bytes/k);
        if(slab_max<rows)slab_max=slab_max/256*256;
    }
    for(int outer=0;outer<rows;outer+=slab_max){
        const int slab=std::min(rows-outer,slab_max);
        if constexpr(Staged){
            if(!source_ready){
                err=cudaMemcpyAsync(source,output+size_t(outer)*k,size_t(slab)*k,cudaMemcpyDeviceToDevice,stream);
                if(err!=cudaSuccess)return err;
            }
        }
        for(int j=0;j<slab;j+=batch){
            const int count=std::min(batch,slab-j),local=outer+j;
            err=pearl_noise::generate_table(ek,0,0,first+local,count,e,stream);
            if(err!=cudaSuccess)return err;
            const Byte* raw=nullptr;
            if constexpr(Staged)raw=source+size_t(source_ready?local:j)*k;
            const dim3 grid((k+255)/256,(count+15)/16);
            if(wide)mix<W,Staged,true><<<grid,128,0,stream>>>(output,raw,count,k,Size(first)+local,seed,e,f,meta+local,rows,local);
            else mix<W,Staged,false><<<grid,128,0,stream>>>(output,raw,count,k,Size(first)+local,seed,e,f,meta+local,rows,local);
            err=cudaGetLastError();if(err!=cudaSuccess)return err;
        }
    }
    return cudaSuccess;
}
inline cudaError_t prepare(int width,bool use_stage,Byte* output,size_t output_bytes,int rows,int k,std::uint32_t first,
        const pearl_noise::Key& ek,const pearl_noise::Key& fk,const Digest& seed,int batch,Byte* stage,size_t stage_bytes,
        void* scratch,size_t capacity,const Row* meta,Word* invalid,cudaStream_t stream=nullptr,bool build_f=true,bool source_ready=false){
    const auto need=pearl_noise::scratch_bytes(k,batch);
    if(!need||rows<1||rows>=(1<<24)||Size(first)+rows-1>UINT32_MAX||!output||reinterpret_cast<uintptr_t>(output)%4||
       size_t(rows)*k>output_bytes||!scratch||reinterpret_cast<uintptr_t>(scratch)%16||capacity<need||
       !meta||reinterpret_cast<uintptr_t>(meta)%4||!invalid||reinterpret_cast<uintptr_t>(invalid)%4)return cudaErrorInvalidValue;
    const size_t size=size_t(rows)*k,metadata=size_t(rows)*sizeof(Row);
    if(!disjoint(output,size,scratch,need)||!disjoint(output,size,meta,metadata)||!disjoint(output,size,invalid,4)||
       !disjoint(scratch,need,meta,metadata)||!disjoint(scratch,need,invalid,4)||!disjoint(meta,metadata,invalid,4))return cudaErrorInvalidValue;
    if(use_stage){
        const size_t usable=std::min(stage_bytes,size_t(4096)*k);
        if(!stage||reinterpret_cast<uintptr_t>(stage)%4||usable<size_t(std::min(rows,256))*k||
           !disjoint(stage,usable,output,size)||!disjoint(stage,usable,scratch,need)||
           !disjoint(stage,usable,meta,metadata)||!disjoint(stage,usable,invalid,4))return cudaErrorInvalidValue;
    }
    auto issue=[&](auto w){
        constexpr int W=decltype(w)::value;
        if(use_stage)return prepare_impl<W,true>(output,rows,k,first,ek,fk,seed,batch,stage,stage_bytes,static_cast<Byte*>(scratch),meta,invalid,stream,build_f,source_ready);
        return prepare_impl<W,false>(output,rows,k,first,ek,fk,seed,batch,nullptr,0,static_cast<Byte*>(scratch),meta,invalid,stream,build_f);
    };
    switch(width){case 128:return issue(std::integral_constant<int,128>{});case 256:return issue(std::integral_constant<int,256>{});
        case 512:return issue(std::integral_constant<int,512>{});case 1024:return issue(std::integral_constant<int,1024>{});default:return cudaErrorInvalidValue;}
}
}
