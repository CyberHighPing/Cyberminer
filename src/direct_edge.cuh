#pragma once
#include "scan_depth.hpp"
#include "sm120_primitives.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace direct_edge {
using fp8_sm120::shared_address, fp8_sm120::initialize_barrier, fp8_sm120::wait_barrier, fp8_sm120::release_stage;
using fp8_sm120::multiply32, fp8_sm120::canonical_bits, fp8_sm120::fold_step, fp8_sm120::transpose;
using fp8_sm120::check_cuda, fp8_sm120::check_driver, fp8_sm120::Driver;
using fp8_sm120::output_bits;
inline constexpr char name[]="sm120-fused-stream";
using Byte=std::uint8_t;
using Word=std::uint32_t;
constexpr int rows=4096,cols=16384;
constexpr int block_rows=256,block_cols=128;
constexpr int depth_per_stage=128,stage_count=2;
constexpr int shared_bytes=stage_count*(block_rows+block_cols)*depth_per_stage;


enum class Raster {strip,footprint};
struct TilePosition {int row,col;};
// Both orders group up to 1024 input rows in one band.
template<int M,int N> __host__ __device__ inline TilePosition position(int id,int gm,int gn,Raster raster){
    constexpr int height=1024/M,width=512/N;
    if(gm==4096/M&&gn==16384/N){
        constexpr int native_cols=16384/N;
        const int y=(id/(height*native_cols))*height,within=id%(height*native_cols);
        if(raster==Raster::strip)return {y+within%height,within/height};
        return {y+(within/width)%height,(within/(height*width))*width+within%width};
    }
    const int y=(id/(height*gn))*height,within=id-y*gn;
    const int h=gm-y<height?gm-y:height;
    if(raster==Raster::strip)return {y+within%h,within/h};
    const int x=(within/(h*width))*width;
    const int w=gn-x<width?gn-x:width,rem=within-x*h;
    return {y+rem/w,x+rem%w};
}

__device__ __forceinline__ int addr128(int r,int k){int p=r*128+k;return p^((p&0x380)>>3);}
__device__ __forceinline__ void load_a128(const Byte* s,int r,int k,Word (&d)[4]){
    const int lane=threadIdx.x&31;const Word a=shared_address(s+addr128(r+(lane&15),k+16*(lane>>4)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]):"r"(a));
}
__device__ __forceinline__ void load_b128(const Byte* s,int r,int k,Word (*d)[2]){
    const int lane=threadIdx.x&31;const Word a=shared_address(s+addr128(r+(lane&7)+8*(lane>>4),k+16*((lane>>3)&1)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(d[0][0]),"=r"(d[0][1]),"=r"(d[1][0]),"=r"(d[1][1]):"r"(a));
}
template<int W,int R,int BK> __device__ __forceinline__ void packed_b(const CUtensorMap& map,const Byte* base,Byte* dst,
                                                                       int k0,int first,int full_k,Word bar){
    if constexpr(W==-1){
        const Byte* physical=base+size_t(first)*full_k+size_t(k0)*R;
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"
                     ::"r"(shared_address(dst)),"l"(physical),"r"(R*BK),"r"(bar):"memory");
    }else{
        if constexpr(W==0){
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                         ::"r"(shared_address(dst)),"l"(&map),"r"(k0),"r"(first),"r"(bar):"memory");
        }else{
        static_assert(W>=BK && W%BK==0);
        const int x=k0%W,y=first*(full_k/W)+(k0/W)*R;
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                     ::"r"(shared_address(dst)),"l"(&map),"r"(x),"r"(y),"r"(bar):"memory");
        }
    }
}
template<int M,int N,int L,int W>
__device__ __forceinline__ void fetch_pair(const CUtensorMap& a,const CUtensorMap& b,Byte* dst,
                                          int k0,int row,int col,std::uint64_t* ready,const Byte* bp,int full_k){
    constexpr int stage_bytes=(M+N)*128;
    const Word bar=shared_address(ready);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(stage_bytes):"memory");
    packed_b<L,256,128>(a,nullptr,dst,k0,row,full_k,bar);
    packed_b<W,128,128>(b,bp,dst+M*128,k0,col,full_k,bar);
}

template<int M,int N,int L,int W,bool Finite=false>
__global__ __launch_bounds__(256,1) void scan_window(const __grid_constant__ CUtensorMap a,
            const __grid_constant__ CUtensorMap b,Word* output,int m,int n,int k,int fold_cols,Raster raster,const Byte* bp){
    static_assert(M==block_rows&&N==block_cols);
    constexpr int bk=depth_per_stage,stages=stage_count,stage_bytes=(M+N)*bk;
    extern __shared__ __align__(1024) Byte ring[];
    __shared__ __align__(8) std::uint64_t ready[stages],empty[stages];
    const int tid=threadIdx.x,warp=tid/32,lane=tid&31;
    if(tid<stages){initialize_barrier(ready+tid,1);initialize_barrier(empty+tid,8);}
    asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();

    const auto coord=position<M,N>(int(blockIdx.x),(m+M-1)/M,(n+N-1)/N,raster);
    const int row0=coord.row*M,col0=coord.col*N;
    const int steps=k/bk;
    if(tid==0){
#pragma unroll
        for(int s=0;s<stages;s++)if(s<steps)fetch_pair<M,N,L,W>(a,b,ring+s*stage_bytes,s*bk,row0,col0,ready+s,bp,k);
    }
    const int warp_row=(warp/(N/64))*64,warp_col=(warp%(N/64))*64;
    float accum[4][8][4]={};
    int slot=0,phase=0;
    for(int step=0;step<steps;step++){
        wait_barrier(ready+slot,phase);
        const Byte* sa=ring+slot*stage_bytes;const Byte* sb=sa+M*bk;
        auto recycle=[&](){
            __syncwarp();if(lane==0)release_stage(empty+slot);
            if(tid==0&&step+stages<steps){wait_barrier(empty+slot,phase);fetch_pair<M,N,L,W>(a,b,ring+slot*stage_bytes,(step+stages)*bk,row0,col0,ready+slot,bp,k);}
        };
        // Load each consecutive pair of K32 fragments before its ordered accumulation.
        constexpr int Q=2;
#pragma unroll
        for(int sub=0;sub<4;sub+=Q){
            Word av[Q][4][4],bv[Q][8][2];
#pragma unroll
            for(int j=0;j<Q;j++){
#pragma unroll
                for(int i=0;i<4;i++)load_a128(sa,warp_row+16*i,32*(sub+j),av[j][i]);
#pragma unroll
                for(int n=0;n<8;n+=2)load_b128(sb,warp_col+8*n,32*(sub+j),&bv[j][n]);
            }
            if(sub+Q==4)recycle();
#pragma unroll
            for(int j=0;j<Q;j++){
#pragma unroll
                for(int i=0;i<4;i++){
#pragma unroll
                    for(int n=0;n<8;n++)multiply32(accum[i][n],av[j][i],bv[j][n]);
                }
            }
        }
        if(++slot==stages){slot=0;phase^=1;}
    }
    const int half=(lane>>1)&1,segment=lane&1;
#pragma unroll
    for(int i=0;i<4;i++){
        const int row=row0+warp_row+16*i+lane/4+8*half,col=col0+warp_col+32*segment;
        Word word=0;
#pragma unroll
        for(int q=0;q<4;q++){
            Word lo[4],hi[4];
#pragma unroll
            for(int t=0;t<4;t++){
                const int pair=4*(t&1)+q,part=2*(t/2);
                lo[t]=output_bits<Finite>(accum[i][pair][part]);hi[t]=output_bits<Finite>(accum[i][pair][part+1]);
            }
            transpose(lo,lane);transpose(hi,lane);
#pragma unroll
            for(int t=0;t<4;t++){word=fold_step(word,lo[t]);word=fold_step(word,hi[t]);}
        }
        if((row/16+1)*16<=m&&col+32<=n)output[(size_t(row/16)*fold_cols+col/32)*16+row%16]=word;
    }
}

struct Descriptor {
    CUtensorMap map{};const Byte* previous=nullptr;int old_count=0,old_k=0,old_width=0;
    void bind(const Driver& api,const Byte* data,int count,int k,int box_rows,int physical_width){
        if(!previous||old_count!=count||old_k!=k||old_width!=physical_width){
            cuuint64_t size[2]={cuuint64_t(physical_width),cuuint64_t(count)*(k/physical_width)},stride[1]={cuuint64_t(physical_width)};
            cuuint32_t box[2]={depth_per_stage,cuuint32_t(box_rows)},increments[2]={1,1};
            check_driver(api.encode(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,const_cast<Byte*>(data),size,stride,box,increments,
                CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"encode streamed tensor map");
        }else if(previous!=data){
            check_driver(api.replace(&map,const_cast<Byte*>(data)),"update streamed tensor map address");
        }
        previous=data;old_count=count;old_k=k;old_width=physical_width;
    }
};
class Plan {
    Descriptor left_,right_;
    int k_=0;std::size_t calls_=0;
public:
    std::size_t calls()const{return calls_;}
    template<int L,int W,bool Finite=false> void run(const Byte* a,const Byte* b,Word* output,int k,int m,int width,int ld,cudaStream_t stream){
        if(!a||!b||!output||!pearl_geometry::supported_scan_depth(k))throw std::invalid_argument("unsupported fused stream input");
        static const Driver api;
        if(!k_){
            check_cuda(cudaFuncSetAttribute(scan_window<block_rows,block_cols,L,W,Finite>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes),"configure fused stream kernel");
            k_=k;
        }else if(k!=k_)throw std::logic_error("fused stream plan reused for a different job depth");
        left_.bind(api,a,m,k,block_rows,L?L:k);right_.bind(api,b,width,k,block_cols,W?W:k);
        const int blocks=(m/block_rows)*(width/block_cols);
        scan_window<block_rows,block_cols,L,W,Finite><<<blocks,256,shared_bytes,stream>>>(left_.map,right_.map,output,m,width,k,ld,Raster::strip,b);
        check_cuda(cudaGetLastError(),"launch fused stream kernel");++calls_;
    }
};
}
