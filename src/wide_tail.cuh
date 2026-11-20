#pragma once
#include "scan_depth.hpp"
#include "sm120_primitives.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace wide_tail {
using fp8_sm120::shared_address, fp8_sm120::initialize_barrier, fp8_sm120::wait_barrier, fp8_sm120::release_stage;
using fp8_sm120::multiply32, fp8_sm120::canonical_bits, fp8_sm120::fold_step, fp8_sm120::transpose;
using fp8_sm120::check_cuda, fp8_sm120::check_driver, fp8_sm120::Driver;
inline constexpr char name[]="sm120-fused-stream";
using Byte=std::uint8_t;
using Word=std::uint32_t;
constexpr int rows=4096,cols=16384,tile_m=256,tile_n=192,bk=64,stages=2,stage_bytes=(tile_m+tile_n)*bk;

__device__ __forceinline__ void load_stage(const CUtensorMap& a,const CUtensorMap& b,Byte* stage,
                                          int k0,int row,int col,std::uint64_t* ready){
    const Word barrier=shared_address(ready);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(barrier),"r"(stage_bytes):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                 ::"r"(shared_address(stage)),"l"(&a),"r"(k0),"r"(row),"r"(barrier):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                 ::"r"(shared_address(stage+tile_m*bk)),"l"(&b),"r"(k0),"r"(col),"r"(barrier):"memory");
}
__device__ __forceinline__ int swizzled(int row,int k){
    const int offset=row*bk+k;
    return offset^((offset&0x180)>>3);
}
__device__ __forceinline__ void read_a(const Byte* base,int row,int k,Word (&a)[4]){
    const int lane=threadIdx.x&31;
    const Word addr=shared_address(base+swizzled(row+(lane&15),k+(lane>>4)*16));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 :"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(addr));
}
__device__ __forceinline__ void read_b_pair(const Byte* base,int row,int k,Word (*b)[2]){
    const int lane=threadIdx.x&31;
    // The low two matrices supply B rows [row,row+8); the upper two supply [row+8,row+16).
    const Word addr=shared_address(base+swizzled(row+(lane&7)+8*(lane>>4),k+((lane>>3)&1)*16));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 :"=r"(b[0][0]),"=r"(b[0][1]),"=r"(b[1][0]),"=r"(b[1][1]):"r"(addr));
}
__device__ __forceinline__ void retire_stage(const CUtensorMap& a,const CUtensorMap& b,Byte* ring,
        std::uint64_t* ready,std::uint64_t* empty,int slot,int epoch,int step,int steps,int row0,int col0){
    __syncwarp();
    if((threadIdx.x&31)==0)release_stage(empty+slot);
    if(threadIdx.x==0&&step+stages<steps){
        wait_barrier(empty+slot,epoch);
        load_stage(a,b,ring+slot*stage_bytes,(step+stages)*bk,row0,col0,ready+slot);
    }
}

__global__ __launch_bounds__(256,1) void scan_window(const __grid_constant__ CUtensorMap a,
        const __grid_constant__ CUtensorMap b,Word* output,int m,int n,int k,int fold_cols){
    extern __shared__ __align__(1024) Byte ring[];
    __shared__ __align__(8) std::uint64_t ready[stages],empty[stages];
    const int tid=threadIdx.x,warp=tid/32,lane=tid&31;
    if(tid<stages){initialize_barrier(ready+tid,1);initialize_barrier(empty+tid,8);}
    asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();

    // Visit four neighboring M tiles per column. The final N tile can extend
    // past the input, where a tensor-map load fills the missing rows with zero.
    constexpr int strip_height=4;
    const int grid_m=(m+tile_m-1)/tile_m,grid_n=(n+tile_n-1)/tile_n;
    const int id=int(blockIdx.x),first=(id/(strip_height*grid_n))*strip_height;
    const int local=id-first*grid_n,height=grid_m-first<strip_height?grid_m-first:strip_height;
    const int row0=(first+local%height)*tile_m,col0=(local/height)*tile_n;
    const int steps=k/bk;
    if(tid==0){
#pragma unroll
        for(int s=0;s<stages;s++)if(s<steps)load_stage(a,b,ring+s*stage_bytes,s*bk,row0,col0,ready+s);
    }
    const int warp_row=(warp/2)*64,warp_col=(warp%2)*96;
    float accum[4][12][4]={};
    int slot=0,epoch=0;
    for(int step=0;step<steps;step++){
        wait_barrier(ready+slot,epoch);
        const Byte* sa=ring+slot*stage_bytes;const Byte* sb=sa+tile_m*bk;
#pragma unroll
        for(int q=0;q<2;q++){
            Word av[4][4];
#pragma unroll
            for(int i=0;i<4;i++)read_a(sa,warp_row+16*i,32*q,av[i]);
#pragma unroll
            for(int j=0;j<12;j+=2){
                Word bp[2][2];read_b_pair(sb,warp_col+8*j,32*q,bp);
                // All eight warps must finish reading the stage before it can be refilled.
                if(q==1&&j==10)retire_stage(a,b,ring,ready,empty,slot,epoch,step,steps,row0,col0);
#pragma unroll
                for(int i=0;i<4;i++){
                    multiply32(accum[i][j],av[i],bp[0]);
                    multiply32(accum[i][j+1],av[i],bp[1]);
                }
            }
        }
        if(++slot==stages){slot=0;epoch^=1;}
    }
    const int half=(lane>>1)&1,segment=lane&1;
#pragma unroll
    for(int i=0;i<4;i++){
        const int row=row0+warp_row+16*i+lane/4+8*half;
#pragma unroll
        for(int group=0;group<2;group++){
            // A 96-column warp yields three 32-column tickets. Both lane
            // segments participate in the shuffle for the last ticket, and
            // the even segment owns its only write.
            const bool last=group==1;
            const int col=col0+warp_col+64*group+(last?0:32*segment);
            Word word=0;
#pragma unroll
            for(int q=0;q<4;q++){
                Word lo[4],hi[4];
#pragma unroll
                for(int t=0;t<4;t++){
                    const int frag=8*group+(last?0:4*(t&1))+q,part=2*(t/2);
                    lo[t]=canonical_bits(accum[i][frag][part]);hi[t]=canonical_bits(accum[i][frag][part+1]);
                }
                transpose(lo,lane);transpose(hi,lane);
#pragma unroll
                for(int t=0;t<4;t++){word=fold_step(word,lo[t]);word=fold_step(word,hi[t]);}
            }
            if((!last||segment==0)&&(row/16+1)*16<=m&&col+32<=n)
                output[(size_t(row/16)*fold_cols+col/32)*16+row%16]=word;
        }
    }
}

struct Descriptor {
    CUtensorMap map{};const Byte* previous=nullptr;int old_count=0;
    void bind(const Driver& api,const Byte* data,int count,int k,int box_rows){
        if(!previous||old_count!=count){
            cuuint64_t size[2]={cuuint64_t(k),cuuint64_t(count)},stride[1]={cuuint64_t(k)};
            cuuint32_t box[2]={bk,static_cast<cuuint32_t>(box_rows)},increments[2]={1,1};
            check_driver(api.encode(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,const_cast<Byte*>(data),size,stride,box,increments,
                CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"encode streamed tensor map");
        }else if(previous!=data){
            check_driver(api.replace(&map,const_cast<Byte*>(data)),"update streamed tensor map address");
        }
        previous=data;old_count=count;
    }
};
class Plan {
    Descriptor left_,right_;
    int k_=0;std::size_t calls_=0;
public:
    std::size_t calls()const{return calls_;}
    void run(const Byte* a,const Byte* b,Word* output,int k,int m,int width,int ld,cudaStream_t stream){
        if(!a||!b||!output||!pearl_geometry::supported_scan_depth(k))throw std::invalid_argument("unsupported fused stream input");
        static const Driver api;
        if(!k_){
            check_cuda(cudaFuncSetAttribute(scan_window,cudaFuncAttributeMaxDynamicSharedMemorySize,stages*stage_bytes),"configure fused stream kernel");
            k_=k;
        }else if(k!=k_)throw std::logic_error("fused stream plan reused for a different job depth");
        left_.bind(api,a,m,k,tile_m);right_.bind(api,b,width,k,tile_n);
        const int blocks=(m/tile_m)*((width+tile_n-1)/tile_n);
        scan_window<<<blocks,256,stages*stage_bytes,stream>>>(left_.map,right_.map,output,m,width,k,ld);
        check_cuda(cudaGetLastError(),"launch fused stream kernel");++calls_;
    }
};
}
