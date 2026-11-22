#include <cstdlib>
#include <cstring>
#pragma once
#include "scan_depth.hpp"
#include "sm120_primitives.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace direct_main {
using fp8_sm120::shared_address, fp8_sm120::initialize_barrier, fp8_sm120::wait_barrier, fp8_sm120::release_stage;
using fp8_sm120::multiply32, fp8_sm120::canonical_bits, fp8_sm120::fold_step, fp8_sm120::transpose;
using fp8_sm120::check_cuda, fp8_sm120::check_driver, fp8_sm120::Driver;
using fp8_sm120::output_bits;
inline constexpr char name[]="sm120-fused-stream";
using Byte=std::uint8_t;
using Word=std::uint32_t;
constexpr int rows=4096,cols=16384,tile_m=256,tile_n=192,bk=64,stages=2,stage_bytes=(tile_m+tile_n)*bk;

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
template<int L,int W> __device__ __forceinline__ void load_stage(const CUtensorMap& a,const CUtensorMap& b,Byte* stage,
                                          int k0,int row,int col,std::uint64_t* ready,const Byte* bp,int full_k){
    const Word barrier=shared_address(ready);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"(barrier),"r"(stage_bytes):"memory");
    packed_b<L,256,64>(a,nullptr,stage,k0,row,full_k,barrier);
    packed_b<W,192,64>(b,bp,stage+tile_m*bk,k0,col,full_k,barrier);
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
template<int L,int W> __device__ __forceinline__ void retire_stage(const CUtensorMap& a,const CUtensorMap& b,Byte* ring,
        std::uint64_t* ready,std::uint64_t* empty,int slot,int epoch,int step,int steps,int row0,int col0,const Byte* bp,int full_k){
    __syncwarp();
    if((threadIdx.x&31)==0)release_stage(empty+slot);
    if(threadIdx.x==0&&step+stages<steps){
        wait_barrier(empty+slot,epoch);
        load_stage<L,W>(a,b,ring+slot*stage_bytes,(step+stages)*bk,row0,col0,ready+slot,bp,full_k);
    }
}

template<bool Active>
__device__ __forceinline__ void scaled_multiply32(float (&d)[4],const Word (&a)[4],const Word (&b)[2]){
    if constexpr(!Active){
        multiply32(d,a,b);
    }else{
        constexpr Word alpha=0xfeu, beta=0x0u;
        constexpr unsigned short zero=0,select_a=0,select_b=0;
        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%18,%15},{%16},{%17,%15};"
                     :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
                     :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
                      "f"(d[0]),"f"(d[1]),"f"(d[2]),"f"(d[3]),"r"(alpha),"h"(zero),"r"(beta),"h"(select_b),"h"(select_a));
    }
}

template<int L,int W,int Order=0,int Fold=0,bool PairColumns=false> __global__ __launch_bounds__(256,1) void scan_window(const __grid_constant__ CUtensorMap a,
        const __grid_constant__ CUtensorMap b,Word* output,int m,int n,int k,int fold_cols,const Byte* panel_b){
    extern __shared__ __align__(1024) Byte ring[];
    __shared__ __align__(8) std::uint64_t ready[stages],empty[stages];
    const int tid=threadIdx.x,warp=tid/32,lane=tid&31;
    if(tid<stages){initialize_barrier(ready+tid,1);initialize_barrier(empty+tid,8);}
    asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();

    int row0,col0;
    if constexpr(Order==0){
    constexpr int strip_height=4;
    const int grid_m=(m+tile_m-1)/tile_m,grid_n=(n+tile_n-1)/tile_n;
    const int id=int(blockIdx.x),first=(id/(strip_height*grid_n))*strip_height;
    const int local=id-first*grid_n,height=grid_m-first<strip_height?grid_m-first:strip_height;
    row0=(first+local%height)*tile_m;col0=(local/height)*tile_n;
    }else if constexpr(Order==1){
        int row=blockIdx.x,col=blockIdx.y;
        if constexpr(PairColumns){
            // In each adjacent column pair, issue even and odd columns in turn.
            // An unpaired final column keeps its original row order.
            if((col|1)<int(gridDim.y)){
                const int index=(col&1)*int(gridDim.x)+row;
                row=index>>1;col=(col&~1)+(index&1);
            }
        }
        row0=row*tile_m;col0=col*tile_n;
    }else{
        const int group_m=m/tile_m;
        const int group_n=int(gridDim.x)/group_m;
        const int in_m=int(blockIdx.x)%group_m;
        const int in_n=int(blockIdx.x)/group_m;
        row0=in_m*tile_m;col0=(int(blockIdx.y)*group_n+in_n)*tile_n;
    }
    const int steps=k/bk;
    if(tid==0){
#pragma unroll
        for(int s=0;s<stages;s++)if(s<steps)load_stage<L,W>(a,b,ring+s*stage_bytes,s*bk,row0,col0,ready+s,panel_b,k);
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
                if(q==1&&j==10)retire_stage<L,W>(a,b,ring,ready,empty,slot,epoch,step,steps,row0,col0,panel_b,k);
#pragma unroll
                for(int i=0;i<4;i++){
                    scaled_multiply32<(Fold==2 && Order==1 && L==128 && W==256)>(accum[i][j],av[i],bp[0]);
                    scaled_multiply32<(Fold==2 && Order==1 && L==128 && W==256)>(accum[i][j+1],av[i],bp[1]);
                }
            }
        }
        if(++slot==stages){slot=0;epoch^=1;}
    }
    if constexpr(Fold==2 && Order==1 && L==128 && W==256){
    const int half=(lane>>1)&1,segment=lane&1;
    constexpr int wave=3;
#pragma unroll
    for(int start=0;start<6;start+=wave){
        Word words[wave]={};
#pragma unroll
        for(int q=0;q<4;q++){
            Word lo[wave][4],hi[wave][4];
#pragma unroll
            for(int g=0;g<wave;g++){
                const int ticket=start+g;
#pragma unroll
                for(int t=0;t<4;t++){
                    const int row=ticket<4?ticket:2*(ticket-4)+(t&1);
                    const int fragment=ticket<4?4*(t&1)+q:8+q;
                    const int part=2*(t/2);
                    lo[g][t]=__float_as_uint(accum[row][fragment][part]);
                    hi[g][t]=__float_as_uint(accum[row][fragment][part+1]);
                }
                transpose(lo[g],lane);
                transpose(hi[g],lane);
            }
#pragma unroll
            for(int t=0;t<4;t++){
#pragma unroll
                for(int g=0;g<wave;g++)words[g]=fold_step(words[g],lo[g][t]);
#pragma unroll
                for(int g=0;g<wave;g++)words[g]=fold_step(words[g],hi[g][t]);
            }
        }
#pragma unroll
        for(int g=0;g<wave;g++){
            const int ticket=start+g;
            const int row=row0+warp_row+16*(ticket<4?ticket:2*(ticket-4)+segment)+lane/4+8*half;
            const int col=col0+warp_col+(ticket<4?32*segment:64);
            if((row/16+1)*16<=m&&col+32<=n)output[(size_t(row/16)*fold_cols+col/32)*16+row%16]=words[g];
        }
    }
    }else if constexpr(Fold==2){
    // Each 96-column warp has two complete tickets in its first 64
    // columns. All four lanes of each quad have distinct owners here.
    const int half=(lane>>1)&1,segment=lane&1;
#pragma unroll
    for(int i=0;i<4;i++){
        const int row=row0+warp_row+16*i+lane/4+8*half;
        const int col=col0+warp_col+32*segment;
        Word word=0;
#pragma unroll
        for(int q=0;q<4;q++){
            Word lo[4],hi[4];
#pragma unroll
            for(int t=0;t<4;t++){
                const int frag=4*(t&1)+q,part=2*(t/2);
                lo[t]=__float_as_uint(accum[i][frag][part]);hi[t]=__float_as_uint(accum[i][frag][part+1]);
            }
            transpose(lo,lane);transpose(hi,lane);
#pragma unroll
            for(int t=0;t<4;t++){word=fold_step(word,lo[t]);word=fold_step(word,hi[t]);}
        }
        if((row/16+1)*16<=m&&col+32<=n)output[(size_t(row/16)*fold_cols+col/32)*16+row%16]=word;
    }
    // For the third ticket, input register vectors 0/2 select the
    // even row group and 1/3 the odd group, under the same 4x4
    // transpose. Every lane now owns one complete, ordered ticket.
#pragma unroll
    for(int pair=0;pair<4;pair+=2){
        const int row=row0+warp_row+16*(pair+segment)+lane/4+8*half;
        const int col=col0+warp_col+64;
        Word word=0;
#pragma unroll
        for(int q=0;q<4;q++){
            Word lo[4],hi[4];
#pragma unroll
            for(int t=0;t<4;t++){
                const int i=pair+(t&1),frag=8+q,part=2*(t/2);
                lo[t]=__float_as_uint(accum[i][frag][part]);hi[t]=__float_as_uint(accum[i][frag][part+1]);
            }
            transpose(lo,lane);transpose(hi,lane);
#pragma unroll
            for(int t=0;t<4;t++){word=fold_step(word,lo[t]);word=fold_step(word,hi[t]);}
        }
        if((row/16+1)*16<=m&&col+32<=n)output[(size_t(row/16)*fold_cols+col/32)*16+row%16]=word;
    }
    }else{
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
                    lo[t]=output_bits<Fold!=0>(accum[i][frag][part]);hi[t]=output_bits<Fold!=0>(accum[i][frag][part+1]);
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
}

struct Descriptor {
    CUtensorMap map{};const Byte* previous=nullptr;int old_count=0,old_k=0,old_width=0;
    void bind(const Driver& api,const Byte* data,int count,int k,int box_rows,int physical_width){
        if(!previous||old_count!=count||old_k!=k||old_width!=physical_width){
            cuuint64_t size[2]={cuuint64_t(physical_width),cuuint64_t(count)*(k/physical_width)},stride[1]={cuuint64_t(physical_width)};
            cuuint32_t box[2]={bk,static_cast<cuuint32_t>(box_rows)},increments[2]={1,1};
            check_driver(api.encode(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,const_cast<Byte*>(data),size,stride,box,increments,
                CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"encode streamed tensor map");
        }else if(previous!=data){
            check_driver(api.replace(&map,const_cast<Byte*>(data)),"update streamed tensor map address");
        }
        previous=data;old_count=count;old_k=k;old_width=physical_width;
    }
};
inline int selected_grid(){
    const char* s=std::getenv("PEARL_B_REUSE_GRID");
    if(!s||!std::strcmp(s,"stock"))return 0;
    if(!std::strcmp(s,"columns"))return 1;
    if(!std::strcmp(s,"n4"))return 4;
    if(!std::strcmp(s,"n16"))return 16;
    throw std::invalid_argument("unknown B reuse grid");
}
class Plan {
    const int grid_=selected_grid();
    const bool pair_columns_;
    Descriptor left_,right_,tail_right_;
    int k_=0;std::size_t calls_=0;
public:
    explicit Plan(bool pair_columns=false):pair_columns_(pair_columns){}
    std::size_t calls()const{return calls_;}
    template<int L,int W,int Fold=0> void run(const Byte* a,const Byte* b,Word* output,int k,int m,int width,int ld,cudaStream_t stream){
        if(!a||!b||!output||!pearl_geometry::supported_scan_depth(k))throw std::invalid_argument("unsupported fused stream input");
        static const Driver api;
        if(!k_){
            if(!grid_)check_cuda(cudaFuncSetAttribute(scan_window<L,W,0,Fold>,cudaFuncAttributeMaxDynamicSharedMemorySize,stages*stage_bytes),"configure stock grid");
            else if(grid_==1 && pair_columns_)check_cuda(cudaFuncSetAttribute(scan_window<L,W,1,Fold,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,stages*stage_bytes),"configure paired columns");
            else if(grid_==1)check_cuda(cudaFuncSetAttribute(scan_window<L,W,1,Fold>,cudaFuncAttributeMaxDynamicSharedMemorySize,stages*stage_bytes),"configure column grid");
            else check_cuda(cudaFuncSetAttribute(scan_window<L,W,2,Fold>,cudaFuncAttributeMaxDynamicSharedMemorySize,stages*stage_bytes),"configure N-group grid");
            k_=k;
        }else if(k!=k_)throw std::logic_error("fused stream plan reused for a different job depth");
        left_.bind(api,a,m,k,tile_m,L?L:k);
        const int gm=m/tile_m,gn=(width+tile_n-1)/tile_n;
        if(!grid_||grid_==1){
            right_.bind(api,b,width,k,tile_n,W?W:k);
            if(!grid_)scan_window<L,W,0,Fold><<<gm*gn,256,stages*stage_bytes,stream>>>(left_.map,right_.map,output,m,width,k,ld,b);
            else if(pair_columns_)scan_window<L,W,1,Fold,true><<<dim3(gm,gn),256,stages*stage_bytes,stream>>>(left_.map,right_.map,output,m,width,k,ld,b);
            else scan_window<L,W,1,Fold><<<dim3(gm,gn),256,stages*stage_bytes,stream>>>(left_.map,right_.map,output,m,width,k,ld,b);
            check_cuda(cudaGetLastError(),"launch full B reuse grid");++calls_;
        }else{
            if(width%tile_n)throw std::invalid_argument("N groups require a complete set of body columns");
            const int cols=gn/grid_*grid_;
            auto issue=[&](int first,int count,int band,Descriptor& descriptor){
                if(!count)return;
                const int n=count*tile_n;
                const Byte* bp=b+size_t(first)*tile_n*k;
                Word* dest=output+size_t(first)*(tile_n/32)*16;
                descriptor.bind(api,bp,n,k,tile_n,W?W:k);
                scan_window<L,W,2,Fold><<<dim3(gm*band,count/band),256,stages*stage_bytes,stream>>>(left_.map,descriptor.map,dest,m,n,k,ld,bp);
                check_cuda(cudaGetLastError(),"launch bounded B reuse grid");++calls_;
            };
            issue(0,cols,grid_,right_);issue(cols,gn-cols,gn-cols,tail_right_);
        }
    }
};
}
