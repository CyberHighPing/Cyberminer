#pragma once
#include "prep_gpu.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace native_fold {
enum class Mode {canonical,finite,paired};
inline const char* name(Mode m){
    return m==Mode::paired?"paired":m==Mode::finite?"finite":"canonical";
}
inline Mode requested(){
    const char* p=std::getenv("PEARL_NATIVE_FINITE_FOLD");
    if(!p||!std::strcmp(p,"canonical"))return Mode::canonical;
    if(!std::strcmp(p,"finite"))return Mode::finite;
    if(!std::strcmp(p,"paired"))return Mode::paired;
    throw std::invalid_argument("unknown native matrix fold");
}
__global__ void check_strips(const pearl_prep::Row* rows,size_t count,unsigned* fallback){
    unsigned bad=0;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x){
        const unsigned strip=rows[i].strip;
        bad|=unsigned(strip>112u||(strip&7u)!=0);
    }
    if(bad)atomicOr(fallback,1u);
}
inline Mode certify(Mode desired,const pearl_prep::Row* metadata,size_t rows,int k,unsigned* scratch){
    if(desired==Mode::canonical)return desired;
    if(!metadata||!rows||!(k==8192||k==12288||k==13312||k==16384||k==20480||k==24576||k==32768)||!scratch)return Mode::canonical;
    auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    check(cudaMemsetAsync(scratch,0,sizeof(*scratch)));
    check_strips<<<256,128>>>(metadata,rows,scratch);check(cudaGetLastError());
    unsigned bad=1;check(cudaMemcpy(&bad,scratch,sizeof(bad),cudaMemcpyDeviceToHost));
    return bad?Mode::canonical:desired;
}
}
