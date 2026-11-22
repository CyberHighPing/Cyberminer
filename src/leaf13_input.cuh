#pragma once
#include "leaf13_math.hpp"
#include "prep_inputs.cuh"

namespace leaf13 {
inline bool enabled(unsigned k){return k==row_bytes;}
template<unsigned Rows,Side Operand>
__global__ void tile(Size row_count,Digest seed,Digest key,pearl_jobgen::RawPrefix prefix,Digest* values,
                     prep_inputs::Row* metadata,Word* invalid){
    constexpr unsigned leaves=Rows*leaves_per_row;
    __shared__ Word summaries[leaves];
    const unsigned id=threadIdx.x;
    const Size row0=Size(blockIdx.x)*Rows;
    const Size leaf0=row0*leaves_per_row;
    const Size leaf=leaf0+id;
    if(id<leaves && leaf<row_count*leaves_per_row){
        Digest digest;
        summaries[id]=value_leaf<Operand>(leaf,seed,key,prefix,digest);
        values[leaf]=digest;
    }
    __syncthreads();
    if(id<Rows && row0+id<row_count){
        Word sum,peak;
        row_statistics(summaries+id*leaves_per_row,sum,peak);
        metadata[row0+id]=pearl_prep::row_coefficients(sum,peak,row_bytes,invalid);
    }
}
template<unsigned Rows,Side Operand>
inline void launch_tile(Size n,Digest seed,Digest key,pearl_jobgen::RawPrefix prefix,Digest* values,
                        prep_inputs::Row* meta,Word* invalid,cudaStream_t stream){
    constexpr unsigned threads=(Rows*leaves_per_row+31)/32*32;
    tile<Rows,Operand><<<unsigned((n+Rows-1)/Rows),threads,0,stream>>>(n,seed,key,prefix,values,meta,invalid);
}
template<Side Operand>
inline cudaError_t run(const pearl_jobgen::Plan& p,Digest seed,Digest key,
                       Digest* values,prep_inputs::Row* meta,Word* invalid,cudaStream_t stream){
    if(!meta||!invalid||!values||!p.rows||p.rows>=(1u<<24)||p.k!=row_bytes||
       p.value_chunks!=p.rows*leaves_per_row||p.value_bytes!=p.value_chunks*leaf_bytes)
        return cudaErrorInvalidValue;
    const auto prefix=pearl_jobgen::raw_prefix(seed,Operand);
    launch_tile<32,Operand>(p.rows,seed,key,prefix,values,meta,invalid,stream);
    return cudaGetLastError();
}
}
