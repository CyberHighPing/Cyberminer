#pragma once
#include "ticket_filter.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace queued_filter {
using pearl_ticket::Count;
using pearl_ticket::Hit;
using pearl_ticket::Job;
using pearl_ticket::Slab;
using pearl_ticket::Word;

inline void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string("batched ticket filter: ")+cudaGetErrorString(status));
}

inline int requested() {
    const char* value=std::getenv("PEARL_R10_FILTER_BATCH");
    if(!value||!std::strcmp(value,"off"))return 1;
    for(int size:{1,2,4,8,16,32,64})if(std::to_string(size)==value)return size;
    throw std::invalid_argument("invalid ticket batch size");
}

// Folded words are consumed on the producer's own stream before it overwrites
// the only fold slab. Each pending slab owns independent bounded hit storage.
class Batch {
    Hit* device_hits_;
    Count* device_counts_;
    size_t capacity_,slots_;
    std::vector<Slab> slabs_;
    std::vector<Count> counts_;
    std::vector<Hit> hits_;
public:
    Batch(Hit* hits,Count* counts,size_t capacity,size_t slots)
        :device_hits_(hits),device_counts_(counts),capacity_(capacity),slots_(slots),counts_(slots) {
        if(!hits||!counts||!capacity||!slots||slots>64)throw std::invalid_argument("invalid filter storage");
        slabs_.reserve(slots);
    }
    void begin() {
        if(!slabs_.empty())throw std::logic_error("previous batch not consumed");
        check(cudaMemsetAsync(device_counts_,0,slots_*sizeof(Count),nullptr));
    }
    void issue(const Word* folds,Slab slab,const Job& job) {
        size_t i=slabs_.size();
        if(i>=slots_||!pearl_ticket::valid_slab(slab))throw std::logic_error("invalid filter batch interval");
        const auto tiles=slab.tile_rows*slab.tile_cols;
        check(pearl_ticket::launch_range(folds,slab,job,0,tiles,device_hits_+i*capacity_,device_counts_+i,capacity_,nullptr));
        slabs_.push_back(slab);
    }
    template<class Recover,class Accept>
    size_t collect(Recover&& recover,Accept&& accept) {
        check(cudaMemcpy(counts_.data(),device_counts_,slabs_.size()*sizeof(Count),cudaMemcpyDeviceToHost));
        size_t overflow_count=0;
        for(size_t i=0;i<slabs_.size();i++) {
            const auto& slab=slabs_[i];
            const Count count=counts_[i];
            if(count>slab.tile_rows*slab.tile_cols)throw std::runtime_error("invalid ticket count");
            if(!count)continue;
            auto checked=[&](const Hit* records,size_t n) {
                if(n!=count)throw std::runtime_error("inconsistent replayed ticket count");
                for(size_t j=0;j<n;j++) {
                    auto row=records[j].tile/slab.full_tile_cols,col=records[j].tile%slab.full_tile_cols;
                    if(row<slab.tile_row0||row-slab.tile_row0>=slab.tile_rows||
                       col<slab.tile_col0||col-slab.tile_col0>=slab.tile_cols||
                       (j&&records[j-1].tile>=records[j].tile))throw std::runtime_error("unexpected ticket location");
                }
                accept(records,n);
            };
            if(count>capacity_) {
                // Tentative records cannot be committed. Reproduce the complete
                // genuine folded slab and let the standard range splitter drain it.
                recover(slab,device_hits_+i*capacity_,device_counts_+i,checked);
                ++overflow_count;
            }else{
                hits_.resize(static_cast<size_t>(count));
                check(cudaMemcpy(hits_.data(),device_hits_+i*capacity_,hits_.size()*sizeof(Hit),cudaMemcpyDeviceToHost));
                std::sort(hits_.begin(),hits_.end(),[](const Hit& a,const Hit& b){return a.tile<b.tile;});
                checked(hits_.data(),hits_.size());
            }
        }
        slabs_.clear();
        return overflow_count;
    }
};
}
