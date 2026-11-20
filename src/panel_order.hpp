#pragma once
#include "ordered_hits.hpp"
#include <algorithm>
#include <memory>
#include <vector>
namespace native_width {
// The file format has two established orders. Small jobs are globally sorted.
// Large rectangular jobs are sorted inside legacy column bands, then by band.
// Each input run is already sorted. OrderedHits bounds memory for an easy target.
template<class Hit> class CanonicalOrder {
    using Merge=pearl_square::OrderedHits<Hit>;
    int fd_;size_t runs_,tile_cols_,physical_,legacy_,total_;
    bool global_;
    std::vector<std::unique_ptr<Merge>> bands_;
    size_t band(const Hit& h)const{return global_?0:(h.tile%tile_cols_)/legacy_;}
public:
    CanonicalOrder(int fd,size_t rows,size_t full_cols,size_t physical,size_t legacy,bool global):
       fd_(fd),runs_((full_cols+physical-1)/physical),tile_cols_(full_cols),physical_(physical),legacy_(legacy),
       total_(rows*full_cols),global_(global),bands_(global?1:(full_cols+legacy-1)/legacy){}
    void append(size_t run,const Hit* h,size_t count){
        for(size_t i=0;i<count;){
            const size_t id=band(h[i]);size_t j=i+1;
            while(j<count&&band(h[j])==id)++j;
            if(!bands_[id])bands_[id]=std::make_unique<Merge>(fd_,runs_,tile_cols_,physical_,total_);
            bands_[id]->append(run,h+i,j-i);
            i=j;
        }
    }
    template<class Emit>void emit(Emit sink){for(auto& b:bands_)if(b)b->emit(sink);}
};
}
