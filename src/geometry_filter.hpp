#pragma once
#include "worker_buffer_pool.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace pearl_geometry {

constexpr std::size_t legacy_words=std::size_t(128)<<20;

inline bool expandable(bool requested,bool certified,int m,int n,int k,int h,int w,int rows,int cols,
                       std::size_t passes,std::size_t words) {
    return requested && certified && h==16 && w==32 && rows==4096 && cols==16384 &&
           m%rows==0 && n%cols==0 && (k==16384 || k==32768) &&
           passes<=8192 && words>legacy_words;
}

inline bool fits_full(std::size_t compact_bytes,std::size_t panel_words,std::size_t full_words,
                      std::size_t extra_workspace,std::size_t credited_available) {
    if (full_words<panel_words || compact_bytes>pearl_pool::ceiling ||
        extra_workspace>pearl_pool::ceiling-compact_bytes) return false;
    const std::size_t fixed=compact_bytes+extra_workspace;
    if (full_words-panel_words>(pearl_pool::ceiling-fixed)/sizeof(std::uint32_t)) return false;
    const std::size_t full_bytes=fixed+(full_words-panel_words)*sizeof(std::uint32_t);
    return pearl_pool::admit(full_bytes,0,false,credited_available).fits;
}

// filter_slab emits ascending global tile IDs. The previous per-window
// writer emitted each column window first and ascending tile IDs within it.
// Sorting the rare hits only preserves the old wire bytes at both pressures.
template<class Hit,class Accept>
inline void append_window_order(Hit* hits,std::size_t count,std::uint64_t full_columns,
                                std::uint64_t window_columns,Accept accept) {
    if(count<2 || full_columns==window_columns){accept(hits,count);return;}
    std::sort(hits,hits+count,[&](const Hit& a,const Hit& b) {
        const auto aw=(a.tile%full_columns)/window_columns, bw=(b.tile%full_columns)/window_columns;
        return aw==bw ? a.tile<b.tile : aw<bw;
    });
    accept(hits,count);
}

}  // namespace pearl_geometry
