#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>
#include "private_spill.hpp"

namespace pearl_square {
// Merge per-column sorted runs into global physical ticket order. Sparse jobs
// keep small runs in memory. Unusually easy targets spill to private unnamed
// temporary files, so a large job never keeps all winning tickets in host RAM.
template<class Hit> class OrderedHits {
    struct Closer {void operator()(std::FILE* f)const {if(f)std::fclose(f);}};
    struct Run {
        std::unique_ptr<std::FILE,Closer> spill;
        std::vector<Hit> small;
        std::size_t total=0,read=0;
        std::uint64_t previous=0;
        void append(int staging_dir,const Hit* items,std::size_t count) {
            if(!count)return;
            for(std::size_t i=0;i<count;++i) {
                if((total || i) && items[i].tile<=previous) throw std::runtime_error("column hits are not strictly increasing");
                previous=items[i].tile;
            }
            if(!spill && count<=256-small.size()) {small.insert(small.end(),items,items+count);total+=count;return;}
            if(!spill) {
                spill.reset(private_spill::open(staging_dir));
                if(!small.empty() && std::fwrite(small.data(),sizeof(Hit),small.size(),spill.get())!=small.size())
                    throw std::runtime_error("cannot stage initial column hit run");
                small.clear();
            }
            if(std::fwrite(items,sizeof(Hit),count,spill.get())!=count) throw std::runtime_error("cannot stage column hit run");
            total+=count;
        }
        bool next(Hit& hit) {
            if(read==total)return false;
            if(spill) {
                if(read==0 && (std::fflush(spill.get()) || std::fseek(spill.get(),0,SEEK_SET)))
                    throw std::runtime_error("cannot rewind column hit run");
                if(std::fread(&hit,sizeof(Hit),1,spill.get())!=1)throw std::runtime_error("cannot read column hit run");
            } else hit=small[read];
            ++read;return true;
        }
    };
    struct Entry { Hit hit; std::size_t run; };
    struct Later {bool operator()(const Entry& a,const Entry& b)const{return a.hit.tile>b.hit.tile;}};
    std::vector<Run> runs_;
    std::uint64_t full_cols_,window_cols_,total_;
    int staging_dir_; // borrowed from paired_output::File, which outlives all runs
public:
    OrderedHits(int staging_dir,std::size_t count,std::uint64_t full_cols,std::uint64_t window_cols,std::uint64_t tickets)
        :runs_(count),full_cols_(full_cols),window_cols_(window_cols),total_(tickets),staging_dir_(staging_dir) {}
    void append(std::size_t run,const Hit* items,std::size_t n) {
        if(run>=runs_.size() || !full_cols_ || !window_cols_)throw std::logic_error("column hit run not configured");
        for(std::size_t i=0;i<n;++i)
            if(items[i].tile>=total_ || (items[i].tile%full_cols_)/window_cols_!=run)throw std::runtime_error("ticket outside column run");
        runs_[run].append(staging_dir_,items,n);
    }
    template<class Accept> void emit(Accept output) {
        std::priority_queue<Entry,std::vector<Entry>,Later> queue;
        for(std::size_t i=0;i<runs_.size();++i){Entry e{{},i};if(runs_[i].next(e.hit))queue.push(e);}
        std::array<Hit,64> batch;std::size_t n=0;std::uint64_t previous=0;bool emitted=false;
        while(!queue.empty()) {
            auto item=queue.top();queue.pop();
            if(emitted && item.hit.tile<=previous)throw std::runtime_error("duplicate or inverted global ticket");
            emitted=true;previous=item.hit.tile;batch[n++]=item.hit;
            if(n==batch.size()){output(batch.data(),n);n=0;}
            if(runs_[item.run].next(item.hit))queue.push(item);
        }
        if(n)output(batch.data(),n);
    }
};
}  // namespace pearl_square
