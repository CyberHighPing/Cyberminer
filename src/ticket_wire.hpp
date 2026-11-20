#pragma once

#include "ticket_filter.cuh"
#include <array>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace pearl_ticket {
class SparseWire {
    std::array<unsigned char,160> header_{};
    Index expected_ = 0, hits_ = 0;
    static Word read32(const unsigned char* p) {
        return Word(p[0]) | Word(p[1])<<8 | Word(p[2])<<16 | Word(p[3])<<24;
    }
    static Index read64(const unsigned char* p) {
        return Index(read32(p)) | (Index(read32(p+4)) << 32);
    }
    static void write64(unsigned char* p,Index x) {
        for(unsigned i=0;i<8;++i) p[i]=static_cast<unsigned char>(x>>(8*i));
    }
public:
    Job job{};
    SparseWire(std::istream& params,Word m,Word n,Word k,Word h,Word w) {
        const Word one=1;
        if (*reinterpret_cast<const unsigned char*>(&one)!=1)
            throw std::runtime_error("sparse CUDA writer requires little-endian hit records");
        params.read(reinterpret_cast<char*>(header_.data()),header_.size());
        if(!params || params.peek()!=std::char_traits<char>::eof())
            throw std::runtime_error("ticket parameter file must be exactly 160 bytes");
        const Word dims[]={1,160,104,m,n,k,h,w};
        if(std::memcmp(header_.data(),"PRLFP8S1",8)!=0 || !h || !w || m<h || n<w)
            throw std::runtime_error("invalid ticket parameter header or dimensions");
        for(unsigned i=0;i<8;++i)
            if(read32(header_.data()+8+4*i)!=dims[i])
                throw std::runtime_error("ticket parameters disagree with scanner geometry");
        expected_ = Index(m/h)*(n/w);
        if(read64(header_.data()+40)!=expected_ || read64(header_.data()+48) || read64(header_.data()+56))
            throw std::runtime_error("invalid initial ticket counts");
        for(unsigned i=0;i<8;++i) {
            job.key[i]=read32(header_.data()+64+4*i);
            job.bound[i]=read32(header_.data()+96+4*i);
        }
    }
    Index hits() const { return hits_; }
    void begin(std::ostream& out) const {
        out.write(reinterpret_cast<const char*>(header_.data()),header_.size());
        if(!out) throw std::runtime_error("failed to write sparse header");
    }
    void append(std::ostream& out,const Hit* hits,std::size_t n) {
        if(n>expected_-hits_ || n>std::size_t((std::numeric_limits<std::streamsize>::max)())/sizeof(Hit))
            throw std::runtime_error("invalid sparse hit count");
        if(n) out.write(reinterpret_cast<const char*>(hits),std::streamsize(n*sizeof(Hit)));
        if(!out) throw std::runtime_error("failed to write sparse hit records");
        hits_+=n;
    }
    void finish(std::ostream& out,const Coverage& cover) const {
        if(!cover.complete() || cover.scanned()!=expected_)
            throw std::runtime_error("incomplete sparse ticket grid");
        unsigned char counts[16];
        write64(counts,cover.scanned()); write64(counts+8,hits_);
        out.seekp(48);out.write(reinterpret_cast<const char*>(counts),sizeof(counts));
        out.seekp(0,std::ios::end);out.flush();
        if(!out) throw std::runtime_error("failed to finalize sparse hit file");
    }
};
}  // namespace pearl_ticket
