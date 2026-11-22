#pragma once
#include "job_generator.cuh"
#include "panel_b.cuh"
#include "prepare_resident.cuh"
#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace native_inputs {
using Byte=std::uint8_t;
using Word=std::uint32_t;
using Digest=pearl_jobgen::Digest;
using Clock=std::chrono::steady_clock;

inline void gpu(cudaError_t result,const char* op) {
    if(result!=cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(result));
}
template<class T> T at(const Byte* p) {
    T v; std::memcpy(&v,p,sizeof(v)); return v;
}
inline const Byte* bytes(const Digest& d) {return reinterpret_cast<const Byte*>(d.words);}
inline void append(std::vector<Byte>& buf,const Digest& d) {
    buf.insert(buf.end(),bytes(d),bytes(d)+32);
}

// All transcript messages here are in one BLAKE3 chunk. A non-final block
// updates the chaining value; the final block carries CHUNK_END and ROOT.
inline Digest keyed(const Digest& key,const Byte* message,size_t length) {
    if(!length || length>1024) throw std::logic_error("unexpected V4 transcript message length");
    Digest cv=key;
    for(size_t offset=0;offset<length;offset+=64) {
        const auto part=std::min(size_t(64),length-offset);
        Word block[16]={};
        std::memcpy(block,message+offset,part);
        const Word flags=16u|(offset==0?1u:0u)|(offset+part==length?10u:0u);
        pearl_jobgen::detail::compress<false>(cv.words,block,0,Word(part),flags,cv.words);
    }
    return cv;
}
inline Digest subkey(const Digest& key,const char* label) {
    return keyed(key,reinterpret_cast<const Byte*>(label),std::strlen(label));
}

struct State {
    Digest ha{}, hb{}, sa{}, sb{}, jackpot{};
    double generation_s=0, quantization_s=0;
    std::array<Byte,136> receipt() const {
        std::array<Byte,136> out{};
        std::memcpy(out.data(),"PRLNF8R1",8);
        std::memcpy(out.data()+8,&ha,32); std::memcpy(out.data()+40,&hb,32);
        std::memcpy(out.data()+72,&sa,32); std::memcpy(out.data()+104,&sb,32);
        return out;
    }
};

class Plan {
    std::vector<Byte> file_;
    std::array<Word,5> dims_;
    Digest seed_{}, key_a_{}, key_b_{}, role_a_{}, role_b_{};
    pearl_jobgen::Plan a_{},b_{};
public:
    Plan(const std::filesystem::path& path,int m,int n,int k,int h,int w)
        :dims_{Word(m),Word(n),Word(k),Word(h),Word(w)} {
        const size_t count=std::filesystem::file_size(path);
        if(count<504 || count>828) throw std::runtime_error("invalid native control size");
        file_.resize(count);
        std::ifstream in(path,std::ios::binary);
        if(!in.read(reinterpret_cast<char*>(file_.data()),count))
            throw std::runtime_error("truncated native control");
        if(std::memcmp(file_.data(),"PRLNF8J1",8) ||
           std::memcmp(file_.data()+8,dims_.data(),20) ||
           at<Word>(file_.data()+248)!=count-252 ||
           (count-504)%108)
            throw std::runtime_error("native control geometry or context mismatch");
        seed_=at<Digest>(file_.data()+28); key_a_=at<Digest>(file_.data()+60);
        key_b_=at<Digest>(file_.data()+92); role_a_=at<Digest>(file_.data()+184);
        role_b_=at<Digest>(file_.data()+216);
        if(!pearl_jobgen::make_plan(m,k,a_) || !pearl_jobgen::make_plan(n,k,b_))
            throw std::runtime_error("invalid native operand sizes");
    }
    size_t scratch_bytes(int batch) const {
        return std::max({a_.scratch_bytes,b_.scratch_bytes,pearl_noise::scratch_bytes(dims_[2],batch)});
    }
    size_t panel_a_storage() const {return pearl_panel_b::storage(a_);}
    Digest commit_stream_role(pearl_jobgen::Side side,Byte* arena,size_t size,prep_inputs::Row* meta,Word* invalid,bool fused,proof_checkpoints::Planes* cuts=nullptr,Byte* raw_output=nullptr) const {
        const bool left=side==pearl_jobgen::Side::A;
        pearl_jobgen::Roots roots{};
        gpu(pearl_panel_b::commit(arena,size,left?a_:b_,seed_,left?key_a_:key_b_,roots,nullptr,meta,invalid,fused,side,cuts,raw_output),"stream and commit raw operand");
        return roots.combined;
    }
    size_t panel_b_storage() const {return pearl_panel_b::storage(b_);}
    size_t panel_b_scratch(int batch) const {return std::max(a_.scratch_bytes,pearl_noise::scratch_bytes(dims_[2],batch));}
    const Digest& generator_seed()const{return seed_;}
    Digest commit_panel_b(Byte* arena,size_t size,prep_inputs::Row* meta=nullptr,Word* invalid=nullptr,bool inline_leaves=false,proof_checkpoints::Planes* cuts=nullptr) const {
        pearl_jobgen::Roots roots{};gpu(pearl_panel_b::commit(arena,size,b_,seed_,key_b_,roots,nullptr,meta,invalid,inline_leaves,pearl_jobgen::Side::B,cuts),"stream raw B leaves and commit");
        return roots.combined;
    }
    void write_cut(std::ostream& out,const State& state,const proof_checkpoints::Planes& cuts) const {
        std::array<Byte,SHA256_DIGEST_LENGTH> hash{};
        if(!SHA256(file_.data(),file_.size(),hash.data()))throw std::runtime_error("control hash failed");
        const auto roots=state.receipt();
        out.write("PRLMCUT1",8);
        out.write(reinterpret_cast<const char*>(hash.data()),hash.size());
        out.write(reinterpret_cast<const char*>(roots.data()),roots.size());
        for(const auto& c:cuts) {
            if(!c.leaves||c.nodes.size()!=proof_checkpoints::width(c.leaves))throw std::logic_error("incomplete Merkle checkpoint");
            out.write(reinterpret_cast<const char*>(&c.leaves),sizeof(c.leaves));
            out.write(reinterpret_cast<const char*>(c.nodes.data()),c.nodes.size()*sizeof(Digest));
        }
    }
    State bind(const Digest& hash_a,const Digest& hash_b) const {
        State s;
        s.ha=hash_a; s.hb=hash_b;
        std::vector<Byte> msg;
        append(msg,s.hb); append(msg,key_b_); msg.insert(msg.end(),file_.begin()+165,file_.begin()+184);
        s.sb=keyed(role_b_,msg.data(),msg.size());
        msg.clear(); append(msg,s.ha); append(msg,s.sb); append(msg,key_a_);
        msg.insert(msg.end(),file_.begin()+156,file_.begin()+165);
        s.sa=keyed(role_a_,msg.data(),msg.size());
        s.jackpot=subkey(s.sa,"pearl/v4/FP8/jackpot");
        return s;
    }
    Digest generate_and_commit(pearl_jobgen::Side side,Byte* data,size_t capacity,void* scratch,size_t scratch_capacity,prep_inputs::Row* meta=nullptr,Word* invalid=nullptr) const {
        const bool left=side==pearl_jobgen::Side::A;
        pearl_jobgen::Roots r{};
        if(meta){
            const auto& p=left?a_:b_;const auto& key=left?key_a_:key_b_;
            gpu(prep_inputs::generate_and_measure(data,capacity,p,side,seed_,0,p.rows,meta,invalid,nullptr),"generate raw rows and coefficients");
            gpu(pearl_jobgen::launch_commit(data,capacity,p,key,scratch,scratch_capacity,nullptr),"commit complete raw operand");
            gpu(cudaMemcpy(&r,pearl_jobgen::device_roots(scratch,p),sizeof(r),cudaMemcpyDeviceToHost),"read operand roots");return r.combined;
        }
        gpu(pearl_jobgen::generate_and_commit(data,capacity,left?a_:b_,side,seed_,left?key_a_:key_b_,
                                              scratch,scratch_capacity,&r,nullptr),"generate operand and commit");
        return r.combined;
    }
    void regenerate_a(Byte* data,size_t capacity,unsigned first,unsigned rows,cudaStream_t stream) const {
        gpu(pearl_jobgen::launch_generate_rows(data,capacity,a_,pearl_jobgen::Side::A,seed_,first,rows,stream),
            "regenerate A row strip");
    }
    void regenerate_b(Byte* data,size_t capacity,unsigned first,unsigned rows) const {
        gpu(pearl_jobgen::launch_generate_rows(data,capacity,b_,pearl_jobgen::Side::B,seed_,first,rows,nullptr),
            "regenerate B column panel");
    }
    State materialize(Byte* a,Byte* b,void* scratch,size_t scratch_capacity,unsigned* invalid,int batch) const {
        const auto begin=Clock::now();
        const Digest ha=generate_and_commit(pearl_jobgen::Side::A,a,a_.value_bytes,scratch,scratch_capacity);
        const Digest hb=generate_and_commit(pearl_jobgen::Side::B,b,b_.value_bytes,scratch,scratch_capacity);
        State s=bind(ha,hb);
        const auto ea=subkey(s.sa,"pearl/v4/FP8/noise-line"), fb=subkey(s.sb,"pearl/v4/FP8/noise-line");
        const auto prepared=Clock::now();
        gpu(cudaMemset(invalid,0,sizeof(unsigned)),"reset preparation status");
        gpu(pearl_noise::prepare_resident(reinterpret_cast<std::int8_t*>(a),int(dims_[0]),int(dims_[2]),0,
                                          bytes(ea),bytes(fb),batch,scratch,scratch_capacity,invalid,nullptr),"quantize A");
        gpu(pearl_noise::prepare_resident(reinterpret_cast<std::int8_t*>(b),int(dims_[1]),int(dims_[2]),1,
                                          bytes(fb),bytes(fb),batch,scratch,scratch_capacity,invalid,nullptr),"quantize B");
        unsigned bad=0; gpu(cudaMemcpy(&bad,invalid,sizeof(bad),cudaMemcpyDeviceToHost),"read preparation status");
        if(bad) throw std::runtime_error("native inputs failed the row policy");
        s.generation_s=std::chrono::duration<double>(prepared-begin).count();
        s.quantization_s=std::chrono::duration<double>(Clock::now()-prepared).count();
        return s;
    }
    std::string ticket_header(const State& s) const {
        std::array<Byte,160> buf{};
        std::memcpy(buf.data(),"PRLFP8S1",8);
        Word layout[3]={1,160,104};
        std::memcpy(buf.data()+8,layout,sizeof(layout));
        std::memcpy(buf.data()+20,dims_.data(),20);
        const std::uint64_t total=std::uint64_t(dims_[0]/dims_[3])*(dims_[1]/dims_[4]);
        std::memcpy(buf.data()+40,&total,sizeof(total));
        std::memcpy(buf.data()+64,&s.jackpot,32);
        std::memcpy(buf.data()+96,file_.data()+124,32);
        std::vector<Byte> message(file_.begin()+252,file_.end());
        append(message,s.ha);append(message,s.hb);append(message,s.sa);append(message,s.sb);
        if(!SHA256(message.data(),message.size(),buf.data()+128)) throw std::runtime_error("job tag digest failed");
        return std::string(reinterpret_cast<const char*>(buf.data()),buf.size());
    }
};
}
