#pragma once
#include "proof_checkpoints.hpp"

// Included after the original wire encoders. These nodes are the scanner's
// ordinary keyed Merkle tree at or below height six, never a different tree.
namespace checkpoint_open {
using Side=pearl_jobgen::Side;
using Plane=proof_checkpoints::Plane;

inline proof_checkpoints::Planes load(const fs::path& path,const Control& cfg,const Bytes& roots) {
    const std::array<Size,4> sizes={Size(cfg.m)*cfg.k,Size(cfg.n)*cfg.k,
                                  Size(cfg.m)*(cfg.k/4),Size(cfg.n)*(cfg.k/4)};
    Size bytes=8+SHA256_DIGEST_LENGTH+roots.size();
    for(Size size:sizes)bytes+=8+proof_checkpoints::width(divide_up(size,Chunk))*sizeof(Digest);
    require(fs::is_regular_file(path)&&fs::file_size(path)==bytes,"invalid Merkle checkpoint size");
    std::ifstream in(path,std::ios::binary);
    auto read=[&](void* p,Size n) {
        require(bool(in.read(static_cast<char*>(p),static_cast<std::streamsize>(n))),"truncated Merkle checkpoint");
    };
    char magic[8];read(magic,8);require(!std::memcmp(magic,"PRLMCUT1",8),"invalid Merkle checkpoint revision");
    std::array<Byte,SHA256_DIGEST_LENGTH> stored{},expected{};
    read(stored.data(),stored.size());
    require(SHA256(cfg.data.data(),cfg.data.size(),expected.data())&&stored==expected,"checkpoint control differs");
    Bytes original(roots.size());read(original.data(),original.size());
    require(original==roots,"checkpoint root receipt differs");
    proof_checkpoints::Planes planes;
    for(unsigned i=0;i<planes.size();++i) {
        Size stored_leaves;read(&stored_leaves,sizeof(stored_leaves));
        require(stored_leaves==divide_up(sizes[i],Chunk),"checkpoint leaf count differs");
        planes[i].prepare(stored_leaves);
        read(planes[i].nodes.data(),planes[i].nodes.size()*sizeof(Digest));
    }
    require(in.peek()==std::char_traits<char>::eof(),"trailing Merkle checkpoint data");
    return planes;
}

inline void dense_bytes(Byte* out,Size count,Size byte_offset,const Digest& seed,Side side) {
    require(byte_offset%32==0,"unaligned generator offset");
    for(Size i=0;i<count;i+=32) {
        Word words[8];
        pearl_jobgen::raw_block(seed,side,(byte_offset+i)/32,words);
        std::memcpy(out+i,words,std::min<Size>(32,count-i));
    }
}

template<bool Scales>
struct Source {
    Size size,leaves;
    const Digest& key;
    const Digest& seed;
    Side side;

    void fill(Byte* destination,Size count,Size offset) const {
        if constexpr(Scales) {
            for(Size i=0;i<count;++i)destination[i]=((offset+i)&1)?0x3f:0x80;
        } else dense_bytes(destination,count,offset,seed,side);
    }

    Digest data_leaf(const Byte* padded,Size index) const {
        return pearl_panel_b::value_leaf(padded,Chunk,0,index,key,leaves==1);
    }

    // Only siblings below the saved level are reconstructed. Use global leaf
    // counters and global child counts, including final unpaired promotions.
    Digest sibling(Size index,unsigned height) const {
        if(!height) {
            if constexpr(Scales)return pearl_jobgen::leaf<true>(nullptr,size,index,key,leaves==1);
            Byte raw[Chunk]{};
            fill(raw,std::min<Size>(Chunk,size-index*Chunk),index*Chunk);
            return data_leaf(raw,index);
        }
        const Size children=divide_up(leaves,Size(1)<<(height-1));
        const Size first=2*index;
        const Digest left=sibling(first,height-1);
        if(first+1==children)return left;
        return pearl_jobgen::fold_pair(left,sibling(first+1,height-1),key,children==2?28:20);
    }
};

template<bool Scales>
Opening open(const Control& cfg,Side side,Size start,const Plane& cut) {
    const bool a=side==Side::A;
    const Size stride=Scales?cfg.k/4:cfg.k;
    const Size bytes=(a?cfg.m:cfg.n)*stride;
    const Size n=divide_up(bytes,Chunk), rows=a?cfg.h:cfg.w;
    const Digest& key=a?cfg.key_a:cfg.key_b;
    Size left=start*stride/Chunk,right=divide_up((start+rows)*stride,Chunk),length=n;
    require(cut.leaves==n&&left<right&&right<=n&&(right-left)*Chunk<=MaxProof,"checkpoint opening outside operand");
    Source<Scales> source{bytes,n,key,cfg.seed,side};
    Opening result{start,rows,left,n};
    result.data.resize((right-left)*Chunk,0);
    source.fill(result.data.data(),std::min<Size>(result.data.size(),bytes-left*Chunk),left*Chunk);
    std::vector<Digest> opened;
    opened.reserve(right-left+2);
    for(Size i=left;i<right;++i)opened.push_back(source.data_leaf(result.data.data()+(i-left)*Chunk,i));
    unsigned height=0;
    while(length>cut.nodes.size()) {
        if(left&1) {
            const Digest d=source.sibling(left-1,height);
            result.siblings.push_back(d);opened.insert(opened.begin(),d);
        }
        if((right&1)&&right<length) {
            const Digest d=source.sibling(right,height);
            result.siblings.push_back(d);opened.push_back(d);
        }
        for(Size i=0;i<opened.size();i+=2) {
            opened[i/2]=i+1==opened.size()?opened[i]:pearl_jobgen::fold_pair(opened[i],opened[i+1],key,length==2?28:20);
        }
        opened.resize(divide_up(opened.size(),2));
        left/=2;right=divide_up(right,2);length=divide_up(length,2);++height;
    }
    require(length==cut.nodes.size()&&opened.size()==right-left,"checkpoint tree has the wrong level");
    for(Size i=left;i<right;++i)require(equal(opened[i-left],cut.nodes[i]),"opened leaves differ from scanner checkpoint");

    std::vector<Digest> upper=cut.nodes;
    while(length>1) {
        if(left&1)result.siblings.push_back(upper[left-1]);
        if((right&1)&&right<length)result.siblings.push_back(upper[right]);
        for(Size i=0;i<length;i+=2) {
            upper[i/2]=i+1==length?upper[i]:pearl_jobgen::fold_pair(upper[i],upper[i+1],key,length==2?28:20);
        }
        left/=2;right=divide_up(right,2);length=divide_up(length,2);
    }
    result.root=upper.front();
    return result;
}

inline Operand operand(const Control& cfg,Side side,Size start,const Digest& expected,const proof_checkpoints::Planes& cuts) {
    const unsigned i=side==Side::A?0:1;
    const Digest& key=i?cfg.key_b:cfg.key_a;
    auto values=open<false>(cfg,side,start,cuts[i]);
    auto scales=open<true>(cfg,side,start,cuts[i+2]);
    require(equal(pearl_jobgen::fold_pair(values.root,scales.root,key,27),expected),"checkpoint differs from committed operand root");
    return {std::move(values),std::move(scales),0,true};
}
}
