#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#ifdef __CUDACC__
#define DP_HD __host__ __device__
#else
#define DP_HD
#endif
namespace directpack {
enum class Mode { row,slab_row,tma64,tma128,tma256,tma512,main256,bulk };
struct Options {Mode mode;bool xof;};
inline const char* name(Mode m){
    switch(m){case Mode::row:return "row";case Mode::slab_row:return "slab-row";case Mode::tma64:return "tma64";
        case Mode::tma128:return "tma128";case Mode::tma256:return "tma256";case Mode::tma512:return "tma512";
        case Mode::main256:return "main256";case Mode::bulk:return "bulk";}
    throw std::invalid_argument("invalid direct B layout");
}
inline Mode parse(const std::string& s){
    for(auto m:{Mode::row,Mode::slab_row,Mode::tma64,Mode::tma128,Mode::tma256,Mode::tma512,Mode::main256,Mode::bulk})
        if(s==name(m))return m;
    throw std::invalid_argument("invalid PEARL_DIRECTPACK");
}
inline Options options(){
    const char* layout=std::getenv("PEARL_DIRECTPACK");const char* source=std::getenv("PEARL_DIRECTPACK_SOURCE");
    if(source&&std::strcmp(source,"raw")&&std::strcmp(source,"xof"))throw std::invalid_argument("PEARL_DIRECTPACK_SOURCE must be raw or xof");
    return {parse(layout?layout:"row"),source&&std::strcmp(source,"xof")==0};
}
inline bool rearranged(Mode m){return m!=Mode::row&&m!=Mode::slab_row;}
inline bool original(Options o){return o.mode==Mode::row&&!o.xof;}

template<Mode How,bool Tail> DP_HD constexpr int width(){
    if constexpr(How==Mode::row||How==Mode::slab_row||(Tail&&How==Mode::main256))return 0;
    if constexpr(How==Mode::bulk||How==Mode::tma64)return Tail?128:64;
    if constexpr(How==Mode::tma128)return 128;
    if constexpr(How==Mode::tma256||How==Mode::main256)return 256;
    return 512;
}
template<Mode How,bool Tail>
DP_HD inline std::size_t part_offset(int row,int col,int count,int k){
    constexpr int r=Tail?128:192,w=width<How,Tail>();
    if constexpr(w==0)return std::size_t(row)*k+col;
    else{
        const int first=row/r*r,within=row-first,actual=(count-first<r?count-first:r);
        const int ks=col/w*w,clen=k-ks<w?k-ks:w;
        int pos=within*clen+(col-ks);
        if constexpr(How==Mode::bulk){
            // bulk uses native-width segments (64 or 128) and stores the exact
            // per-row SMEM bank permutation before its one-dimensional copy.
            constexpr int mask=Tail?0x380:0x180;
            pos^=(pos&mask)>>3;
        }
        return std::size_t(first)*k+std::size_t(ks)*actual+pos;
    }
}
template<Mode How> DP_HD inline std::size_t offset(int row,int col,int count,int k,int split){
    return row<split?part_offset<How,false>(row,col,split,k):
                     std::size_t(split)*k+part_offset<How,true>(row-split,col,count-split,k);
}
}
#undef DP_HD
