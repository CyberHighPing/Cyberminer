// Build the ordinary dense certificate-v4 Merkle openings for one completed tile.
#include "job_generator.cuh"
#include "panel_b.cuh"
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using Byte = std::uint8_t;
using Size = std::uint64_t;
using Word = std::uint32_t;
using Digest = pearl_jobgen::Digest;
using Bytes = std::vector<Byte>;
constexpr Size Chunk = 1024;
constexpr Size MaxProof = 4u << 20;

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
static void gpu(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> class Buffer {
public:
    T* data = nullptr;
    explicit Buffer(Size n) {
        require(n && n <= SIZE_MAX / sizeof(T), "invalid GPU allocation size");
        gpu(cudaMalloc(reinterpret_cast<void**>(&data), n*sizeof(T)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

static Bytes read_file(const fs::path& path, Size limit) {
    const auto n = fs::file_size(path);
    require(n > 0 && n <= limit, "unexpected input file size");
    Bytes data(n);
    std::ifstream in(path, std::ios::binary);
    require(bool(in.read(reinterpret_cast<char*>(data.data()), n)), "cannot read input");
    return data;
}
static Word u32(const Bytes& b, size_t i) {
    return Word(b.at(i)) | Word(b.at(i+1))<<8 | Word(b.at(i+2))<<16 | Word(b.at(i+3))<<24;
}
static void append64(Bytes& out, Size x) {
    for (unsigned i=0; i<8; ++i) out.push_back(Byte(x>>(8*i)));
}
static void append(Bytes& out, const void* data, size_t n) {
    if (n) {
        const auto* p = static_cast<const Byte*>(data);
        out.insert(out.end(), p, p+n);
    }
}
static void blob(Bytes& out, const void* data, size_t n) {
    append64(out,n); append(out,data,n);
}
static Size divide_up(Size n, Size d) { return n/d + (n%d != 0); }
static bool equal(const Digest& a, const Digest& b) { return std::memcmp(&a,&b,sizeof(a))==0; }

struct Control {
    Bytes data;
    Word m,n,k,h,w;
    size_t ancestors;
    unsigned depth;
    Digest seed,key_a,key_b;
    explicit Control(const fs::path& p):data(read_file(p,828)) {
        static constexpr char domain[]="pearl/v4/sparse-job/v1";
        static constexpr Byte pin[]={0xd7,0x97,0xc3,0x07,0x0a,0x17,0xe1,0xd0,0xfc,0x1b,0x30,0x34,0x6a,0x07,0x3a,0x7d,0xe2,0xb0,0xeb,0x23};
        require(data.size()>=504 && !std::memcmp(data.data(),"PRLNF8J1",8), "invalid V4 control");
        m=u32(data,8); n=u32(data,12); k=u32(data,16); h=u32(data,20); w=u32(data,24);
        require(m && n && m<=1048576 && n<=1048576 && k>=1024 && k<=65536 && k%32==0 &&
                h>=4 && w>=16 && Size(h)*w>=256 && Size(h)*w<=2048 && Size(h)*w%16==0 &&
                h<=m && w<=n, "unsupported V4 geometry");
        size_t pos=252;
        require(u32(data,248)==data.size()-pos && !std::memcmp(&data[pos],domain,sizeof(domain)), "invalid job domain");
        pos+=sizeof(domain);
        require(!std::memcmp(&data[pos],pin,sizeof(pin)), "wrong protocol revision");
        pos+=sizeof(pin);
        require(u32(data,pos)==4 && !std::memcmp(&data[pos+4],&data[8],20), "control context has different geometry");
        depth=data.at(pos+24+76); ancestors=pos+24+76+1;
        require(depth>=1 && depth<=4 && ancestors+108*depth==data.size(), "incomplete ancestor chain");
        require(u32(data,156)==m && data[160]==3 && u32(data,161)==h && u32(data,165)==k &&
                data[169]==32 && data[170]==0 && data[171]==0 && data[172]==1 &&
                u32(data,173)==n && data[177]==3 && u32(data,178)==w && data[182]==0 && data[183]==0,
                "inconsistent dense operand parameters");
        seed=pearl_jobgen::from_bytes(&data[28]);
        key_a=pearl_jobgen::from_bytes(&data[60]); key_b=pearl_jobgen::from_bytes(&data[92]);
    }
};

struct Opening {
    Size start,rows,first_leaf,total_leaves;
    Digest root{};
    Bytes data;
    std::vector<Digest> siblings;
    void encode(Bytes& out) const {
        const Size count=data.size()/Chunk;
        append64(out,count);
        for (Size i=0;i<count;++i) blob(out,data.data()+i*Chunk,Chunk);
        append64(out,count);
        for (Size i=0;i<count;++i) append64(out,first_leaf+i);
        append64(out,total_leaves); append(out,&root,sizeof(root)); append64(out,siblings.size());
        append(out,siblings.data(),siblings.size()*sizeof(Digest));
        append64(out,rows);
        for (Size i=0;i<rows;++i) append64(out,start+i);
    }
};

template<bool Scales>
static Opening open_plane(Byte* raw, Size size, Size stride, Size start, Size rows, const Digest& key,
                          Digest* level, Digest* next, Digest* saved,
                          const pearl_jobgen::Plan* panel_plan=nullptr,const Digest* seed=nullptr,pearl_jobgen::Side side=pearl_jobgen::Side::B) {
    namespace gen=pearl_jobgen::detail;
    const Size count=divide_up(size,Chunk), first=start*stride/Chunk, end=divide_up((start+rows)*stride,Chunk);
    require(first<end && end<=count && (end-first)*Chunk<=MaxProof, "opening outside the dense operand");
    Opening opening{start,rows,first,count};
    opening.data.resize((end-first)*Chunk,0);
    const auto valid=std::min<Size>(opening.data.size(),size-first*Chunk);
    if constexpr (Scales) {
        for (Size i=0;i<valid;++i) opening.data[i]=(i&1)?0x3f:0x80;
    } else {
        if(panel_plan){
            require(seed && size==panel_plan->value_bytes,"inconsistent operand opening plan");
            gpu(pearl_panel_b::generate_bytes(raw,pearl_panel_b::panel_capacity(*panel_plan),valid,first*Chunk,*seed,nullptr,side));
            gpu(cudaMemcpy(opening.data.data(),raw,valid,cudaMemcpyDeviceToHost));
        } else gpu(cudaMemcpy(opening.data.data(),raw+first*Chunk,valid,cudaMemcpyDeviceToHost));
    }
    if(panel_plan){
        require(!Scales && seed,"only raw values use panel addressing");
        gpu(pearl_panel_b::value_leaves(raw,pearl_panel_b::panel_capacity(*panel_plan),*panel_plan,*seed,key,level,nullptr,nullptr,nullptr,side));
    } else {
        gen::leaf_kernel<Scales><<<gen::blocks(count),gen::threads>>>(raw,size,count,key,level);
        gpu(cudaGetLastError());
    }
    Size left=first,right=end,length=count,nsaved=0;
    // Contiguous rows expose a contiguous interval at every tree level.
    while (length>1) {
        if (left&1) gpu(cudaMemcpyAsync(saved+nsaved++,level+left-1,sizeof(Digest),cudaMemcpyDeviceToDevice));
        if ((right&1) && right<length) gpu(cudaMemcpyAsync(saved+nsaved++,level+right,sizeof(Digest),cudaMemcpyDeviceToDevice));
        Size parents=divide_up(length,2);
        gen::parents_kernel<<<gen::blocks(parents),gen::threads>>>(level,length,key,next);
        gpu(cudaGetLastError());
        std::swap(level,next); left/=2; right=divide_up(right,2); length=parents;
    }
    gpu(cudaMemcpy(&opening.root,level,sizeof(Digest),cudaMemcpyDeviceToHost));
    opening.siblings.resize(nsaved);
    if (nsaved) gpu(cudaMemcpy(opening.siblings.data(),saved,nsaved*sizeof(Digest),cudaMemcpyDeviceToHost));
    return opening;
}

struct Operand {
    Opening values,scales;
    Size gpu_bytes;
    bool panel;
};
static Operand open_operand(const Control& cfg, pearl_jobgen::Side side, Size start, const Digest& expected) {
    const bool is_a=side==pearl_jobgen::Side::A;
    const Size rows=is_a?cfg.m:cfg.n, opened=is_a?cfg.h:cfg.w;
    const auto& key=is_a?cfg.key_a:cfg.key_b;
    pearl_jobgen::Plan plan;
    require(pearl_jobgen::make_plan(rows,cfg.k,plan),"invalid operand size");
    const Size cap=plan.value_chunks;
    const Size cv_bytes=(cap+divide_up(cap,2)+128)*sizeof(Digest);
    const bool panel=pearl_panel_b::enabled(pearl_panel_b::selected(),plan.value_bytes+cv_bytes);
    const Size raw_bytes=panel?pearl_panel_b::panel_capacity(plan):plan.value_bytes;
    const Size needed=raw_bytes+cv_bytes;
    size_t free_bytes,total;
    gpu(cudaMemGetInfo(&free_bytes,&total));
    require(needed<(Size(7)<<30) && needed+(Size(128)<<20)<free_bytes,"insufficient available VRAM to construct this opening");
    Buffer<Byte> raw(raw_bytes);
    Buffer<Digest> level(cap),next(divide_up(cap,2)),saved(128);
    if(!panel)gpu(pearl_jobgen::launch_generate(raw.data,raw_bytes,plan,side,cfg.seed,nullptr));
    auto values=open_plane<false>(raw.data,plan.value_bytes,cfg.k,start,opened,key,level.data,next.data,saved.data,
                                  panel?&plan:nullptr,panel?&cfg.seed:nullptr,side);
    auto scales=open_plane<true>(nullptr,plan.scale_bytes,cfg.k/4,start,opened,key,level.data,next.data,saved.data);
    require(equal(pearl_jobgen::fold_pair(values.root,scales.root,key,27),expected),"reconstructed Merkle root differs from scanned operand");
    return {std::move(values),std::move(scales),needed,panel};
}

static void publish(const fs::path& out, const Bytes& data) {
    require(!fs::exists(out),"proof output already exists");
    std::string tmp=out.string()+".part.XXXXXX";
    int fd=::mkstemp(tmp.data());
    require(fd>=0,"cannot create proof output");
    try {
        size_t at=0;
        while (at<data.size()) {
            auto n=::write(fd,data.data()+at,data.size()-at);
            if (n<0 && errno==EINTR) continue;
            require(n>0,"proof write failed"); at+=size_t(n);
        }
        int err=::close(fd); fd=-1; require(!err,"proof close failed");
        require(!::link(tmp.c_str(),out.c_str()),"cannot publish to a new proof pathname");
    } catch (...) {
        if (fd>=0) ::close(fd);
        ::unlink(tmp.c_str()); throw;
    }
    ::unlink(tmp.c_str());
}
int main(int argc,char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--help") {
            std::cout<<"pearl-fp8-open --control control.bin --roots roots.bin --tile-index I --out new.plain.bin\n";
            return 0;
        }
        fs::path control,roots,out;
        Size index=std::numeric_limits<Size>::max();
        require(argc==9,"supply --control, --roots, --tile-index and --out");
        for (int i=1;i<argc;i+=2) {
            const std::string flag=argv[i],value=argv[i+1];
            if (flag=="--control" && control.empty()) control=value;
            else if (flag=="--roots" && roots.empty()) roots=value;
            else if (flag=="--out" && out.empty()) out=value;
            else if (flag=="--tile-index" && index==std::numeric_limits<Size>::max()) {
                auto p=std::from_chars(value.data(),value.data()+value.size(),index);
                require(p.ec==std::errc() && p.ptr==value.data()+value.size(),"invalid tile index");
            } else throw std::runtime_error("duplicate or unknown argument");
        }
        require(!out.empty() && !fs::exists(out),"a new output file is required");
        Control cfg(control);
        const Size cols=cfg.n/cfg.w, tiles=Size(cfg.m/cfg.h)*cols;
        require(index<tiles,"tile index out of bounds");
        const auto receipt=read_file(roots,136);
        require(receipt.size()==136 && !std::memcmp(receipt.data(),"PRLNF8R1",8),"invalid scan root receipt");
        cudaDeviceProp dev{}; gpu(cudaSetDevice(0)); gpu(cudaGetDeviceProperties(&dev,0));
        require(std::string(dev.name).find("RTX 5060 Ti")!=std::string::npos && dev.major==12 && dev.minor==0 &&
                dev.totalGlobalMem>=(Size(7000)<<20) && dev.totalGlobalMem<=(Size(9000)<<20),"select a 5060 Ti 8GB");
        auto a=open_operand(cfg,pearl_jobgen::Side::A,(index/cols)*cfg.h,pearl_jobgen::from_bytes(receipt.data()+8));
        auto b=open_operand(cfg,pearl_jobgen::Side::B,(index%cols)*cfg.w,pearl_jobgen::from_bytes(receipt.data()+40));
        Bytes raw,transcript;
        append(transcript,cfg.data.data()+cfg.ancestors+(cfg.depth-1)*108,108);
        append(transcript,&cfg.data[165],19); append(transcript,&cfg.data[156],9);
        blob(raw,transcript.data(),transcript.size());
        blob(raw,cfg.data.data()+cfg.ancestors,(cfg.depth-1)*108);
        a.values.encode(raw); b.values.encode(raw); a.scales.encode(raw); b.scales.encode(raw);
        raw.push_back(0);
        publish(out,raw);
        std::array<Byte,SHA256_DIGEST_LENGTH> sha{};
        SHA256(raw.data(),raw.size(),sha.data());
        const char hex[]="0123456789abcdef"; std::string digest;
        for (Byte v:sha) { digest+=hex[v>>4]; digest+=hex[v&15]; }
        std::cout<<"{\"tile\":"<<index<<",\"proof_bytes\":"<<raw.size()<<",\"proof_sha256\":\""<<digest
                 <<"\",\"b_commit_mode\":\""<<(b.panel?"panel":"whole")<<"\",\"a_gpu_bytes\":"<<a.gpu_bytes
                 <<",\"b_gpu_bytes\":"<<b.gpu_bytes<<",\"peak_explicit_gpu_bytes\":"<<std::max(a.gpu_bytes,b.gpu_bytes)<<"}\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr<<"pearl-fp8-open: "<<err.what()<<'\n'; return 1;
    }
}
