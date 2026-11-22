// Linux, CUDA 12.8+ with SM120 support. Dense E4M3, B200 arithmetic.
// A and B are row-major [M,K] and [N,K], already transformed by Pearl.
#include <cuda_runtime.h>
#include "power_limit.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <poll.h>
#define PEARL_SCAN_INPUTS 1
#include "worker_buffer_pool.hpp"
#include "geometry_filter.hpp"
#include "ticket_wire.hpp"
#include "native_inputs.cuh"
#include "coalesced_fold.cuh"
#ifdef PEARL_WITH_CUTLASS
#include "cutlass120.cuh"
#include "verified_lt.hpp"
#include "sm120_fused_stream.cuh"
#endif

using byte = std::uint8_t;
using word = std::uint32_t;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

class DeviceMemoryError : public std::runtime_error { public: using std::runtime_error::runtime_error; };
static void check(cudaError_t status, const char* operation) {
    if (status==cudaErrorMemoryAllocation)
        throw DeviceMemoryError(std::string(operation)+": "+cudaGetErrorString(status));
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define CUDA(call) check((call), #call)

template<class T> struct Buffer {
    T* ptr = nullptr;
    pearl_pool::Meter* meter;
    explicit Buffer(size_t count, void* retained=nullptr, pearl_pool::Meter* measured=nullptr)
        : ptr(static_cast<T*>(retained)), meter(measured) {
        if (!count || ptr) return;
        if (!meter) { CUDA(cudaMalloc(&ptr,count*sizeof(T))); return; }
        const auto t=Clock::now();
        CUDA(cudaMalloc(&ptr,count*sizeof(T)));
        meter->acquire_s+=std::chrono::duration<double>(Clock::now()-t).count();
        ++meter->allocations;
    }
    void reset() noexcept {
        if (!ptr) return;
        if (!meter) { cudaFree(ptr); ptr=nullptr; return; }
        const auto t=Clock::now();
        cudaFree(ptr); ptr=nullptr;
        meter->release_s+=std::chrono::duration<double>(Clock::now()-t).count();
        ++meter->frees;
    }
    void* detach() noexcept { return std::exchange(ptr,nullptr); }
    ~Buffer() { reset(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// Ownership stays in a Buffer or in this single group, never in both. If
// checkout fails part-way, this destructor releases any not-yet-adopted slots.
struct RetainedScanPointers {
    std::array<void*,10> ptrs{};
    void* take(size_t i) { return std::exchange(ptrs[i],nullptr); }
    ~RetainedScanPointers() {
        for (auto i=ptrs.rbegin(); i!=ptrs.rend(); ++i) if (*i) cudaFree(*i);
    }
};
using WorkerBufferCache = pearl_pool::Cache<RetainedScanPointers>;

#include <thread>
struct Event {
    cudaEvent_t value;
    Event() { CUDA(cudaEventCreate(&value)); }
    explicit Event(unsigned flags) { CUDA(cudaEventCreateWithFlags(&value,flags)); }
    ~Event() { cudaEventDestroy(value); }
};
struct FoldStream {
    cudaStream_t value=nullptr;
    explicit FoldStream(bool enabled) {
        if (enabled) CUDA(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));
    }
    ~FoldStream() { if (value) cudaStreamDestroy(value); }
    FoldStream(const FoldStream&)=delete;
    FoldStream& operator=(const FoldStream&)=delete;
};
struct PhaseTime {
    Event start,mid,end;
    void add(double& gemm, double& fold, const Event* fold_start=nullptr) const {
        float ms;
        CUDA(cudaEventElapsedTime(&ms,start.value,mid.value)); gemm+=ms;
        CUDA(cudaEventElapsedTime(&ms,fold_start?fold_start->value:mid.value,end.value)); fold+=ms;
    }
};

constexpr int BM = 64, BK = 128, PanelRows = 32;
constexpr int Threads = 128;

__device__ __forceinline__ int swizzle(int row, int col) {
    return row * BK + (((col / 16) ^ (row & 7)) * 16) + col % 16;
}
__device__ __forceinline__ word shared_word(const byte* block, int r, int c) {
    return *reinterpret_cast<const word*>(block + swizzle(r, c));
}
__host__ __device__ inline size_t packed_offset(int row, int col, int k) {
    const int groups = (k + BK - 1) / BK;
    return ((size_t(row / PanelRows) * groups + col / BK) * PanelRows +
            row % PanelRows) * BK + col % BK;
}
static size_t packed_bytes(int rows, int k) {
    return size_t((rows+PanelRows-1)/PanelRows)*PanelRows*((k+BK-1)/BK)*BK;
}
template<int BN, bool Packed>
__device__ __forceinline__ void stage(byte* shared, const byte* a, const byte* b,
                                      int m, int n, int k, int pitch, int k0) {
    constexpr int vecs = BK / 16;
    for (int v = threadIdx.x; v < (BM + BN) * vecs; v += Threads) {
        const int row = v / vecs, c = (v % vecs) * 16;
        const bool left = row < BM;
        const int gr = left ? int(blockIdx.y) * BM + row : int(blockIdx.x) * BN + row - BM;
        const bool valid = gr < (left ? m : n) && k0 + c < k;
        const byte* base = left ? a : b;
        const size_t offset = Packed ? packed_offset(gr,k0+c,pitch) : size_t(gr)*pitch+k0+c;
        const byte* src = valid ? base + offset : base;
        const auto dst = static_cast<unsigned>(__cvta_generic_to_shared(shared + swizzle(row, c)));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(dst), "l"(src), "r"(valid ? 16 : 0) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int BN, int H, int W>
__device__ __forceinline__ void fold_from_shared(const word* tile, word* folded, int m, int n, int fold_cols) {
    constexpr int tile_cols=BN/W, span=H*W/16;
    constexpr int words=(BM/H)*tile_cols*16;
    for (int u=threadIdx.x; u<words; u+=Threads) {
        const int tile_row=(u/16/tile_cols)*H, tile_col=(u/16%tile_cols)*W;
        const int row=int(blockIdx.y)*BM+tile_row, col=int(blockIdx.x)*BN+tile_col;
        if (row+H>m || col+W>n) continue;
        const int start=(u%16)*span;
        word x=0;
        #pragma unroll
        for (int i=0; i<span; ++i) {
            const int r=tile_row+(start+i)/W, c=tile_col+(start+i)%W;
            x=x*0x9e3779b1u + tile[r*BN+(c^(r&31))];
            x=__funnelshift_l(x,x,13);
        }
        folded[(size_t(row/H)*fold_cols+col/W)*16+u%16]=x;
    }
}

template<int BN, bool Packed, bool MatrixLoad=false> __global__ __launch_bounds__(Threads)
void multiply(const byte* __restrict__ a, const byte* __restrict__ b,
              word* __restrict__ c, int m, int n, int k, int pitch, int first_k, int last_k,
              word* __restrict__ folded, int height, int width, int fold_cols, bool carry_in,bool final) {
    __shared__ __align__(16) byte staged[2][(BM + BN) * BK];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int warp_width = BN / 2, fragments = warp_width / 8;
    const int row0 = (warp / 2) * 32, col0 = (warp % 2) * warp_width;
    float acc[2][fragments][4] = {};
    // Resume the exact FP32 carry from the previous adjacent K interval.
    // No parallel partial sums or floating-point additions are introduced.
    if (carry_in) {
        #pragma unroll
        for (int i=0; i<2; ++i) {
            #pragma unroll
            for (int j=0; j<fragments; ++j) {
                #pragma unroll
                for (int e=0; e<4; ++e) {
                    const int row=int(blockIdx.y)*BM+row0+i*16+lane/4+(e/2)*8;
                    const int col=int(blockIdx.x)*BN+col0+j*8+(lane%4)*2+e%2;
                    if (row<m && col<n) acc[i][j][e]=__uint_as_float(c[size_t(row)*n+col]);
                }
            }
        }
    }
    stage<BN,Packed>(staged[0], a, b, m, n, k, pitch, first_k);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    for (int k0 = first_k, s = 0; k0 < last_k; k0 += BK, s ^= 1) {
        const bool next = k0 + BK < last_k;
        if (next) stage<BN,Packed>(staged[s ^ 1], a, b, m, n, k, pitch, k0 + BK);
        const byte* sa = staged[s];
        const byte* sb = sa + BM * BK;
        #pragma unroll
        for (int g = 0; g < BK / 32; ++g) {
            // An extra all-zero MMA can change the low bits of a finite carry.
            if (k0 + g * 32 >= last_k) break;
            const int pos = g * 32 + (lane % 4) * 4;
            word aa[2][4], bb[fragments][2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int row = row0 + i * 16 + lane / 4;
                if constexpr (MatrixLoad) {
                    const int r=row0+i*16+lane%16;
                    const int c=g*32+(lane/16)*16;
                    const unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(sa+swizzle(r,c)));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(aa[i][0]),"=r"(aa[i][1]),"=r"(aa[i][2]),"=r"(aa[i][3]) : "r"(addr));
                } else {
                    aa[i][0] = shared_word(sa, row, pos);
                    aa[i][1] = shared_word(sa, row + 8, pos);
                    aa[i][2] = shared_word(sa, row, pos + 16);
                    aa[i][3] = shared_word(sa, row + 8, pos + 16);
                }
            }
            #pragma unroll
            for (int j = 0; j < fragments; ++j) {
                const int row = col0 + j * 8 + lane / 4;
                if constexpr (MatrixLoad) {
                    const int r=col0+j*8+lane%8;
                    const int c=g*32+((lane/8)%2)*16;
                    const unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(sb+swizzle(r,c)));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bb[j][0]),"=r"(bb[j][1]) : "r"(addr));
                } else {
                    bb[j][0] = shared_word(sb, row, pos);
                    bb[j][1] = shared_word(sb, row, pos + 16);
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < fragments; ++j) {
                    auto& d = acc[i][j];
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                                 : "r"(aa[i][0]), "r"(aa[i][1]), "r"(aa[i][2]), "r"(aa[i][3]),
                                   "r"(bb[j][0]), "r"(bb[j][1]));
                }
            }
        }
        if (next) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
    // Reuse the operand buffers only after every warp has read the last K group.
    const bool publish=folded!=nullptr && final;
    word* final_tile=reinterpret_cast<word*>(staged);
    if (publish) __syncthreads();
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < fragments; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = int(blockIdx.y) * BM + row0 + i * 16 + lane / 4 + (e / 2) * 8;
                const int col = int(blockIdx.x) * BN + col0 + j * 8 + (lane % 4) * 2 + e % 2;
                if (row < m && col < n) {
                    const word bits = __float_as_uint(acc[i][j][e]);
                    const word value=final && (bits & 0x7fffffffu)>0x7f800000u ? 0x7fc00000u : bits;
                    if (publish) {
                        const int local_row=row%BM, local_col=col%BN;
                        final_tile[local_row*BN+(local_col^(local_row&31))]=value;
                    } else {
                        c[size_t(row)*n+col]=value;
                    }
                }
            }
        }
    }
    if (publish) {
        __syncthreads();
        if (height==16 && width==16) fold_from_shared<BN,16,16>(final_tile,folded,m,n,fold_cols);
        if (height==16 && width==32) fold_from_shared<BN,16,32>(final_tile,folded,m,n,fold_cols);
        if (height==4 && width==64) fold_from_shared<BN,4,64>(final_tile,folded,m,n,fold_cols);
        if constexpr (BN>=128)
            if (height==4 && width==128) fold_from_shared<BN,4,128>(final_tile,folded,m,n,fold_cols);
    }
}

// The layout is committed before computing the product. The current protocol
// assigns contiguous row-major cells to each lane. An explicit permutation
// also allows a caller to verify another committed layout against its oracle.
__global__ void fold_tiles(const word* matrix, int n, int height, int width,
                           const std::uint16_t* lanes, size_t count, word* out, int fold_cols) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const size_t ticket = idx / 16, tickets_per_row = n / width;
    const size_t base = (ticket / tickets_per_row * height) * n + ticket % tickets_per_row * width;
    const int cells = height * width / 16, first = int(idx % 16) * cells;
    word x = 0;
    for (int i = 0; i < cells; ++i) {
        const int p = lanes[first + i];
        x = x * 0x9e3779b1u + matrix[base + (p / width) * n + p % width];
        x = __funnelshift_l(x, x, 13);
    }
    out[((ticket/tickets_per_row)*fold_cols+ticket%tickets_per_row)*16+idx%16]=x;
}

// Deterministic finite E4M3 bytes for measuring the scan only. This path has
// no header or proof and cannot produce a submit-ready candidate.
__global__ void bench_input(byte* out, size_t elems, size_t base, word salt, int k, int pitch, bool packed) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elems) return;
    const size_t id = base + i;
    word x = word(id) ^ (word(id >> 32) * 0x9e3779b9u) ^ salt;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    const size_t where=packed ? packed_offset(int(i/k), int(i%k), pitch) :
                         (pitch==k ? i : (i/k)*pitch + i%k);
    out[where] = byte((x & 128u) | ((x >> 8) % 127u));
}
__global__ void pack_vectors(const uint4* raw, uint4* dest, size_t vectors, int k, int pitch, bool packed) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=vectors) return;
    const size_t byte_offset=i*16;
    const size_t where=packed ? packed_offset(int(byte_offset/k),int(byte_offset%k),pitch) :
                                (byte_offset/k)*pitch + byte_offset%k;
    dest[where/16]=raw[i];
}
__global__ void pack_k_segments(const uint4* input,uint4* out,size_t vectors,int rows,int k,int segment) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=vectors)return;
    const size_t q=i*16;
    const int row=int(q/k), col=int(q%k);
    out[((size_t(col/segment)*rows+row)*segment+col%segment)/16]=input[i];
}

struct Options {
    int m=0, n=0, k=0, h=4, w=128, rows=512, cols=8192, ks=-1, device=0, hit_capacity=256;
    unsigned bench_seed=0;
    int row_pad=0;
    std::string a, b, out, dump, lanes, variant="auto", params, native_job, roots_out;
    bool bench=false, progress=false, resident=false, fused=false, deferred=false, k_major=false, stream_b=false;
};
static int number(const std::string& s, int low, int high) {
    int v=0;
    const auto value=std::from_chars(s.data(),s.data()+s.size(),v);
    if (value.ec!=std::errc() || value.ptr!=s.data()+s.size() || v<low || v>high)
        throw std::runtime_error("invalid integer: " + s);
    return v;
}
static Options parse(int argc, char** argv) {
    Options o;
    for (int i=1; i<argc; ++i) {
        std::string key=argv[i];
        if (key=="--bench") { o.bench=true; continue; }
        if (key=="--progress") { o.progress=true; continue; }
        if (key=="--resident") { o.resident=true; continue; }
        if (key=="--fuse-fold") { o.fused=true; continue; }
        if (key=="--defer-filter") { o.deferred=true; continue; }
        if (key=="--stream-b") { o.stream_b=true; continue; }
        if (key=="--k-major") { o.k_major=true; continue; }
        if (++i >= argc) throw std::runtime_error("missing value for " + key);
        std::string v=argv[i];
        if (key=="--m") o.m=number(v,1,1048576);
        else if (key=="--n") o.n=number(v,1,1048576);
        else if (key=="--k") o.k=number(v,32,65536);
        else if (key=="--tile") {
            const size_t sep=v.find('x');
            if (sep==std::string::npos) throw std::runtime_error("expected HxW");
            o.h=number(v.substr(0,sep),1,4096); o.w=number(v.substr(sep+1),1,4096);
        } else if (key=="--rows-per-pass") o.rows=number(v,1,1048576);
        else if (key=="--cols-per-pass") o.cols=number(v,1,1048576);
        else if (key=="--k-per-pass") o.ks=number(v,0,65536);
        else if (key=="--row-pad") o.row_pad=number(v,0,4096);
        else if (key=="--device") o.device=number(v,0,256);
        else if (key=="--a") o.a=v;
        else if (key=="--b") o.b=v;
        else if (key=="--out") o.out=v;
        else if (key=="--dump-c") o.dump=v;
        else if (key=="--lanes") o.lanes=v;
        else if (key=="--variant") o.variant=v;
        else if (key=="--ticket-params") o.params=v;
        else if (key=="--native-job") o.native_job=v;
        else if (key=="--roots-out") o.roots_out=v;
        else if (key=="--hit-capacity") o.hit_capacity=number(v,1,65536);
        else if (key=="--bench-seed") o.bench_seed=unsigned(number(v,0,std::numeric_limits<int>::max()));
        else throw std::runtime_error("unknown option: " + key);
    }
    if (o.stream_b && o.ks>0 && o.ks!=o.k)
        throw std::invalid_argument("stream-B requires the complete K in each multiplication");
    o.rows=std::min(o.rows,o.m); o.cols=std::min(o.cols,o.n);
    if (o.ks<0) o.ks=o.k>=8192 ? 4096 : 0;
    if (o.ks && o.ks%128) throw std::runtime_error("K pass size must be a multiple of 128");
    o.ks=o.ks?std::min(o.ks,o.k):o.k;
    if (o.row_pad%16) throw std::runtime_error("row padding must be a multiple of 16 bytes");
    if (o.variant=="auto") o.variant=o.k>=8192?"row128":"row64";
    if (o.bench && !o.params.empty()) throw std::runtime_error("ticket parameters require prepared file inputs");
    if (!o.m || !o.n || !o.k || o.k%32 || o.m<o.h || o.n<o.w ||
        (o.rows<o.m && o.rows%o.h) || (o.cols<o.n && o.cols%o.w) ||
        o.h*o.w%16 || o.h*o.w>65536)
        throw std::runtime_error("invalid shape, tile, or pass alignment");
    if (!o.native_job.empty()) {
        if (o.bench || !o.a.empty() || !o.b.empty() || !o.params.empty() || o.out.empty() ||
            o.roots_out.empty() || o.roots_out==o.out || o.roots_out==o.dump || !o.lanes.empty())
            throw std::runtime_error("native jobs require distinct --out and --roots-out and use the committed plain layout");
        o.resident=true;
    } else if (!o.roots_out.empty() ||
               (o.bench ? (!o.a.empty() || !o.b.empty()) : (o.a.empty() || o.b.empty() || o.out.empty())))
        throw std::runtime_error("supply --a, --b and --out for mining, or --bench for synthetic input");
    if (!o.dump.empty() && size_t(o.m)*o.n > (1u<<20))
        throw std::runtime_error("--dump-c is for small arithmetic comparisons");
    if (!o.out.empty() && o.out==o.dump) throw std::runtime_error("output paths must differ");
    bool known=o.variant=="row64" || o.variant=="row128" || o.variant=="ld64" || o.variant=="ld128" ||
               o.variant=="packed64" || o.variant=="packed128";
#ifdef PEARL_WITH_CUTLASS
    if(o.variant==pearl_fused_stream::name){
        if(!o.stream_b||o.h!=16||o.w!=32)throw std::invalid_argument("sm120-fused-stream requires --stream-b and 16x32 tickets");
        known=true;o.ks=o.k;
    }
    if (pearl_cutlass::recognized(o.variant)) { known=true; o.ks=o.k; }
    if (o.variant==verified_lt::name) {
        known=true;
        if (!o.row_pad && !o.k_major) o.ks=o.k;
    }
#endif
    if (!known)
        throw std::runtime_error("unknown SM120 kernel variant");
    return o;
}

struct Input {
    std::ifstream file;
    std::vector<byte> scratch;
    Input(const std::string& name, size_t size, size_t capacity) {
        if (name.empty()) return;
        if (fs::file_size(name)!=size) throw std::runtime_error("wrong byte count: " + name);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(name,std::ios::binary);
        scratch.resize(std::min(capacity,size_t(4)*1024*1024));
    }
    void load(byte* gpu, byte* raw, size_t offset, size_t length, int k, int pitch, word salt, bool packed) {
        if (!file.is_open()) {
            bench_input<<<(length+255)/256,256>>>(gpu,length,offset,salt,k,pitch,packed);
            CUDA(cudaGetLastError());
            return;
        }
        file.seekg(offset);
        for (size_t done=0; done<length;) {
            const size_t chunk=std::min(scratch.size(),length-done);
            file.read(reinterpret_cast<char*>(scratch.data()),chunk);
            CUDA(cudaMemcpy(((packed || pitch!=k)?raw:gpu)+done,scratch.data(),chunk,cudaMemcpyHostToDevice));
            done+=chunk;
        }
        if (packed || pitch!=k) {
            const size_t vectors=length/16;
            pack_vectors<<<(vectors+255)/256,256>>>(reinterpret_cast<const uint4*>(raw),
                                reinterpret_cast<uint4*>(gpu),vectors,k,pitch,packed);
            CUDA(cudaGetLastError());
        }
    }
};

struct Output {
    std::string target, temp;
    std::ofstream file;
    explicit Output(const std::string& name) : target(name) {
        if (name.empty()) return;
        if (fs::exists(name)) throw std::runtime_error("output exists: "+name);
        temp=name+".part."+std::to_string(getpid());
        if (fs::exists(temp)) throw std::runtime_error("temporary output exists: "+temp);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.open(temp,std::ios::binary);
    }
    void write(size_t word_offset, const word* data, size_t count) {
        if (!file.is_open()) return;
        file.seekp(word_offset*sizeof(word));
        file.write(reinterpret_cast<const char*>(data),count*sizeof(word));
    }
    void finish() {
        if (!file.is_open()) return;
        file.close();
        // Same-directory hard link publishes only if the requested output is still new.
        fs::create_hard_link(temp,target);
        fs::remove(temp);
        temp.clear();
    }
    ~Output() {
        if (!temp.empty()) { std::error_code ec; fs::remove(temp,ec); }
    }
};

#ifdef PEARL_WITH_CUTLASS
#include "column_stream.cuh"
#include "right_run.cuh"
#include "middle_reference.cuh"
#endif

static int scan_attempt(const Options& o, WorkerBufferCache* pool, bool allow_extended, bool& may_retry) {
    std::vector<std::uint16_t> order(size_t(o.h)*o.w);
    std::iota(order.begin(),order.end(),0);
    if (!o.lanes.empty()) {
        if (fs::file_size(o.lanes)!=order.size()*2) throw std::runtime_error("wrong lane table length");
        std::ifstream in(o.lanes,std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(order.data()),order.size()*2))
            throw std::runtime_error("cannot read lane table");
        auto sorted=order;
        std::sort(sorted.begin(),sorted.end());
        for (size_t i=0; i<sorted.size(); ++i)
            if (sorted[i]!=i) throw std::runtime_error("lane table must be a permutation");
    }
    const size_t ac=size_t(o.rows)*o.k, bc=size_t(o.cols)*o.k;
    const size_t cells=size_t(o.rows)*o.cols, fold_count=cells/order.size()*16;
    const bool packed=o.variant.rfind("packed",0)==0;
    const bool matrix_load=o.variant.rfind("ld",0)==0;
    const int pitch=o.k+o.row_pad;
    const bool lt_requested=o.variant==verified_lt::name;
    const bool lt_layout=lt_requested && !o.row_pad && !o.k_major;
    bool vendor=false;
#ifdef PEARL_WITH_CUTLASS
    vendor=pearl_cutlass::recognized(o.variant) || lt_layout;
#endif
    if (vendor && o.row_pad) throw std::runtime_error("CUTLASS path requires unpadded rows");
    if (o.k_major && (!o.resident || packed || vendor || o.row_pad))
        throw std::runtime_error("K-major packing requires resident native MMA with unpadded rows");
    const bool k_major=o.k_major && o.ks<o.k;
    const int bn=(o.variant.find("128")!=std::string::npos || (lt_requested && o.k>=8192)) ? 128 : 64;
    const bool fold_shape = (o.h==16 && (o.w==16 || o.w==32)) ||
                            (o.h==4 && (o.w==64 || o.w==128));
    const bool fused=!vendor && o.fused && fold_shape && o.w<=bn && o.dump.empty() &&
                     std::is_sorted(order.begin(),order.end());
    if (o.resident && packed && ((o.rows<o.m && o.rows%PanelRows) ||
                                 (o.cols<o.n && o.cols%PanelRows)))
        throw std::runtime_error("resident packed passes must start on 32-row boundaries");
    const int a_rows=o.resident?o.m:o.rows, b_rows=o.resident?o.n:o.cols;
    const int stored_k=k_major?((o.k+o.ks-1)/o.ks)*o.ks:pitch;
    const size_t gpu_a=packed?packed_bytes(a_rows,pitch):size_t(a_rows)*stored_k;
    const size_t gpu_b=packed?packed_bytes(b_rows,pitch):size_t(b_rows)*stored_k;
    std::unique_ptr<native_inputs::Plan> native;
    constexpr int prep_rows=256;
    if (!o.native_job.empty()) native=std::make_unique<native_inputs::Plan>(o.native_job,o.m,o.n,o.k,o.h,o.w);
    const size_t reorder_bytes=(packed || o.row_pad)&&!o.bench?std::max(ac,bc):0;
    const size_t segment_temp=k_major?size_t(std::max(a_rows,b_rows))*o.k:0;
    const size_t scratch_bytes=std::max({reorder_bytes,native?native->scratch_bytes(prep_rows):0,segment_temp});
    const size_t acc_cells=fused && o.ks==o.k ? 0 : cells;
    const std::size_t capacity=o.params.empty()&&!native?0:std::size_t(o.hit_capacity);
    const size_t total_folds=size_t(o.m/o.h)*(o.n/o.w)*16;
    const size_t num_passes=size_t((o.m+o.rows-1)/o.rows)*((o.n+o.cols-1)/o.cols);
    const bool defer_ok=o.deferred && o.resident && o.dump.empty() && num_passes<=8192;
    const bool original_deferred=defer_ok && total_folds<=pearl_geometry::legacy_words;
    bool deferred=original_deferred;
    const bool extended_candidate=pearl_geometry::expandable(defer_ok && capacity!=0,lt_layout,
        o.m,o.n,o.k,o.h,o.w,o.rows,o.cols,num_passes,total_folds);
    const bool swizzled=lt_layout || o.variant=="cutlass120-coop-swizzle" || o.variant=="cutlass120-coop-swizzle-pipeline";
    const bool fast_fold=vendor && (o.variant=="cutlass120-coop-fold" || o.variant=="cutlass120-coop-pipeline" || swizzled) &&
                         o.h==16 && o.w==32 && std::is_sorted(order.begin(),order.end());
    const bool may_overlap=fast_fold && (lt_layout || o.variant=="cutlass120-coop-pipeline" || o.variant=="cutlass120-coop-swizzle-pipeline") && num_passes>=2;
    bool want_overlap=may_overlap && deferred;
    size_t fold_capacity=deferred?total_folds:fold_count;
    size_t bytes=gpu_a+gpu_b+scratch_bytes+4*acc_cells+4*fold_capacity+2*order.size()+
                       capacity*sizeof(pearl_ticket::Hit)+(capacity?sizeof(pearl_ticket::Count):0)+
                       (native?sizeof(unsigned):0);
    if (bytes>size_t(7168)*1024*1024) throw std::runtime_error("choose smaller pass dimensions");
    Input ia(o.a,size_t(o.m)*o.k,ac), ib(o.b,size_t(o.n)*o.k,bc);
    Output output(o.out), dump(o.dump), roots(o.roots_out);
    std::unique_ptr<pearl_ticket::SparseWire> tickets;
    pearl_ticket::Coverage coverage(o.m/o.h,o.n/o.w);
    if (!o.params.empty()) {
        std::ifstream in(o.params,std::ios::binary);
        if (!in) throw std::runtime_error("cannot read ticket parameters");
        tickets=std::make_unique<pearl_ticket::SparseWire>(in,o.m,o.n,o.k,o.h,o.w);
        tickets->begin(output.file);
    }
    std::optional<pearl_pool::Shape> shape,extended_shape;
    pearl_pool::Meter meter;
    if (pool) {
        meter=pool->previous_idle_release();
        if (native && o.dump.empty()) {
            const unsigned flags = unsigned(o.resident) | (unsigned(fused)<<1) | (unsigned(deferred)<<2)
                | (unsigned(k_major)<<3) | (unsigned(want_overlap)<<4);
            shape.emplace(pearl_pool::Shape{o.device,
                {o.m,o.n,o.k,o.h,o.w,o.rows,o.cols,o.ks,o.row_pad,o.hit_capacity},
                {gpu_a,gpu_b,scratch_bytes,acc_cells,fold_capacity,capacity,size_t(capacity!=0),1,order.size()},
                flags,o.variant});
        }
        // Preserve either exact layout on this device until free memory decides
        // which one we will use. Other devices and jobs are discarded here.
        if (allow_extended && extended_candidate && shape) {
            extended_shape=*shape;
            extended_shape->capacities[4]=total_folds;
            extended_shape->flags|=(1u<<2) | (unsigned(may_overlap)<<4);
        }
        if (!extended_shape || !pool->matches(*extended_shape)) pool->match(shape ? &*shape : nullptr,meter);
    }
    CUDA(cudaSetDevice(o.device));
    cudaDeviceProp dev;
    CUDA(cudaGetDeviceProperties(&dev,o.device));
#ifdef PEARL_WITH_CUTLASS
    pearl_cutlass::Candidate candidate(o.device,dev.multiProcessorCount);
#endif
    verified_lt::Plan lt_plan;
    if (lt_layout && (o.rows==4096 || o.m%o.rows==4096) && (o.cols==16384 || o.n%o.cols==16384)) {
        lt_plan.prepare(4096,16384,o.k);
        if (!lt_plan.ready()) lt_plan.clear();
    }
    size_t available0,total;
    CUDA(cudaMemGetInfo(&available0,&total));
    const size_t retained_before=shape?pool->retained():0;
    auto credit=[&]{return shape?pool->admission_available(available0,total):available0;};
    if (lt_plan.ready() && scratch_bytes<verified_lt::offered_workspace) {
        const auto required=bytes+verified_lt::offered_workspace;
        if (!pearl_pool::admit(required,0,false,credit()).fits) lt_plan.pressure();
        else lt_plan.reserve_dedicated();
        CUDA(cudaMemGetInfo(&available0,&total));
    }
    // raw free already excludes a successful dedicated allocation. Add it to
    // both sides so the original 7168 MiB cap includes it exactly once.
    const size_t original_bytes=bytes;
    bool extended=false;
    auto choose=[&]{
        const size_t extra=lt_plan.extra_bytes(),available=credit()+extra;
        extended=allow_extended && extended_candidate && lt_plan.ready() &&
            pearl_geometry::fits_full(original_bytes,fold_count,total_folds,extra,available);
        deferred=original_deferred || extended;
        want_overlap=may_overlap && deferred;
        fold_capacity=deferred?total_folds:fold_count;
        bytes=original_bytes+(extended ? 4*(total_folds-fold_count) : 0);
        return pearl_pool::admit(bytes+extra,4*acc_cells,want_overlap,available);
    };
    auto decision=choose();
    if (lt_plan.ready() && !decision.fits) {
        lt_plan.pressure();
        CUDA(cudaMemGetInfo(&available0,&total));
        decision=choose();
    }
    if (shape && pool->has() && (!decision.fits || decision.overlap!=pool->overlap() ||
        !pool->matches(extended?*extended_shape:*shape))) {
        pool->drop(meter);
        CUDA(cudaMemGetInfo(&available0,&total));
        decision=choose();
    }
    if (!decision.fits) throw std::runtime_error("insufficient free VRAM for this pass size");
    if (extended && shape) shape=std::move(extended_shape);
    may_retry=extended;
    const bool overlap=decision.overlap;
    const std::size_t explicit_bytes_at_entry=decision.bytes;
    bytes=decision.bytes-lt_plan.extra_bytes(); // only these slots may remain cached
    std::unique_ptr<RetainedScanPointers> owner;
    if (shape && pool->reuse) {
        meter.warm=pool->has();
        owner=meter.warm ? pool->take() : std::make_unique<RetainedScanPointers>();
    }
    auto slot=[&](size_t i)->void* { return owner?owner->take(i):nullptr; };
    auto* measured=shape?&meter:nullptr;
    Buffer<byte> a(gpu_a,slot(0),measured), b(gpu_b,slot(1),measured), staging(scratch_bytes,slot(2),measured);
    Buffer<word> c(acc_cells,slot(3),measured), alternate(overlap?acc_cells:0,slot(4),measured), folded(fold_capacity,slot(5),measured);
    FoldStream fold_stream(overlap);
    Buffer<pearl_ticket::Hit> hits(capacity,slot(6),measured);
    Buffer<pearl_ticket::Count> hit_count(capacity!=0,slot(7),measured);
    Buffer<unsigned> bad_input(bool(native)?1:0,slot(8),measured);
    Buffer<std::uint16_t> lanes(order.size(),slot(9),measured);
    lt_plan.bind_workspace(staging.ptr,scratch_bytes);
    CUDA(cudaMemcpy(lanes.ptr,order.data(),2*order.size(),cudaMemcpyHostToDevice));
    size_t available1;
    CUDA(cudaMemGetInfo(&available1,&total));
    const size_t read_words=deferred?std::min(total_folds,size_t(1)<<18):fold_count;
    std::vector<word> host_folds(capacity?0:read_words);
    std::vector<word> debug(o.dump.empty()?0:cells);
    std::vector<PhaseTime> phases(deferred?num_passes:1);
    std::vector<Event> fold_starts(overlap?num_passes:0);
    may_retry=false; // all scan storage is ready; never replay a running request
    double gemm_ms=0, fold_ms=0, ticket_wall_s=0;
    auto filter=[&](const word* values,const pearl_ticket::Slab& slab) {
        const auto then=Clock::now();
        CUDA(pearl_ticket::filter_slab(values,slab,tickets->job,hits.ptr,hit_count.ptr,capacity,
            nullptr,[&](pearl_ticket::Hit* records,size_t size) {
                if (extended) {
                    pearl_geometry::append_window_order(records,size,o.n/o.w,o.cols/o.w,
                        [&](const auto* data,size_t n){tickets->append(output.file,data,n);});
                } else tickets->append(output.file,records,size);
            }));
        ticket_wall_s+=std::chrono::duration<double>(Clock::now()-then).count();
    };
    size_t passes=0;
    std::uint64_t checksum=0;
    const auto t0=Clock::now();
    auto last_tick=t0;
    const word salt_a=0xc31a5e71u + 0x9e3779b9u*o.bench_seed;
    const word salt_b=0xa5b357d1u + 0x7f4a7c15u*o.bench_seed;
    auto operand_offset=[&](int row) {
        return packed ? packed_offset(row,0,pitch) : size_t(row)*pitch;
    };
    native_inputs::State native_state;
    if (native) {
        native_state=native->materialize(a.ptr,b.ptr,staging.ptr,scratch_bytes,bad_input.ptr,prep_rows);
        // Work backward so that padding or panel packing cannot overwrite
        // the unconverted prefix of the same resident operand.
        if (reorder_bytes) {
            auto reorder=[&](byte* data,int rows,int step) {
                for (int r=(rows-1)/step*step; r>=0; r-=step) {
                    const size_t size=size_t(std::min(step,rows-r))*o.k;
                    CUDA(cudaMemcpy(staging.ptr,data+size_t(r)*o.k,size,cudaMemcpyDeviceToDevice));
                    pack_vectors<<<(size/16+255)/256,256>>>(reinterpret_cast<const uint4*>(staging.ptr),
                        reinterpret_cast<uint4*>(data+operand_offset(r)),size/16,o.k,pitch,packed);
                    CUDA(cudaGetLastError());
                }
            };
            reorder(a.ptr,o.m,o.rows);reorder(b.ptr,o.n,o.cols);
            CUDA(cudaDeviceSynchronize());
        }
        std::istringstream header(native->ticket_header(native_state));
        tickets=std::make_unique<pearl_ticket::SparseWire>(header,o.m,o.n,o.k,o.h,o.w);
        tickets->begin(output.file);
        const auto receipt=native_state.receipt();
        roots.file.write(reinterpret_cast<const char*>(receipt.data()),receipt.size());
    } else if (o.resident) {
        auto fill=[&](Input& input, byte* gpu, int rows, int step, word salt) {
            for (int r=0; r<rows; r+=step)
                input.load(gpu+operand_offset(r),staging.ptr,size_t(r)*o.k,
                           size_t(std::min(step,rows-r))*o.k,o.k,pitch,salt,packed);
        };
        fill(ia,a.ptr,o.m,o.rows,salt_a);
        fill(ib,b.ptr,o.n,o.cols,salt_b);
        CUDA(cudaDeviceSynchronize());
    }
    double repack_s=0;
    if(k_major) {
        const auto start=Clock::now();
        auto reorder=[&](byte* data,int rows) {
            const size_t length=size_t(rows)*o.k;
            CUDA(cudaMemcpy(staging.ptr,data,length,cudaMemcpyDeviceToDevice));
            pack_k_segments<<<(length/16+255)/256,256>>>(reinterpret_cast<const uint4*>(staging.ptr),
                     reinterpret_cast<uint4*>(data),length/16,rows,o.k,o.ks);
            CUDA(cudaGetLastError());
        };
        reorder(a.ptr,o.m);reorder(b.ptr,o.n);
        CUDA(cudaDeviceSynchronize());
        repack_s=std::chrono::duration<double>(Clock::now()-start).count();
    }
    const double preload_s=std::chrono::duration<double>(Clock::now()-t0).count();
    for (int col0=0; col0<o.n; col0+=o.cols) {
        int n=std::min(o.cols,o.n-col0);
        const byte* db=b.ptr+(o.resident?operand_offset(col0):0);
        if (!o.resident) ib.load(b.ptr,staging.ptr,size_t(col0)*o.k,size_t(n)*o.k,o.k,pitch,salt_b,packed);
        for (int row0=0; row0<o.m; row0+=o.rows) {
            const int m=std::min(o.rows,o.m-row0);
            const byte* da=a.ptr+(o.resident?operand_offset(row0):0);
            if (!o.resident && (col0==0 || o.rows<o.m))
                ia.load(a.ptr,staging.ptr,size_t(row0)*o.k,size_t(m)*o.k,o.k,pitch,salt_a,packed);
            PhaseTime& timing=phases[deferred?passes:0];
            word* const product=overlap && (passes&1)?alternate.ptr:c.ptr;
            if(overlap && passes>=2) CUDA(cudaStreamWaitEvent(nullptr,phases[passes-2].end.value,0));
            CUDA(cudaEventRecord(timing.start.value));
            const dim3 grid((n+bn-1)/bn,(m+BM-1)/BM);
            const size_t count=size_t(m/o.h)*(n/o.w)*16;
            const int fold_cols=deferred?o.n/o.w:n/o.w;
            const size_t first_word=deferred && count?(size_t(row0/o.h)*fold_cols+col0/o.w)*16:0;
            word* dest=folded.ptr+first_word;
            word* fused_result=fused?dest:nullptr;
#ifdef PEARL_WITH_CUTLASS
            bool produced=false;
            if (lt_requested) {
                if (lt_layout) produced=lt_plan.run(da,db,product,m,n,o.k);
                if (!produced && lt_layout && n%4==0) {
                    try {
                        candidate.run("cutlass120-coop",da,db,product,m,n,o.k);
                        lt_plan.record_cutlass();
                        produced=true;
                    } catch (const pearl_cutlass::CannotImplement&) {
                        // Odd matrix alignment can require the existing native reference.
                    }
                }
                if (!produced) lt_plan.record_native();
            } else if (vendor) {
                candidate.run(o.variant,da,db,product,m,n,o.k);
                produced=true;
            }
            if (!produced)
#endif
            for (int first=0; first<o.k; first+=o.ks) {
                const int last=std::min(o.k, first+o.ks);
                const byte* ka=k_major?a.ptr+(size_t(first/o.ks)*o.m+row0)*o.ks:da;
                const byte* kb=k_major?b.ptr+(size_t(first/o.ks)*o.n+col0)*o.ks:db;
                const int len=k_major?last-first:o.k, ld=k_major?o.ks:pitch;
                const int lo=k_major?0:first, hi=k_major?len:last;
                const bool carry=first>0, final=last==o.k;
                if (bn==64 && !packed && !matrix_load) multiply<64,false><<<grid,Threads>>>(ka,kb,product,m,n,len,ld,lo,hi,fused_result,o.h,o.w,fold_cols,carry,final);
                if (bn==64 && matrix_load) multiply<64,false,true><<<grid,Threads>>>(ka,kb,product,m,n,len,ld,lo,hi,fused_result,o.h,o.w,fold_cols,carry,final);
                if (bn==64 && packed) multiply<64,true><<<grid,Threads>>>(ka,kb,product,m,n,len,ld,lo,hi,fused_result,o.h,o.w,fold_cols,carry,final);
                if (bn==128 && !packed && !matrix_load) multiply<128,false><<<grid,Threads>>>(ka,kb,product,m,n,len,ld,lo,hi,fused_result,o.h,o.w,fold_cols,carry,final);
                if (bn==128 && matrix_load) multiply<128,false,true><<<grid,Threads>>>(ka,kb,product,m,n,len,ld,lo,hi,fused_result,o.h,o.w,fold_cols,carry,final);
                if (bn==128 && packed) multiply<128,true><<<grid,Threads>>>(ka,kb,product,m,n,len,ld,lo,hi,fused_result,o.h,o.w,fold_cols,carry,final);
            }
            CUDA(cudaGetLastError());
            CUDA(cudaEventRecord(timing.mid.value));
            if(overlap) {
                CUDA(cudaStreamWaitEvent(fold_stream.value,timing.mid.value,0));
                CUDA(cudaEventRecord(fold_starts[passes].value,fold_stream.value));
            }
            if (count && !fused) {
                if(fast_fold) {
                    if(swizzled) CUDA(pearl_fold::launch16x32_swizzled(product,m,n,dest,fold_cols,fold_stream.value));
                    else CUDA(pearl_fold::launch16x32(product,m,n,dest,fold_cols,fold_stream.value));
                } else {
                    fold_tiles<<<(count+255)/256,256>>>(product,n,o.h,o.w,lanes.ptr,count,dest,fold_cols);
                }
                CUDA(cudaGetLastError());
            }
            CUDA(cudaEventRecord(timing.end.value,fold_stream.value));
            if (!deferred) {
                CUDA(cudaEventSynchronize(timing.end.value));
                timing.add(gemm_ms,fold_ms);
            }
            if (tickets) {
                const pearl_ticket::Slab slab{size_t(row0/o.h),size_t(col0/o.w),
                                               size_t(o.n/o.w),size_t(m/o.h),size_t(n/o.w)};
                if (!coverage.append(slab)) throw std::runtime_error("invalid GPU ticket coverage");
                if (!deferred) filter(folded.ptr,slab);
            } else if (!deferred && count) CUDA(cudaMemcpy(host_folds.data(),folded.ptr,4*count,cudaMemcpyDeviceToHost));
            for (int row=0; !deferred && !tickets && count && row<m/o.h; ++row) {
                const size_t width=size_t(n/o.w)*16;
                const size_t first=(size_t(row0/o.h+row)*(o.n/o.w)+col0/o.w)*16;
                const word* data=host_folds.data()+row*width;
                output.write(first,data,width);
                for (size_t i=0; i<width; ++i)
                    checksum+=(std::uint64_t(data[i]) ^ (first+i))*((first+i)*2+1);
            }
            if (!debug.empty()) {
                CUDA(cudaMemcpy(debug.data(),c.ptr,size_t(m)*n*4,cudaMemcpyDeviceToHost));
                for (int row=0; row<m; ++row)
                    dump.write(size_t(row0+row)*o.n+col0,debug.data()+size_t(row)*n,n);
            }
            ++passes;
            if (o.progress && (passes==1 || Clock::now()-last_tick>std::chrono::seconds(3))) {
                last_tick=Clock::now();
                std::cerr<<"pass="<<passes<<" row="<<row0<<" col="<<col0
                         <<" gemm_ms_total="<<gemm_ms<<" elapsed_s="
                         <<std::chrono::duration<double>(last_tick-t0).count()<<std::endl;
            }
        }
    }
    if (deferred) {
        CUDA(cudaEventSynchronize(phases.back().end.value));
        for (size_t i=0;i<phases.size();++i)
            phases[i].add(gemm_ms,fold_ms,overlap?&fold_starts[i]:nullptr);
        if (tickets) {
            filter(folded.ptr,{0,0,size_t(o.n/o.w),size_t(o.m/o.h),size_t(o.n/o.w)});
        } else {
            for (size_t first=0; first<total_folds; first+=read_words) {
                const size_t count=std::min(read_words,total_folds-first);
                CUDA(cudaMemcpy(host_folds.data(),folded.ptr+first,4*count,cudaMemcpyDeviceToHost));
                output.write(first,host_folds.data(),count);
                for (size_t i=0;i<count;++i)
                    checksum+=(std::uint64_t(host_folds[i])^(first+i))*((first+i)*2+1);
            }
        }
    }
    if (tickets) tickets->finish(output.file,coverage);
    roots.finish(); output.finish(); dump.finish();
    const double elapsed=std::chrono::duration<double>(Clock::now()-t0).count();
    if (lt_requested) lt_plan.clear();
    if (shape) {
        // Every supported path has completed its phase events and host copy or
        // sparse filter by here. The next job regenerates all source operands.
        if (pool->reuse && !tickets->hits()) {
            owner->ptrs={a.detach(),b.detach(),staging.detach(),c.detach(),alternate.detach(),
                         folded.detach(),hits.detach(),hit_count.detach(),bad_input.detach(),lanes.detach()};
            pool->keep(std::move(*shape),overlap,bytes,std::move(owner));
        } else {
            // A hit may immediately launch a separate GPU prover process.
            // Free all held scan storage before acknowledging the scan.
            lanes.reset(); bad_input.reset(); hit_count.reset(); hits.reset(); folded.reset();
            alternate.reset(); c.reset(); staging.reset(); b.reset(); a.reset();
        }
    }
    const std::uint64_t macs=std::uint64_t(o.m)*o.n*o.k;
    const std::uint64_t eligible=std::uint64_t(o.m/o.h)*o.h*(o.n/o.w)*o.w*o.k;
    std::cout<<std::dec<<std::setprecision(10)<<"{\"device\":\""<<dev.name<<"\",\"mode\":\""
             <<(native?"native_job":o.bench?"synthetic_scan":"file_scan")<<"\",\"variant\":\""<<o.variant
             <<"\",\"m\":"<<o.m<<",\"n\":"<<o.n
             <<",\"k\":"<<o.k<<",\"tile_h\":"<<o.h<<",\"tile_w\":"<<o.w
             <<",\"rows_per_pass\":"<<o.rows<<",\"cols_per_pass\":"<<o.cols<<",\"k_per_pass\":"<<o.ks
             <<",\"resident_operands\":"<<(o.resident?"true":"false")<<",\"preload_s\":"<<preload_s
             <<",\"gemm_includes_fold\":"<<(fused?"true":"false")
             <<",\"sparse_filter\":"<<(tickets?"true":"false")<<",\"ticket_wall_s\":"<<ticket_wall_s
             <<",\"deferred_filter\":"<<(deferred?"true":"false")
             <<",\"coalesced_fold\":"<<(fast_fold?"true":"false")<<",\"overlapped_fold\":"<<(overlap?"true":"false")
             <<",\"generate_commit_s\":"<<native_state.generation_s
             <<",\"noise_quantize_s\":"<<native_state.quantization_s
             <<",\"threshold_hits\":"<<(tickets?tickets->hits():0)
             <<",\"benchmark_seed\":"<<o.bench_seed
             <<",\"passes\":"<<passes<<",\"explicit_gpu_bytes\":"<<explicit_bytes_at_entry
             <<",\"row_pitch_bytes\":"<<pitch
             <<",\"k_major\":"<<(k_major?"true":"false")<<",\"layout_repack_s\":"<<repack_s
             <<",\"free_before_buffers\":"<<available0<<",\"free_after_buffers\":"<<available1
             <<",\"device_total_bytes\":"<<total<<",\"gemm_ms\":"<<gemm_ms
             <<",\"fold_ms\":"<<fold_ms<<",\"scan_wall_s\":"<<elapsed
             <<",\"kernel_tmac_s\":"<<double(macs)/gemm_ms/1e9
             <<",\"scan_equivalent_th_s\":"<<double(eligible)/elapsed/1e12
             <<",\"tiles\":"<<size_t(o.m/o.h)*(o.n/o.w)
             <<",\"unpaired_rows\":"<<o.m%o.h<<",\"unpaired_cols\":"<<o.n%o.w
             <<",\"fold_checksum\":";
    if (tickets) std::cout<<"null";
    else std::cout<<'\"'<<std::hex<<std::setw(16)<<std::setfill('0')<<checksum<<'\"';
    if (lt_requested) {
        std::cout<<std::dec; lt_plan.write(std::cout);
        std::cout<<",\"extended_deferred_filter\":"<<(extended?"true":"false")
                 <<",\"fold_buffer_bytes\":"<<4*fold_capacity
                 <<",\"fold_allocation_retry\":"<<(!allow_extended?"true":"false");
    }
    if (shape) std::cout<<std::dec<<",\"buffer_pool_mode\":\""<<(pool->reuse?"reuse":"measure")<<"\""
        <<",\"buffer_pool_warm\":"<<(meter.warm?"true":"false")
        <<",\"buffer_pool_before_bytes\":"<<retained_before<<",\"buffer_pool_after_bytes\":"<<pool->retained()
        <<",\"buffer_malloc_calls\":"<<meter.allocations<<",\"buffer_free_calls\":"<<meter.frees
        <<",\"buffer_malloc_s\":"<<meter.acquire_s<<",\"buffer_free_s\":"<<meter.release_s;
    std::cout<<"}\n";
    return 0;
}

static int scan(const Options& o,WorkerBufferCache* pool=nullptr,pearl_device::WorkerPowerReader* power_session=nullptr) {
    cudaDeviceProp dev{};
    CUDA(cudaGetDeviceProperties(&dev, o.device));
    if (!pearl_device::supported(dev))
        throw std::runtime_error(std::string(pearl_device::select_message));
    pearl_device::PowerWatch board_guard;
    board_guard.start(*pearl_device::profile_for(dev.name), o.device, power_session);
    if (!pearl_device::supports_scan_shape(dev.name, o.m, o.n, o.k))
        throw std::invalid_argument("this board supports up to the testnet2 scan dimensions in this build");
#ifdef PEARL_WITH_CUTLASS
    if(o.stream_b && o.variant==verified_lt::name)return panel_reference::run(o);
    if(o.stream_b && right_resident::enabled())return right_resident::run(o,pool);
    if(o.stream_b)return column_stream::run(o,pool);
#else
    if(o.stream_b)throw std::invalid_argument("stream-B requires a CUTLASS-enabled build");
#endif
    bool may_retry=false;
    try { return scan_attempt(o,pool,true,may_retry); }
    catch (const DeviceMemoryError&) {
        if (!may_retry) throw;
        const auto pending=cudaGetLastError();
        if (pending!=cudaSuccess && pending!=cudaErrorMemoryAllocation) CUDA(pending);
    }
    // All temporary files and device owners from the first attempt have
    // unwound. Retry the identical control once using its original window.
    return scan_attempt(o,pool,false,may_retry);
}

static word frame_word() {
    word n=0;
    if (!std::cin.read(reinterpret_cast<char*>(&n),sizeof(n)))
        throw std::runtime_error("truncated worker request");
    return n;
}

// A single trusted local pipe keeps one CUDA process alive across independent
// jobs. Each request still supplies its own input files, target and output.
static int serve() {
    pearl_device::WorkerPowerReader power_session;
    const char* env=std::getenv("PEARL_SCAN_BUFFER_POOL");
    const std::string mode=env?env:"off";
    WorkerBufferCache cache(mode=="reuse");
    WorkerBufferCache* pool=(cache.reuse || mode=="measure")?&cache:nullptr;
    // Keep poll() consistent with the binary frame reader even for pipelined
    // frames. Only an explicitly opted-in reuse worker changes stdin buffering.
    if (cache.reuse && std::setvbuf(stdin,nullptr,_IONBF,0)!=0) pool=nullptr;
    std::cout<<"PEARL-V4-SCAN/1\n"<<std::flush;
    while (true) {
        if (pool && cache.has()) {
            pollfd input{STDIN_FILENO,POLLIN,0};
            if (::poll(&input,1,1000)<=0) cache.drop_idle();
        }
        if (std::cin.peek()==std::char_traits<char>::eof()) break;
        const auto count=frame_word();
        if (!count) return 0;
        if (count>96) throw std::runtime_error("too many worker arguments");
        std::vector<std::string> fields{"pearl-fp8-scan"};
        size_t total=0;
        for (word i=0;i<count;++i) {
            const word bytes=frame_word();
            total+=bytes;
            if (!bytes || bytes>16384 || total>65536) throw std::runtime_error("invalid worker argument length");
            std::string s(bytes,'\0');
            if (!std::cin.read(s.data(),bytes) || s.find('\0')!=std::string::npos)
                throw std::runtime_error("invalid worker argument");
            fields.push_back(std::move(s));
        }
        std::vector<char*> args;
        for (auto& s:fields) args.push_back(s.data());
        scan(parse(int(args.size()),args.data()),pool,&power_session);
        std::cout.flush();
    }
    return 0;
}
int main(int argc,char** argv) {
    if (argc==2 && std::string(argv[1])=="--help") {
        std::cout<<"pearl-fp8-scan --m M --n N --k K --a A.e4m3 --b B.e4m3 --out folds.bin\n"
                 <<"  [--tile HxW] [--lanes lanes.u16le] [--device I] [--rows-per-pass R]\n"
                 <<"  [--cols-per-pass C] [--k-per-pass KSTEP] [--resident] [--variant NAME]\n"
                 <<"  [--fuse-fold] [--ticket-params job.params] [--hit-capacity N]\n"
                 <<"  [--progress] [--dump-c small-c.bin]\n"
                 <<"  [--native-job CONTROL --roots-out ROOTS --stream-b --defer-filter]\n"
                 <<"Use --bench [--bench-seed N] in place of inputs for arithmetic timing.\n";
        return 0;
    }
    try {
        if (argc==2 && (std::string_view(argv[1])=="--device-profile" || std::string_view(argv[1])=="--benchmark-preset")) {
            cudaDeviceProp dev{};
            CUDA(cudaGetDeviceProperties(&dev, 0));
            if (!pearl_device::supported(dev))
                throw std::invalid_argument(std::string(pearl_device::select_message));
            const auto& profile = *pearl_device::profile_for(dev.name);
            pearl_device::verify_power(profile, 0);
            const char* preset = profile.large_scan ? "throughput" : "testnet2";
            if (std::string_view(argv[1])=="--device-profile")
                std::cout << preset << ' ' << profile.strip_rows << '\n';
            else std::cout << preset << '\n';
            return 0;
        }
        if (argc==2 && std::string(argv[1])=="--worker") return serve();
        return scan(parse(argc,argv));
    }
    catch (const pearl_device::PowerLimitError& e) { std::cerr << e.what() << '\n'; return 78; }
    catch (const std::exception& e) { std::cerr<<"pearl-fp8-scan: "<<e.what()<<'\n'; return 1; }
}
