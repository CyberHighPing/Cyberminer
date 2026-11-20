#pragma once

// Pearl certificate-v4, one keyed BLAKE3 root block per complete lottery tile.
// The numerical core is shared with the offline CPU differential tests.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define PEARL_TICKET_INLINE __host__ __device__ __forceinline__
#else
#define PEARL_TICKET_INLINE inline
#endif

namespace pearl_ticket {
using Word = std::uint32_t;
using Index = std::uint64_t;
using Count = unsigned long long;
static_assert(sizeof(Count) == 8, "CUDA atomic count must be 64-bit");

struct Job {
    Word key[8];     // role subkey, not the A-side seed
    Word bound[8];   // nonnegative U256, low word first
};

struct Slab {
    Index tile_row0;
    Index tile_col0;
    Index full_tile_cols;
    Index tile_rows;
    Index tile_cols;
};

struct Hit {
    Index tile;
    Word fold[16];
    Word digest[8];
};
static_assert(sizeof(Job) == 64 && sizeof(Hit) == 104, "bad ticket record ABI");
static_assert(offsetof(Hit, fold) == 8 && offsetof(Hit, digest) == 72, "bad field offsets");

inline bool valid_slab(const Slab& s) {
    if (s.tile_rows == 0 || s.tile_cols == 0) return true;
    const Index max = (std::numeric_limits<Index>::max)();
    if (!s.full_tile_cols || s.tile_col0 >= s.full_tile_cols ||
        s.tile_cols > s.full_tile_cols - s.tile_col0 ||
        s.tile_rows - 1 > max - s.tile_row0 ||
        s.tile_rows > (std::numeric_limits<std::size_t>::max)() / 64 / s.tile_cols)
        return false;
    const Index last_col = s.tile_col0 + s.tile_cols - 1;
    return s.tile_row0 + s.tile_rows - 1 <= (max - last_col) / s.full_tile_cols;
}

PEARL_TICKET_INLINE Index global_tile(const Slab& slab, Index local) {
    return (slab.tile_row0 + local / slab.tile_cols) * slab.full_tile_cols +
           slab.tile_col0 + local % slab.tile_cols;
}

// Check rectangular, aligned slabs online. The original scanner traverses
// column slabs first with row slabs inside, so this is the default.
class Coverage {
    Index major_total_, minor_total_, global_cols_;
    Index major_pos_ = 0, minor_pos_ = 0, current_width_ = 0, counted_ = 0;
    bool column_outer_;
public:
    Coverage(Index rows, Index cols, bool column_outer = true)
        : major_total_(column_outer ? cols : rows), minor_total_(column_outer ? rows : cols),
          global_cols_(cols), column_outer_(column_outer) {}
    bool append(const Slab& s) {
        if (!valid_slab(s) || s.full_tile_cols != global_cols_) return false;
        if (!s.tile_rows || !s.tile_cols) return true;
        const Index major = column_outer_ ? s.tile_col0 : s.tile_row0;
        const Index minor = column_outer_ ? s.tile_row0 : s.tile_col0;
        const Index width = column_outer_ ? s.tile_cols : s.tile_rows;
        const Index span = column_outer_ ? s.tile_rows : s.tile_cols;
        if (major != major_pos_ || minor != minor_pos_ || major > major_total_ ||
            minor > minor_total_ || width > major_total_-major || span > minor_total_-minor ||
            (minor && width != current_width_) ||
            width > ((std::numeric_limits<Index>::max)()-counted_)/span) return false;
        current_width_ = width;
        minor_pos_ += span;
        counted_ += width*span;
        if (minor_pos_ == minor_total_) {
            minor_pos_ = 0;
            major_pos_ += width;
        }
        return true;
    }
    bool complete() const { return !major_total_ || !minor_total_ || (major_pos_ == major_total_ && !minor_pos_); }
    Index scanned() const { return counted_; }
};

namespace detail {
template<unsigned N>
PEARL_TICKET_INLINE Word rotate_right(Word x) {
    static_assert(N > 0 && N < 32, "invalid rotate");
#ifdef __CUDA_ARCH__
    return __funnelshift_r(x, x, N);
#else
    return (x >> N) | (x << (32 - N));
#endif
}

PEARL_TICKET_INLINE void mix(Word& a, Word& b, Word& c, Word& d, Word x, Word y) {
    a += b + x; d = rotate_right<16>(d ^ a);
    c += d;     b = rotate_right<12>(b ^ c);
    a += b + y; d = rotate_right<8>(d ^ a);
    c += d;     b = rotate_right<7>(b ^ c);
}

// All schedule indices are compile-time constants. This keeps the 16 message
// words and 16 state words scalarized on CUDA, with no shared lookup table.
PEARL_TICKET_INLINE constexpr unsigned perm(unsigned n) {
    constexpr unsigned order[16] = {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};
    return order[n];
}
PEARL_TICKET_INLINE constexpr unsigned at(unsigned round, unsigned i) {
    return round ? at(round - 1, perm(i)) : i;
}

template<unsigned Round,unsigned I>
PEARL_TICKET_INLINE void diagonal(Word (&v)[16],const Word* m) {
    mix(v[I],v[4+(I+1)%4],v[8+(I+2)%4],v[12+(I+3)%4],m[at(Round,8+2*I)],m[at(Round,9+2*I)]);
}

template<unsigned R>
PEARL_TICKET_INLINE void columns(Word (&v)[16],const Word* m) {
    mix(v[0],v[4],v[8],v[12],m[at(R,0)],m[at(R,1)]);
    mix(v[1],v[5],v[9],v[13],m[at(R,2)],m[at(R,3)]);
    mix(v[2],v[6],v[10],v[14],m[at(R,4)],m[at(R,5)]);
    mix(v[3],v[7],v[11],v[15],m[at(R,6)],m[at(R,7)]);
}

template<unsigned R>
PEARL_TICKET_INLINE void round(Word (&v)[16],const Word* m) {
    columns<R>(v,m);
    diagonal<R,0>(v,m);diagonal<R,1>(v,m);
    diagonal<R,2>(v,m);diagonal<R,3>(v,m);
}
}  // namespace detail

template<bool CheckHighWord>
PEARL_TICKET_INLINE bool compress_fold(const Word* key,const Word* fold,Word* out,Word high_bound) {
    Word v[16]={key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7],
                0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0,0,64,27};
    detail::round<0>(v,fold);detail::round<1>(v,fold);detail::round<2>(v,fold);
    detail::round<3>(v,fold);detail::round<4>(v,fold);detail::round<5>(v,fold);
    detail::columns<6>(v,fold);
    // The four diagonals are disjoint. Only diagonals 0 and 2 contribute
    // to v[7] XOR v[15], the most significant word in the unsigned target.
    detail::diagonal<6,0>(v,fold);
    detail::diagonal<6,2>(v,fold);
    if constexpr(CheckHighWord){
        if((v[7]^v[15])>high_bound)return false;
    }
    detail::diagonal<6,1>(v,fold);
    detail::diagonal<6,3>(v,fold);
#ifdef __CUDACC__
#pragma unroll
#endif
    for(unsigned i=0;i<8;i++)out[i]=v[i]^v[i+8];
    return true;
}

PEARL_TICKET_INLINE void digest_fold(const Word* key,const Word* fold,Word* out) {
    compress_fold<false>(key,fold,out,0);
}

PEARL_TICKET_INLINE bool meets_bound(const Word* digest, const Word* bound) {
#ifdef __CUDACC__
#pragma unroll
#endif
    for (int i=7; i>=0; --i) {
        if (digest[i] != bound[i]) return digest[i] < bound[i];
    }
    return true;
}

// On an overflowing range, discard its tentative output. Split according to
// the observed hit count. If every tile wins the second pass already fits.
// Range intervals are left-closed/right-open; only accepted intervals commit.
template<class Probe, class Accept>
bool scan_ranges(Index total, std::size_t capacity, Probe&& probe, Accept&& accept) {
    if (!total) return true;
    if (!capacity) return false;
    struct Range { Index begin, end, step; };
    std::vector<Range> todo{{0,total,0}};
    while (!todo.empty()) {
        Range r = todo.back();
        todo.pop_back();
        if (r.step && r.end-r.begin > r.step) {
            const Index next = r.begin+r.step;
            todo.push_back({next,r.end,r.step});
            r.end = next;
        }
        Count count = 0;
        if (!probe(r.begin,r.end,count) || count > r.end-r.begin) return false;
        if (count > capacity) {
            const Index parts = 1+(count-1)/capacity;
            const Index step = 1+(r.end-r.begin-1)/parts;
            todo.push_back({r.begin,r.end,step});
        } else if (count && !accept(std::size_t(count))) {
            return false;
        }
    }
    return true;
}

#ifdef __CUDACC__
// Low-level asynchronous launch. d_count must have been zeroed on this stream.
// If *d_count > capacity, ALL records from this launch are provisional.
static __global__ void filter_range_kernel(const Word* folds, Slab slab, Job job,
                                          Index begin, Index end, Hit* hits,
                                          Count* count, std::size_t capacity) {
    const Index step = Index(blockDim.x) * gridDim.x;
    for (Index local = begin + Index(blockIdx.x)*blockDim.x + threadIdx.x;
         local < end; local += step) {
        Word fold[16], digest[8];
#pragma unroll
        for (unsigned i=0; i<16; ++i) fold[i] = folds[local*16+i];
        if (!compress_fold<true>(job.key,fold,digest,job.bound[7]) || !meets_bound(digest,job.bound))
            continue;
        const Count slot = atomicAdd(count, Count(1));
        if (slot >= capacity) continue;
        Hit& hit = hits[slot];
        hit.tile = global_tile(slab, local);
#pragma unroll
        for (unsigned i=0; i<16; ++i) hit.fold[i] = fold[i];
#pragma unroll
        for (unsigned i=0; i<8; ++i) hit.digest[i] = digest[i];
    }
}

inline cudaError_t launch_range(const Word* d_folds, const Slab& slab, const Job& job,
                                Index begin, Index end, Hit* d_hits, Count* d_count,
                                std::size_t capacity, cudaStream_t stream) {
    if (!valid_slab(slab) || end < begin || end > slab.tile_rows*slab.tile_cols ||
        !d_count || (begin != end && !d_folds) || (capacity && !d_hits))
        return cudaErrorInvalidValue;
    if (begin == end) return cudaSuccess;
    constexpr unsigned threads = 128;
    const Index blocks = 1 + (end-begin-1)/threads;
    const unsigned grid = unsigned((std::min)(blocks, Index(65535)));
    filter_range_kernel<<<grid,threads,0,stream>>>(d_folds,slab,job,begin,end,
                                                  d_hits,d_count,capacity);
    return cudaGetLastError();
}

// A complete slab commits once. A failed CUDA operation or allocation never
// calls on_hits. Keep d_folds immutable through all overflow retries.
template<class OnHits>
cudaError_t filter_slab(const Word* d_folds, const Slab& slab, const Job& job,
                        Hit* d_hits, Count* d_count, std::size_t capacity,
                        cudaStream_t stream, OnHits&& on_hits) {
    if (!valid_slab(slab)) return cudaErrorInvalidValue;
    const Index total = slab.tile_rows * slab.tile_cols;
    if (!total) return cudaSuccess;
    if (!d_folds || !d_hits || !d_count || !capacity ||
        capacity > (std::numeric_limits<std::size_t>::max)()/sizeof(Hit))
        return cudaErrorInvalidValue;

    std::vector<Hit> result;
    cudaError_t status = cudaSuccess;
    auto probe = [&](Index begin,Index end,Count& observed) {
        status = cudaMemsetAsync(d_count,0,sizeof(Count),stream);
        if (status != cudaSuccess) return false;
        status = launch_range(d_folds,slab,job,begin,end,d_hits,d_count,capacity,stream);
        if (status != cudaSuccess) return false;
        status = cudaMemcpyAsync(&observed,d_count,sizeof(observed),cudaMemcpyDeviceToHost,stream);
        if (status != cudaSuccess) return false;
        status = cudaStreamSynchronize(stream);
        return status == cudaSuccess;
    };
    auto accept = [&](std::size_t added) {
        const std::size_t old = result.size();
        if (added > result.max_size()-old) { status=cudaErrorMemoryAllocation; return false; }
        result.resize(old+added);
        status = cudaMemcpyAsync(result.data()+old,d_hits,added*sizeof(Hit),cudaMemcpyDeviceToHost,stream);
        if (status != cudaSuccess) return false;
        status = cudaStreamSynchronize(stream);
        return status == cudaSuccess;
    };
    if (!scan_ranges(total,capacity,probe,accept)) return status==cudaSuccess ? cudaErrorUnknown : status;
    std::sort(result.begin(),result.end(),[](const Hit& a,const Hit& b){return a.tile<b.tile;});
    on_hits(result.data(),result.size());
    return cudaSuccess;
}
#endif
}  // namespace pearl_ticket
#undef PEARL_TICKET_INLINE
