#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace pearl_pool {

using Clock = std::chrono::steady_clock;
constexpr std::size_t reserve = std::size_t(128) << 20;
constexpr std::size_t ceiling = std::size_t(7168) << 20;

struct Shape {
    int device;
    std::array<int, 10> geometry;  // M,N,K,H,W,pass-M,pass-N,pass-K,row pad,hit capacity
    std::array<std::size_t, 9> capacities;  // original typed buffer element counts, excluding alternate
    unsigned flags;
    std::string variant;

    bool operator==(const Shape& other) const {
        return device == other.device && geometry == other.geometry &&
               capacities == other.capacities && flags == other.flags && variant == other.variant;
    }

    unsigned allocation_count(bool alternate) const {
        return static_cast<unsigned>(std::count_if(capacities.begin(), capacities.end(),
            [](std::size_t n) { return n != 0; })) + unsigned(alternate && capacities[3]);
    }
};

struct Decision {
    bool overlap;
    std::size_t bytes;
    bool fits;
};

inline Decision admit(std::size_t base, std::size_t extra, bool want_overlap, std::size_t available) {
    const bool room = available >= reserve && base <= available - reserve;
    const bool overlap = want_overlap && base <= ceiling && extra <= ceiling-base &&
                         room && extra <= available-reserve-base;
    return {overlap, base+(overlap ? extra : 0), base<=ceiling && room};
}

struct Meter {
    unsigned allocations=0, frees=0;
    double acquire_s=0.0, release_s=0.0;
    bool warm=false;
};

// Owns at most one *complete* buffer group. Jobs take ownership before any
// launches: if a job throws, ordinary RAII releases its entire group. No
// buffer pointer, content, event, plan, seed or ticket crosses device IDs.
template<class Buffers>
class Cache {
    std::unique_ptr<Buffers> saved_;
    std::optional<Shape> shape_;
    std::size_t bytes_=0;
    bool overlap_=false;
    Meter deferred_;

public:
    explicit Cache(bool reuse): reuse(reuse) {}
    const bool reuse;
    bool has() const { return bool(saved_); }
    bool matches(const Shape& wanted) const { return saved_ && *shape_==wanted; }
    bool overlap() const { return overlap_; }
    std::size_t retained() const { return bytes_; }
    std::size_t admission_available(std::size_t raw, std::size_t total) const {
        return raw + (total>raw ? std::min(bytes_,total-raw) : 0);
    }
    Meter previous_idle_release() {
        const Meter prev=deferred_;
        deferred_={};
        return prev;
    }
    void drop(Meter& meter) {
        if (!saved_) return;
        const auto t=Clock::now();
        meter.frees+=shape_->allocation_count(overlap_);
        saved_.reset();
        meter.release_s+=std::chrono::duration<double>(Clock::now()-t).count();
        shape_.reset();
        bytes_=0;
        overlap_=false;
    }
    void drop_idle() { drop(deferred_); }
    void match(const Shape* wanted, Meter& meter) {
        if (saved_ && (!wanted || !(*shape_==*wanted))) drop(meter);
    }
    std::unique_ptr<Buffers> take() {
        bytes_=0;
        overlap_=false;
        shape_.reset();
        return std::move(saved_);
    }
    void keep(Shape shape, bool overlap, std::size_t bytes, std::unique_ptr<Buffers> live) {
        shape_=std::move(shape);
        overlap_=overlap;
        bytes_=bytes;
        saved_=std::move(live);
    }
};

}  // namespace pearl_pool
