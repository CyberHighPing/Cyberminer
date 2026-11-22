#pragma once
#include "job_generator.cuh"
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace proof_checkpoints {
using pearl_jobgen::Digest;
using pearl_jobgen::Size;
constexpr unsigned cut_levels = 6;
inline Size width(Size count) {
    for (unsigned d=0; d<cut_levels && count>1; ++d) count=(count+1)/2;
    return count;
}
inline bool enabled() {
    const char* value=std::getenv("PEARL_PROOF_CHECKPOINTS");
    if (!value || !*value || !std::strcmp(value,"0")) return false;
    if (!std::strcmp(value,"1")) return true;
    throw std::invalid_argument("PEARL_PROOF_CHECKPOINTS must be 0 or 1");
}
struct Plane {
    Size leaves=0;
    std::vector<Digest> nodes;
    void prepare(Size count) { leaves=count; nodes.resize(width(count)); }
};
using Planes=std::array<Plane,4>; // A values, B values, A scales, B scales
}
