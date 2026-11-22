#pragma once

#include <cstdint>
#include "prep_inputs.cuh"

namespace finite_mix {

// Native row preparation builds both factors and row coefficients from finite
// integer inputs. Their FP32 mixed values are finite. RN/SATFINITE therefore
// already encodes the complete E4M3 byte, including the sign of a rounded zero.
__device__ __forceinline__ std::uint16_t finish_pair(
        float low, float high, bool strip, pearl_prep::Row row) {
    unsigned short encoded;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;"
        : "=h"(encoded) : "f"(high), "f"(low));
    const unsigned pair = encoded;
    const unsigned magnitudes = pair & 0x7f7fu;
    // u + 128 - t is in [1,255] for u,t in [0,127]. No byte may
    // overflow into the next; bit 7 is clear exactly when u < t.
    const unsigned below = ~(magnitudes + (128u - row.threshold) * 0x0101u);
    unsigned replace;
    asm("prmt.b32 %0,%1,0,0x0098;" : "=r"(replace) : "r"(below));
    const unsigned mask = strip ? replace : 0;
    const unsigned replacement = (pair & 0x8787u) | (row.strip | 1u) * 0x0101u;
    return static_cast<std::uint16_t>(pair ^ ((pair ^ replacement) & mask));
}

}
