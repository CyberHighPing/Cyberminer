// Exact single Blackwell FP8 MMA group for row-major E[M,32] and F[K,32].
// Inputs are E4M3 values expressed in units of 2^-9, with a stored exponent per value.
// Build: c++ -O3 -std=c++17 -fPIC -shared -fopenmp fast_prepare.cpp -o fast_prepare.native.so
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
constexpr int Rank = 32;

uint32_t fp_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float as_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint8_t e4m3(uint32_t bits) {
    const uint32_t mag = bits & 0x7fffffffu;
    uint32_t code;
    if (mag < 0x3c800000u) {
        code = fp_bits(as_float(mag) + 16384.0f) - 0x46800000u;
    } else {
        const uint32_t rounded = mag + 0x7ffffu + ((mag >> 20) & 1u);
        code = (rounded >> 20) - 960u;
        if (code > 126u) code = 126u;
    }
    return static_cast<uint8_t>(code | ((bits >> 24) & 0x80u));
}

uint32_t fp32_rtz(int64_t sum) {
    if (sum == 0) return 0;
    const uint32_t sign = sum < 0 ? 0x80000000u : 0u;
    const uint64_t magnitude = static_cast<uint64_t>(sum < 0 ? -sum : sum);
    const int msb = 63 - __builtin_clzll(magnitude);
    const uint32_t mantissa = static_cast<uint32_t>(
        msb > 23 ? magnitude >> (msb - 23) : magnitude << (23 - msb));
    // Each product is in units of 2^-18. This range cannot overflow or underflow FP32.
    return sign | (static_cast<uint32_t>(msb - 18 + 127) << 23) | (mantissa & 0x7fffffu);
}

template<bool PlainDot>
uint32_t cell(const int32_t* a, const int8_t* exp_a, const int32_t* b, const int8_t* exp_b) {
    int64_t discarded_mask = 0;
    if constexpr (!PlainDot) {
        int anchor = -128;
        for (int p = 0; p < Rank; ++p) {
            const int term = static_cast<int>(exp_a[p]) + exp_b[p];
            if (term > anchor) anchor = term;
        }
        // The accumulation grid is 2^(anchor-25), or 2^(anchor-7) product units.
        if (anchor > 7) discarded_mask = (int64_t(1) << (anchor - 7)) - 1;
    }
    int64_t sum = 0;
    for (int p = 0; p < Rank; ++p) {
        int64_t product = static_cast<int64_t>(a[p]) * b[p];
        if constexpr (!PlainDot) {
            // Drop each product's low bits towards zero before adding the 32 terms.
            product = (product + (product < 0 ? discarded_mask : 0)) & ~discarded_mask;
        }
        sum += product;
    }
    return fp32_rtz(sum);
}

template<bool PlainDot>
void multiply(const int32_t* a, const int8_t* exp_a, const int32_t* b, const int8_t* exp_b,
              size_t m, size_t n, uint32_t* out, int threads) {
    // A 32-row E batch is typical; split B into short runs to reuse each E row.
    constexpr size_t Block = 64;
    const size_t blocks = (n + Block - 1) / Block;
    const auto jobs = static_cast<long long>(m * blocks);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1 && m * n >= 32768)
#endif
    for (long long job = 0; job < jobs; ++job) {
        const size_t r = static_cast<size_t>(job) / blocks;
        const size_t begin = (static_cast<size_t>(job) % blocks) * Block;
        const size_t end = begin + Block < n ? begin + Block : n;
        for (size_t c = begin; c < end; ++c) {
            out[r * n + c] = cell<PlainDot>(a + r * Rank, exp_a + r * Rank,
                                           b + c * Rank, exp_b + c * Rank);
        }
    }
}

template<bool PlainDot>
int write_mixed(const int32_t* a, const int8_t* exp_a, const int32_t* b, const int8_t* exp_b,
                 const int8_t* codes, const float* alpha, const float* beta, const uint8_t* strip,
                 const uint8_t* threshold, size_t m, size_t n, uint8_t* out, int threads) {
    constexpr size_t Block = 64;
    const size_t blocks = (n + Block - 1) / Block;
    const auto jobs = static_cast<long long>(m * blocks);
    int finite = 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&:finite) if(threads > 1 && m * n >= 32768)
#endif
    for (long long job = 0; job < jobs; ++job) {
        const size_t row = static_cast<size_t>(job) / blocks;
        const size_t begin = (static_cast<size_t>(job) % blocks) * Block;
        const size_t end = begin + Block < n ? begin + Block : n;
        bool rewrite = false;
        for (size_t col = begin; col < end; ++col) {
            const float noise = as_float(cell<PlainDot>(a + row * Rank, exp_a + row * Rank,
                                                       b + col * Rank, exp_b + col * Rank));
            const float signal = alpha[row] * static_cast<float>(codes[row * n + col]);
            const float mixed = static_cast<float>(static_cast<double>(noise) * beta[row] + signal);
            const uint32_t bits = fp_bits(mixed);
            finite &= ((bits & 0x7fffffffu) < 0x7f800000u);
            if ((col & 3) == 0) rewrite = ((bits >> 8) & 127u) < 112u;
            uint8_t raw = e4m3(bits);
            if (rewrite && (raw & 127u) < threshold[row]) {
                raw = static_cast<uint8_t>((raw & 128u) | strip[row] | (raw & 7u) | 1u);
            }
            out[row * n + col] = raw;
        }
    }
    return finite;
}
}

extern "C" void pearl_rank32_product(const int32_t* a, const int8_t* exp_a,
                                    const int32_t* b, const int8_t* exp_b,
                                    size_t m, size_t n, uint32_t* out, int exact_dot, int threads) {
    if (exact_dot) multiply<true>(a, exp_a, b, exp_b, m, n, out, threads);
    else multiply<false>(a, exp_a, b, exp_b, m, n, out, threads);
}

extern "C" int pearl_rank32_quantize(const int32_t* a, const int8_t* exp_a,
                                     const int32_t* b, const int8_t* exp_b,
                                     const int8_t* codes, const float* alpha, const float* beta,
                                     const uint8_t* strip, const uint8_t* threshold, size_t m, size_t n,
                                     uint8_t* out, int exact_dot, int threads) {
    if (exact_dot) return write_mixed<true>(a, exp_a, b, exp_b, codes, alpha, beta, strip, threshold, m, n, out, threads);
    return write_mixed<false>(a, exp_a, b, exp_b, codes, alpha, beta, strip, threshold, m, n, out, threads);
}

extern "C" int pearl_encode_f32(const float* in, uint8_t* out, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        const uint32_t bits = fp_bits(in[i]);
        if ((bits & 0x7fffffffu) >= 0x7f800000u) return 0;
        out[i] = e4m3(bits);
    }
    return 1;
}
