#pragma once

#include <cstdint>
#include <string_view>

namespace pearl_device {
struct Profile {
    std::string_view name;
    unsigned min_mib;
    unsigned max_mib;
    int strip_rows;
    bool large_scan;
    unsigned max_board_mw;
};

inline constexpr Profile profiles[] = {
    {"NVIDIA GeForce RTX 5060",     7000,  9000, 1280, false,      0},
    {"NVIDIA GeForce RTX 5060 Ti",  7000,  9000, 1536, true,       0},
    {"NVIDIA GeForce RTX 5070",    11000, 13000, 2048, false,      0},
    {"NVIDIA GeForce RTX 5070 Ti", 15000, 17000, 1536, false,      0},
    {"NVIDIA GeForce RTX 5080",    15000, 17000, 3072, false, 250000},
};

constexpr const Profile* profile_for(std::string_view name) {
    for (const auto& profile : profiles)
        if (profile.name == name) return &profile;
    return nullptr;
}

inline constexpr std::string_view select_message =
    "select a supported desktop SM120 GPU: 5060 / 5060 Ti 8 GB, 5070 12 GB, or 5070 Ti / 5080 16 GB";

constexpr bool supported(std::string_view name, int major, int minor, std::uint64_t bytes) {
    constexpr std::uint64_t mib = 1024 * 1024;
    const Profile* profile = profile_for(name);
    return profile && major == 12 && minor == 0 &&
           bytes >= profile->min_mib * mib && bytes <= profile->max_mib * mib;
}

constexpr bool supports_scan_shape(std::string_view name, int m, int n, int k) {
    const Profile* profile = profile_for(name);
    return profile && (profile->large_scan || (m <= 131072 && n <= 131072 && k <= 16384));
}

constexpr bool power_allowed(const Profile& profile, unsigned configured_mw, unsigned enforced_mw) {
    return !profile.max_board_mw ||
           (configured_mw > 0 && configured_mw <= profile.max_board_mw &&
            enforced_mw > 0 && enforced_mw <= profile.max_board_mw);
}

struct BoardLimits {
    unsigned sm_mhz;
    unsigned stop_mw;
};
inline constexpr BoardLimits conservative_5080{1500, 290000};
inline constexpr BoardLimits performance_5080{2400, 340000};

constexpr bool sample_allowed(unsigned average_mw, std::uint64_t instant_mw, unsigned sm_mhz,
                              int offset_mhz, BoardLimits limits = conservative_5080) {
    return average_mw > 0 && average_mw < limits.stop_mw && instant_mw > 0 && instant_mw < limits.stop_mw &&
           sm_mhz > 0 && sm_mhz <= limits.sm_mhz && offset_mhz == 0;
}

template<class Properties>
constexpr bool supported(const Properties& device) {
    return supported(device.name, device.major, device.minor, device.totalGlobalMem);
}
}  // namespace pearl_device
