#pragma once

#include "supported_gpu.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <string>
#include <utility>

namespace pearl_device {
inline constexpr char power_message[] =
    "RTX 5080 board readings do not satisfy the selected power and clock profile";
class PowerLimitError : public std::runtime_error {
public:
    PowerLimitError() : std::runtime_error(power_message) {}
};

class NvmlPowerReader {
    void* library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    nvmlDevice_t board_{};
    decltype(&nvmlShutdown) shutdown_ = nullptr;
    decltype(&nvmlDeviceGetPowerManagementLimit) requested_ = nullptr;
    decltype(&nvmlDeviceGetEnforcedPowerLimit) enforced_ = nullptr;
    decltype(&nvmlDeviceGetFieldValues) fields_ = nullptr;
    decltype(&nvmlDeviceGetPowerUsage) average_ = nullptr;
    decltype(&nvmlDeviceGetClockInfo) clock_ = nullptr;
    decltype(&nvmlDeviceGetGpcClkVfOffset) offset_ = nullptr;
    BoardLimits limits_ = conservative_5080;
    std::chrono::steady_clock::time_point above_320_since_{};

    template<class F>
    F function(const char* name) const {
        if (!library_) throw PowerLimitError();
        void* symbol = dlsym(library_, name);
        if (!symbol) throw PowerLimitError();
        return reinterpret_cast<F>(symbol);
    }

public:
    NvmlPowerReader() = default;
    NvmlPowerReader(const NvmlPowerReader&) = delete;
    NvmlPowerReader& operator=(const NvmlPowerReader&) = delete;
    ~NvmlPowerReader() {
        if (shutdown_) shutdown_();
        if (library_) dlclose(library_);
    }

    static std::string physical_pci_id(int cuda_device) {
        char pci[32]{};
        if (cudaDeviceGetPCIBusId(pci, sizeof(pci), cuda_device) != cudaSuccess) throw PowerLimitError();
        return pci;
    }

    void open_pci(const char* pci) {
        if (const char* mode = std::getenv("CYBERMINER_5080_PROFILE")) {
            const std::string_view value(mode);
            if (value == "performance") limits_ = performance_5080;
            else if (value != "conservative") throw PowerLimitError();
        }
        auto init = function<decltype(&nvmlInit_v2)>("nvmlInit_v2");
        auto shutdown = function<decltype(&nvmlShutdown)>("nvmlShutdown");
        auto find = function<decltype(&nvmlDeviceGetHandleByPciBusId_v2)>("nvmlDeviceGetHandleByPciBusId_v2");
        requested_ = function<decltype(requested_)>("nvmlDeviceGetPowerManagementLimit");
        enforced_ = function<decltype(enforced_)>("nvmlDeviceGetEnforcedPowerLimit");
        fields_ = function<decltype(fields_)>("nvmlDeviceGetFieldValues");
        average_ = function<decltype(average_)>("nvmlDeviceGetPowerUsage");
        clock_ = function<decltype(clock_)>("nvmlDeviceGetClockInfo");
        offset_ = function<decltype(offset_)>("nvmlDeviceGetGpcClkVfOffset");
        if (init() != NVML_SUCCESS) throw PowerLimitError();
        shutdown_ = shutdown;
        if (find(pci, &board_) != NVML_SUCCESS) throw PowerLimitError();
    }

    void begin_job() noexcept { above_320_since_ = {}; }

    bool within_limits(const Profile& profile) noexcept {
        unsigned requested = 0, enforced = 0, average = 0, sm = 0;
        int offset = 0;
        nvmlFieldValue_t field{};
        field.fieldId = NVML_FI_DEV_POWER_INSTANT;
        if (requested_(board_, &requested) != NVML_SUCCESS || enforced_(board_, &enforced) != NVML_SUCCESS ||
            average_(board_, &average) != NVML_SUCCESS || clock_(board_, NVML_CLOCK_SM, &sm) != NVML_SUCCESS ||
            offset_(board_, &offset) != NVML_SUCCESS || fields_(board_, 1, &field) != NVML_SUCCESS ||
            field.nvmlReturn != NVML_SUCCESS)
            return false;
        unsigned long long instant = 0;
        switch (field.valueType) {
        case NVML_VALUE_TYPE_UNSIGNED_INT: instant = field.value.uiVal; break;
        case NVML_VALUE_TYPE_UNSIGNED_LONG: instant = field.value.ulVal; break;
        case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: instant = field.value.ullVal; break;
        default: return false;
        }
        if (!power_allowed(profile, requested, enforced) || !sample_allowed(average, instant, sm, offset, limits_))
            return false;
        if (average <= 320000 && instant <= 320000) {
            above_320_since_ = {};
            return true;
        }
        const auto now = std::chrono::steady_clock::now();
        if (above_320_since_ == std::chrono::steady_clock::time_point{}) above_320_since_ = now;
        return now - above_320_since_ < std::chrono::milliseconds(500);
    }
};

inline void verify_power(const Profile& profile, int device) {
    if (!profile.max_board_mw) return;
    NvmlPowerReader reader;
    reader.open_pci(NvmlPowerReader::physical_pci_id(device).c_str());
    if (!reader.within_limits(profile)) throw PowerLimitError();
}

// Retain the NVML binding within one worker, and resolve the physical PCI ID
// on every job. A device change or failed bind cannot reuse the old handle.
class WorkerPowerReader {
    std::unique_ptr<NvmlPowerReader> reader_;
    std::string pci_;
public:
    void clear() noexcept { reader_.reset(); pci_.clear(); }
    NvmlPowerReader& for_job(int device) {
        try {
            const auto pci = NvmlPowerReader::physical_pci_id(device);
            if (!reader_ || pci_ != pci) {
                clear();
                auto next = std::make_unique<NvmlPowerReader>();
                next->open_pci(pci.c_str());
                pci_ = pci;
                reader_ = std::move(next);
            }
        } catch (...) {
            clear();
            throw;
        }
        reader_->begin_job();
        return *reader_;
    }
};

// A worker can run several jobs. Check each one, then keep checking its physical
// board while kernels execute. Exiting this process tears down its CUDA context.
class PowerWatch {
    WorkerPowerReader local_;
    NvmlPowerReader* reader_ = nullptr;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread thread_;
    bool finished_ = false;
public:
    void start(const Profile& profile, int device, WorkerPowerReader* session = nullptr) {
        if (!profile.max_board_mw) return;
        auto& owner = session ? *session : local_;
        reader_ = &owner.for_job(device);
        if (!reader_->within_limits(profile)) {
            reader_ = nullptr;
            owner.clear();
            throw PowerLimitError();
        }
        thread_ = std::thread([this, &profile] {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!wake_.wait_for(lock, std::chrono::milliseconds(50), [this] { return finished_; })) {
                if (!reader_->within_limits(profile)) {
                    std::fputs("RTX 5080 board guard stopped this job\n", stderr);
                    std::fflush(stderr);
                    std::_Exit(78);
                }
            }
        });
    }
    ~PowerWatch() {
        if (!thread_.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
        }
        wake_.notify_one();
        thread_.join();
    }
};
}  // namespace pearl_device
