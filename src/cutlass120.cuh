#pragma once
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace pearl_cutlass {
inline bool recognized(const std::string& name) {
    return name == "cutlass120-coop" || name == "cutlass120-ping" ||
           name == "cutlass120-coop-fold" || name == "cutlass120-coop-pipeline" ||
           name == "cutlass120-coop-swizzle" || name == "cutlass120-coop-swizzle-pipeline";
}

class CannotImplement : public std::runtime_error {
public:
    explicit CannotImplement(cutlass::Status status) :
        std::runtime_error(std::string("CUTLASS can_implement: ")+cutlassGetStatusString(status)) {}
};
inline void checked(cutlass::Status status, const char* where) {
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string(where) + ": " + cutlassGetStatusString(status));
}

template<bool Cooperative>
class Plan {
    using A = cutlass::float_e4m3_t;
    using B = cutlass::float_e4m3_t;
    using Row = cutlass::layout::RowMajor;
    using Col = cutlass::layout::ColumnMajor;
    using Shape = cute::Shape<int,int,int,int>;
    using Tile = std::conditional_t<Cooperative, cute::Shape<cute::_128,cute::_128,cute::_128>,
                                                cute::Shape<cute::_64,cute::_128,cute::_128>>;
    using Cluster = cute::Shape<cute::_1,cute::_1,cute::_1>;
    using Schedule = std::conditional_t<Cooperative, cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                                     cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
    using EpiSchedule = std::conditional_t<Cooperative, cutlass::epilogue::TmaWarpSpecializedCooperative,
                                                        cutlass::epilogue::TmaWarpSpecialized>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float,
        void, Row, 4, float, Row, 4, EpiSchedule>::CollectiveOp;
    using Mma = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, A, Row, 16, B, Col, 16, float,
        Tile, Cluster, cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
        Schedule>::CollectiveOp;
    // Only the M/N work tiles are scheduled across the grid. Each warp retains
    // its FP32 accumulator while iterating through every ordered K=32 atom.
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape, Mma, Epi, cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    Gemm gemm_{};
    void* workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
    int old_m_ = 0, old_n_ = 0, old_k_ = 0;
    const void* old_a_ = nullptr;
    const void* old_b_ = nullptr;
    void* old_c_ = nullptr;
public:
    Plan() = default;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    ~Plan() { if (workspace_) cudaFree(workspace_); }

    void operator()(const uint8_t* a, const uint8_t* b, uint32_t* out, int m, int n, int k,
                    int device_id, int sm_count) {
        if ((n & 3) || (k & 31)) throw std::invalid_argument("CUTLASS path needs N divisible by 4 and K divisible by 32");
        if (old_m_ != m || old_n_ != n || old_k_ != k || a != old_a_ || b != old_b_ || out != old_c_) {
            auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, cute::make_shape(m,k,1));
            auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, cute::make_shape(n,k,1));
            auto sc = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, cute::make_shape(m,n,1));
            auto sd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, cute::make_shape(m,n,1));
            typename Gemm::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm, {m,n,k,1},
                {reinterpret_cast<const A*>(a), sa, reinterpret_cast<const B*>(b), sb},
                {{}, nullptr, sc, reinterpret_cast<float*>(out), sd}
            };
            args.hw_info.device_id = device_id;
            args.hw_info.sm_count = sm_count;
            // The epilogue has no source tensor. Set the output scale explicitly.
            args.epilogue.thread.alpha = 1.0f;
            const auto can=gemm_.can_implement(args);
            if (can!=cutlass::Status::kSuccess) throw CannotImplement(can);
            const auto need = Gemm::get_workspace_size(args);
            if (need > workspace_bytes_) {
                void* replacement = nullptr;
                auto st = cudaMalloc(&replacement, need);
                if (st != cudaSuccess) throw std::runtime_error(cudaGetErrorString(st));
                if (workspace_) cudaFree(workspace_);
                workspace_ = replacement;
                workspace_bytes_ = need;
            }
            checked(gemm_.initialize(args, workspace_), "CUTLASS initialize");
            old_m_=m; old_n_=n; old_k_=k; old_a_=a; old_b_=b; old_c_=out;
        }
        checked(gemm_.run(), "CUTLASS run");
    }
};

class Candidate {
    Plan<true> coop_;
    Plan<false> ping_;
    int device_id_;
    int sm_count_;
public:
    Candidate(int device_id, int sm_count) : device_id_(device_id), sm_count_(sm_count) {}
    void run(const std::string& name, const uint8_t* a, const uint8_t* b, uint32_t* c, int m, int n, int k) {
        if (recognized(name) && name != "cutlass120-ping")
            return coop_(a,b,c,m,n,k,device_id_,sm_count_);
        if (name == "cutlass120-ping") return ping_(a,b,c,m,n,k,device_id_,sm_count_);
        throw std::invalid_argument("unrecognized CUTLASS variant");
    }
};
}
