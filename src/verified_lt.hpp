#pragma once
#include "scan_depth.hpp"

#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <link.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>

namespace verified_lt {

inline constexpr char name[]="cublaslt-verified-pipeline";
inline constexpr char config[]="67,20,15,0,18,0,0";
inline constexpr char build_id[]="a960e927f8da0bfc7747bd5a616918d06bcc764c";
inline constexpr std::array<unsigned char,20> id_bytes{
    0xa9,0x60,0xe9,0x27,0xf8,0xda,0x0b,0xfc,0x77,0x47,0xbd,0x5a,0x61,0x69,0x18,0xd0,0x6b,0xcc,0x76,0x4c};
inline constexpr int version=130101;
inline constexpr std::uint64_t numerical_flags=12583426;
inline constexpr std::size_t offered_workspace=32u<<20;

inline bool eligible(int m,int n,int k) { return m==4096 && n==16384 && pearl_geometry::supported_scan_depth(k); }

struct Image {
    void* base=nullptr;
    bool found=false;
};

inline int inspect_notes(dl_phdr_info* info,std::size_t,void* arg) {
    auto& image=*static_cast<Image*>(arg);
    if(reinterpret_cast<void*>(info->dlpi_addr)!=image.base) return 0;
    for(unsigned i=0;i<info->dlpi_phnum;++i) {
        const auto& s=info->dlpi_phdr[i];
        if(s.p_type!=PT_NOTE) continue;
        auto* p=reinterpret_cast<const unsigned char*>(info->dlpi_addr+s.p_vaddr);
        std::size_t length=s.p_memsz;
        while(length>=sizeof(ElfW(Nhdr))) {
            ElfW(Nhdr) note{};
            std::memcpy(&note,p,sizeof(note));p+=sizeof(note);length-=sizeof(note);
            const auto names=(std::size_t(note.n_namesz)+3)&~std::size_t(3);
            const auto desc=(std::size_t(note.n_descsz)+3)&~std::size_t(3);
            if(names>length || desc>length-names) break;
            if(note.n_type==NT_GNU_BUILD_ID && note.n_namesz==4 && note.n_descsz==id_bytes.size()
               && std::memcmp(p,"GNU",4)==0 && std::memcmp(p+names,id_bytes.data(),id_bytes.size())==0) {
                image.found=true;return 1;
            }
            p+=names+desc;length-=names+desc;
        }
    }
    return 1;
}

// Only the new variant loads the optional system/administrator-provided library.
// The ELF has no cuBLASLt dependency. A different build falls back to CUTLASS.
#define PEARL_LT_SYMBOLS(X) \
    X(cublasLtGetVersion) X(cublasLtCreate) X(cublasLtDestroy) \
    X(cublasLtMatmulDescCreate) X(cublasLtMatmulDescDestroy) X(cublasLtMatmulDescSetAttribute) \
    X(cublasLtMatrixLayoutCreate) X(cublasLtMatrixLayoutDestroy) X(cublasLtMatmulAlgoInit) \
    X(cublasLtMatmulAlgoConfigGetAttribute) X(cublasLtMatmulAlgoConfigSetAttribute) \
    X(cublasLtMatmulAlgoCapGetAttribute) X(cublasLtMatmulAlgoCheck) X(cublasLtMatmul)

struct Library {
    void* so=nullptr;
    const char* reason="library_missing";
    std::size_t actual_version=0;
    bool id_matches=false;
#define PEARL_LT_DECLARE(f) decltype(&::f) f=nullptr;
    PEARL_LT_SYMBOLS(PEARL_LT_DECLARE)
#undef PEARL_LT_DECLARE
    Library() {
        so=dlopen("libcublasLt.so.13",RTLD_NOW|RTLD_LOCAL);
        if(!so) return;
        reason="library_symbols";
#define PEARL_LT_LOAD(f) f=reinterpret_cast<decltype(f)>(dlsym(so,#f)); if(!f) return;
        PEARL_LT_SYMBOLS(PEARL_LT_LOAD)
#undef PEARL_LT_LOAD
        Dl_info d{};
        reason="library_build";
        if(!dladdr(reinterpret_cast<void*>(cublasLtGetVersion),&d)) return;
        Image image{d.dli_fbase,false};
        dl_iterate_phdr(inspect_notes,&image);
        if(!image.found) return;
        id_matches=true;
        actual_version=cublasLtGetVersion();
        reason=actual_version==version ? nullptr:"library_version";
    }
    ~Library(){if(so) dlclose(so);}
    Library(const Library&)=delete;
};
#undef PEARL_LT_SYMBOLS

inline const Library& library() { static const Library api;return api; }

// One per request. No device workspace or vendor handle survives completion.
class Plan {
    const Library* api_=nullptr;
    cublasLtHandle_t handle_=nullptr;
    cublasLtMatmulDesc_t op_=nullptr;
    cublasLtMatrixLayout_t left_=nullptr,right_=nullptr,out_=nullptr;
    cublasLtMatmulAlgo_t algo_{};
    int k_=0;
    const char* reason_="shape";
    int last_status_=0;
    std::size_t requested_=0;
    void* owned_workspace_=nullptr;
    void* workspace_=nullptr;
    std::size_t extra_peak_=0,offered_peak_=0;
    std::uint64_t flags_=0;
    int split_=-1;
    std::uint32_t reduction_=~std::uint32_t(0);
    std::size_t attempted_=0,calls_=0,cutlass_=0,native_=0,shape_fallback_=0;

    bool success(cublasStatus_t status) {
        if(status==CUBLAS_STATUS_SUCCESS) return true;
        last_status_=int(status);return false;
    }
    template<class T> bool get(cublasLtMatmulAlgoConfigAttributes_t attr,T& value) {
        std::size_t n=0;
        return success(api_->cublasLtMatmulAlgoConfigGetAttribute(&algo_,attr,&value,sizeof(T),&n)) && n==sizeof(T);
    }
    template<class T> bool set(cublasLtMatmulAlgoConfigAttributes_t attr,T value) {
        T current{};
        if(get(attr,current) && value==current) return true;
        return success(api_->cublasLtMatmulAlgoConfigSetAttribute(&algo_,attr,&value,sizeof(value)))
            && get(attr,current) && current==value;
    }
public:
    Plan()=default;
    Plan(const Plan&)=delete;
    ~Plan(){clear();}
    void clear() {
        if(!api_) return;
        if(out_) api_->cublasLtMatrixLayoutDestroy(out_);
        if(right_) api_->cublasLtMatrixLayoutDestroy(right_);
        if(left_) api_->cublasLtMatrixLayoutDestroy(left_);
        if(op_) api_->cublasLtMatmulDescDestroy(op_);
        if(handle_) api_->cublasLtDestroy(handle_);
        if(owned_workspace_) cudaFree(owned_workspace_);
        owned_workspace_=nullptr;workspace_=nullptr;
        out_=nullptr;right_=nullptr;left_=nullptr;op_=nullptr;handle_=nullptr;
    }
    void prepare(int m,int n,int k) {
        if(!eligible(m,n,k)) return;
        api_=&library();
        reason_=api_->reason;
        if(reason_) return;
        k_=k;
        reason_="handle_or_descriptors";
        if(!success(api_->cublasLtCreate(&handle_)) ||
           !success(api_->cublasLtMatmulDescCreate(&op_,CUBLAS_COMPUTE_32F,CUDA_R_32F))) return;
        const cublasOperation_t trans=CUBLAS_OP_T,normal=CUBLAS_OP_N;
        const std::int8_t fast=0;
        if(!success(api_->cublasLtMatmulDescSetAttribute(op_,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans))) ||
           !success(api_->cublasLtMatmulDescSetAttribute(op_,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal))) ||
           !success(api_->cublasLtMatmulDescSetAttribute(op_,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast,sizeof(fast))) ||
           !success(api_->cublasLtMatrixLayoutCreate(&left_,CUDA_R_8F_E4M3,k,n,k)) ||
           !success(api_->cublasLtMatrixLayoutCreate(&right_,CUDA_R_8F_E4M3,k,m,k)) ||
           !success(api_->cublasLtMatrixLayoutCreate(&out_,CUDA_R_32F,n,m,n))) return;
        reason_="configuration";
        if(!success(api_->cublasLtMatmulAlgoInit(handle_,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_8F_E4M3,
                CUDA_R_8F_E4M3,CUDA_R_32F,CUDA_R_32F,67,&algo_))) return;
        int id=-1;
        if(!get(CUBLASLT_ALGO_CONFIG_ID,id) || id!=67 ||
           !set(CUBLASLT_ALGO_CONFIG_TILE_ID,std::uint32_t(20)) || !set(CUBLASLT_ALGO_CONFIG_STAGES_ID,std::uint32_t(15)) ||
           !set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,std::uint32_t(0)) || !set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,std::uint32_t(18)) ||
           !set(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,std::uint16_t(0)) || !set(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,std::uint16_t(0)) ||
           !set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM,std::int32_t(1)) || !set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,std::uint32_t(0)) ||
           !get(CUBLASLT_ALGO_CONFIG_SPLITK_NUM,split_) || !get(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,reduction_) ||
           split_!=1 || reduction_!=0) return;
        std::size_t nflags=0;
        if(!success(api_->cublasLtMatmulAlgoCapGetAttribute(&algo_,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
            &flags_,sizeof(flags_),&nflags)) || nflags!=sizeof(flags_) || flags_!=numerical_flags) return;
        cublasLtMatmulHeuristicResult_t check{};
        reason_="unsupported";
        if(!success(api_->cublasLtMatmulAlgoCheck(handle_,op_,left_,right_,out_,out_,&algo_,&check)) ||
           !success(check.state)) return;
        requested_=check.workspaceSize;
        // Both qualified shapes need exactly zero workspace. A different
        // requirement must not silently increase the original VRAM budget.
        if(requested_!=0){reason_="workspace";return;}
        last_status_=0;reason_=nullptr;
    }
    bool ready() const { return !reason_; }
    void pressure(){reason_="memory_pressure";clear();}
    std::size_t extra_bytes() const { return owned_workspace_?offered_workspace:0; }
    void reserve_dedicated() {
        const auto status=cudaMalloc(&owned_workspace_,offered_workspace);
        if(status==cudaErrorMemoryAllocation) {
            cudaGetLastError();pressure();return;
        }
        if(status!=cudaSuccess) throw std::runtime_error(std::string("cuBLASLt workspace allocation: ")+cudaGetErrorString(status));
        extra_peak_=offered_workspace;
    }
    void bind_workspace(void* staging,std::size_t staging_bytes) {
        if(!ready()) return;
        workspace_=owned_workspace_?owned_workspace_:staging;
        if(!workspace_ || (!owned_workspace_ && staging_bytes<offered_workspace))
            throw std::logic_error("cuBLASLt workspace contract");
        offered_peak_=offered_workspace;
    }
    void record_cutlass(){++cutlass_;}
    void record_native(){++native_;}

    bool run(const void* a,const void* b,void* output,int m,int n,int k) {
        if(!eligible(m,n,k)){++shape_fallback_;return false;}
        if(!ready() || k!=k_) return false;
        const float alpha=1,beta=0;
        ++attempted_;
        auto st=api_->cublasLtMatmul(handle_,op_,&alpha,b,left_,a,right_,&beta,output,out_,output,out_,
                                      &algo_,workspace_,offered_workspace,nullptr);
        if(st==CUBLAS_STATUS_SUCCESS){++calls_;return true;}
        last_status_=int(st);
        if(st!=CUBLAS_STATUS_NOT_SUPPORTED && st!=CUBLAS_STATUS_ARCH_MISMATCH &&
           st!=CUBLAS_STATUS_ALLOC_FAILED && st!=CUBLAS_STATUS_INVALID_VALUE)
            throw std::runtime_error("cuBLASLt execution failed; scan invalid, status "+std::to_string(last_status_));
        // Quiesce before overwriting the entire product with the reference.
        // A device execution error invalidates the job instead of producing a ticket.
        cudaError_t done=cudaStreamSynchronize(nullptr);
        if(done!=cudaSuccess) throw std::runtime_error(std::string("cuBLASLt fallback fence: ")+cudaGetErrorString(done));
        auto pending=cudaGetLastError();
        if(pending!=cudaSuccess && pending!=cudaErrorMemoryAllocation && pending!=cudaErrorInvalidValue &&
           pending!=cudaErrorNotSupported)
            throw std::runtime_error(std::string("cuBLASLt fallback status: ")+cudaGetErrorString(pending));
        reason_="launch_fallback";
        clear();return false;
    }
    void write(std::ostream& os) const {
        os<<",\"cublaslt_calls\":"<<calls_<<",\"cublaslt_attempts\":"<<attempted_
          <<",\"cublaslt_fallback_calls\":"<<(cutlass_+native_)<<",\"cutlass_fallback_calls\":"<<cutlass_
          <<",\"native_fallback_calls\":"<<native_<<",\"cublaslt_shape_fallback_calls\":"<<shape_fallback_
          <<",\"cublaslt_config\":\""<<config<<"\",\"cublaslt_version\":"<<(api_?api_->actual_version:0)
          <<",\"cublaslt_build_id\":";
        if(api_ && api_->id_matches) os<<'\"'<<build_id<<'\"'; else os<<"null";
        os<<",\"cublaslt_workspace_requested_bytes\":"<<requested_<<",\"cublaslt_workspace_bytes\":"<<offered_peak_
          <<",\"cublaslt_workspace_extra_bytes\":"<<extra_peak_
          <<",\"cublaslt_split_k\":"<<split_<<",\"cublaslt_reduction\":";
        if(reduction_==~std::uint32_t(0))os<<"null";else os<<reduction_;
        os<<",\"cublaslt_numerical_flags\":"<<flags_<<",\"cublaslt_status\":"<<last_status_
          <<",\"cublaslt_plan\":\""<<(reason_?reason_:"ready")<<"\"";
    }
};

}  // namespace verified_lt
