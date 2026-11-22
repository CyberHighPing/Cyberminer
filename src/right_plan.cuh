#pragma once
#include "wide_tail.cuh"
#include "narrow_tail.cuh"
#include "direct_main.cuh"
#include "direct_edge.cuh"
#include "left_pack.cuh"
#include "leaf_packed.cuh"
#include "native_fold.cuh"
#include <algorithm>
#include <cstdlib>
#include <string>

namespace right_resident {
using Byte=std::uint8_t;
using Word=std::uint32_t;

inline bool enabled() {
    const char* raw=std::getenv("PEARL_RESIDENT_AXIS");
    if(!raw||std::string(raw)=="left")return false;
    if(std::string(raw)=="right")return true;
    throw std::invalid_argument("unknown resident operand");
}
inline int row_strip() {
    const char* raw=std::getenv("PEARL_ROW_STRIP");
    if(!raw)return 1024;
    for(int rows=256;rows<=8192;rows+=256)
        if(std::to_string(rows)==raw)return rows;
    throw std::invalid_argument("invalid row strip height");
}
inline bool packed_selected(){
    const char* mode=std::getenv("PEARL_RESIDENT_BYTES");
    if(!mode||!std::strcmp(mode,"row"))return false;
    if(!std::strcmp(mode,"packed"))return true;
    throw std::invalid_argument("invalid right-resident operand layout");
}
inline int body_columns(int cols){
    if(cols<256||cols%128)throw std::invalid_argument("invalid number of resident columns");
    const int remainder=cols%192,edge=remainder==64?256:remainder;
    return cols-edge;
}
class Plan {
    wide_tail::Plan wide_;
    narrow_tail::Plan edge_;
    native_fold::Mode fold_=native_fold::Mode::canonical;
    bool packed_;direct_main::Plan packed_main_;direct_edge::Plan packed_edge_;
    cudaStream_t edge_stream_=nullptr;
    cudaEvent_t begin_=nullptr,end_=nullptr;
    std::uint64_t cells_=0;
    void close() noexcept {
        if(edge_stream_)cudaStreamSynchronize(edge_stream_);
        if(begin_)cudaEventDestroy(begin_);
        if(end_)cudaEventDestroy(end_);
        if(edge_stream_)cudaStreamDestroy(edge_stream_);
        edge_stream_=nullptr;begin_=nullptr;end_=nullptr;
    }
    void prepare_edge_stream() {
        if(edge_stream_)return;
        try{
            wide_tail::check_cuda(cudaStreamCreateWithFlags(&edge_stream_,cudaStreamNonBlocking),"create right edge stream");
            wide_tail::check_cuda(cudaEventCreateWithFlags(&begin_,cudaEventDisableTiming),"create right begin event");
            wide_tail::check_cuda(cudaEventCreateWithFlags(&end_,cudaEventDisableTiming),"create right end event");
        }catch(...){close();throw;}
    }
public:
    explicit Plan(bool packed=packed_selected(),bool pair_columns=false):packed_(packed),packed_main_(pair_columns){}
    ~Plan(){close();}
    Plan(const Plan&)=delete;
    std::uint64_t cells()const{return cells_;}
    native_fold::Mode fold()const{return fold_;}
    void certify_native(const pearl_prep::Row* metadata,size_t count,int k,Word* temporary,native_fold::Mode desired){
        if(cells_)throw std::logic_error("fold policy was changed after a matrix launch");
        if(packed_)fold_=native_fold::certify(desired,metadata,count,k,temporary);
    }
    size_t launches()const{return wide_.calls()+edge_.calls()+packed_main_.calls()+packed_edge_.calls();}
    void run(const Byte* a,const Byte* b,Word* out,int k,int rows,int cols) {
        if(rows<256 || rows%256 || cols<256 || cols%128)
            throw std::invalid_argument("incomplete matrix tile in right resident plan");
        const int wide=body_columns(cols),edge=cols-wide,pitch=cols/32;
        if(wide%192 || edge%128)throw std::logic_error("invalid column split");
        if(edge){
            prepare_edge_stream();
            wide_tail::check_cuda(cudaEventRecord(begin_),"right inputs ready");
            wide_tail::check_cuda(cudaStreamWaitEvent(edge_stream_,begin_,0),"wait for right inputs");
            if(packed_&&fold_!=native_fold::Mode::canonical)packed_edge_.run<128,256,true>(a,b+size_t(wide)*k,out+size_t(wide/32)*16,k,rows,edge,pitch,edge_stream_);
            else if(packed_)packed_edge_.run<128,256>(a,b+size_t(wide)*k,out+size_t(wide/32)*16,k,rows,edge,pitch,edge_stream_);
            else edge_.run(a,b+size_t(wide)*k,out+size_t(wide/32)*16,k,rows,edge,pitch,edge_stream_);
        }
        if(wide){
            if(packed_&&fold_==native_fold::Mode::paired)packed_main_.run<128,256,2>(a,b,out,k,rows,wide,pitch,nullptr);
            else if(packed_&&fold_==native_fold::Mode::finite)packed_main_.run<128,256,1>(a,b,out,k,rows,wide,pitch,nullptr);
            else if(packed_)packed_main_.run<128,256>(a,b,out,k,rows,wide,pitch,nullptr);
            else wide_.run(a,b,out,k,rows,wide,pitch,nullptr);
        }
        if(edge){
            wide_tail::check_cuda(cudaEventRecord(end_,edge_stream_),"right edge done");
            wide_tail::check_cuda(cudaStreamWaitEvent(nullptr,end_,0),"join right edge");
        }
        cells_+=std::uint64_t(rows)*cols;
    }
};
class CacheWindow {
    cudaStreamAttrValue old_attr_{};
    size_t old_limit_=0,bytes_=0;
    bool limit_changed_=false,attr_changed_=false;
    void restore() noexcept {
        if(attr_changed_){cudaStreamSetAttribute(nullptr,cudaStreamAttributeAccessPolicyWindow,&old_attr_);attr_changed_=false;}
        if(limit_changed_){cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,old_limit_);limit_changed_=false;}
    }
public:
    CacheWindow(Byte* ptr,size_t size,const cudaDeviceProp& dev) {
        const char* raw=std::getenv("PEARL_STRIP_CACHE");
        if(!raw||std::string(raw)=="default")return;
        if(std::string(raw)!="persist")throw std::invalid_argument("unknown row cache policy");
        if(!dev.persistingL2CacheMaxSize||!dev.accessPolicyMaxWindowSize)
            throw std::runtime_error("device does not expose a persisting L2 window");
        bytes_=std::min<size_t>(size,dev.persistingL2CacheMaxSize);
        const size_t window=std::min<size_t>(size,dev.accessPolicyMaxWindowSize);
        try{
            wide_tail::check_cuda(cudaDeviceGetLimit(&old_limit_,cudaLimitPersistingL2CacheSize),"read L2 reservation");
            wide_tail::check_cuda(cudaStreamGetAttribute(nullptr,cudaStreamAttributeAccessPolicyWindow,&old_attr_),"read L2 window");
            wide_tail::check_cuda(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,bytes_),"reserve L2 for the strip");
            limit_changed_=true;
            cudaStreamAttrValue attr{};
            attr.accessPolicyWindow.base_ptr=ptr;
            attr.accessPolicyWindow.num_bytes=window;
            attr.accessPolicyWindow.hitRatio=float(bytes_)/float(window);
            attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
            wide_tail::check_cuda(cudaStreamSetAttribute(nullptr,cudaStreamAttributeAccessPolicyWindow,&attr),"apply row strip L2 window");
            attr_changed_=true;
        }catch(...){restore();throw;}
    }
    CacheWindow(const CacheWindow&)=delete;
    ~CacheWindow(){restore();}
    size_t bytes()const{return bytes_;}
};
}
