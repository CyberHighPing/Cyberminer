#pragma once
#include "wide_tail.cuh"
#include "narrow_tail.cuh"
#include <cstdlib>
#include <string>
namespace pearl_fused_stream {
using Byte=std::uint8_t;using Word=std::uint32_t;
inline constexpr char name[]="sm120-fused-stream";
class Plan {
    wide_tail::Plan main_;
    narrow_tail::Plan edge_;
    int edge_cols_=256,windows_=1;
    bool full_width_=false;
    size_t launches_=0;
    bool concurrent_=false;
    std::size_t calls_=0;
    cudaStream_t side_=nullptr;
    cudaEvent_t start_=nullptr,finish_=nullptr;
    void setup_side(){
        if(side_)return;
        wide_tail::check_cuda(cudaStreamCreateWithFlags(&side_,cudaStreamNonBlocking),"create edge stream");
        wide_tail::check_cuda(cudaEventCreateWithFlags(&start_,cudaEventDisableTiming),"create input event");
        wide_tail::check_cuda(cudaEventCreateWithFlags(&finish_,cudaEventDisableTiming),"create edge event");
    }
public:
    Plan(){
        const char* group=std::getenv("PEARL_PANEL_GROUP");
        const std::string mode=group?group:"1";
        for(int size:{1,2,3,4,6,7,8,14,21,42,256})if(std::to_string(size)==mode){windows_=size;break;}
        if(std::to_string(windows_)!=mode)throw std::invalid_argument("invalid row window group");
        const char* raw=std::getenv("PEARL_TAIL_LAYOUT");const std::string m=raw?raw:"serial256";
        if(m=="wide"){full_width_=true;return;}
        for(int n:{256,640,1024}){
            if(m=="serial"+std::to_string(n) || m=="overlap"+std::to_string(n)){
                edge_cols_=n;concurrent_=m.rfind("overlap",0)==0;return;
            }
        }
        throw std::invalid_argument("unknown column edge configuration");
    }
    Plan(const Plan&)=delete;
    Plan& operator=(const Plan&)=delete;
    ~Plan(){if(side_)cudaStreamSynchronize(side_);if(start_)cudaEventDestroy(start_);if(finish_)cudaEventDestroy(finish_);if(side_)cudaStreamDestroy(side_);}
    std::size_t calls()const{return calls_;}
    int group()const{return windows_;}
    size_t launches()const{return launches_;}
    void run(const Byte* a,const Byte* b,Word* out,int k){run_rows(a,b,out,k,4096);}
    void run_rows(const Byte* a,const Byte* b,Word* out,int k,int m){
        if(m<4096||m%4096)throw std::invalid_argument("unaligned row grouping");
        if(full_width_){
            main_.run(a,b,out,k,m,16384,512,nullptr);calls_+=m/4096;++launches_;return;
        }
        const int left=16384-edge_cols_,ld=16384/32;
        if(left%192 || edge_cols_%128)throw std::logic_error("invalid window coverage");
        const Byte* edge_start=b+size_t(left)*k;
        Word* edge_out=out+size_t(left/32)*16;
        if(concurrent_){
            setup_side();
            try{
                wide_tail::check_cuda(cudaEventRecord(start_),"input ready");
                wide_tail::check_cuda(cudaStreamWaitEvent(side_,start_,0),"edge input ready");
                main_.run(a,b,out,k,m,left,ld,nullptr);
                edge_.run(a,edge_start,edge_out,k,m,edge_cols_,ld,side_);
                wide_tail::check_cuda(cudaEventRecord(finish_,side_),"edge done");
                wide_tail::check_cuda(cudaStreamWaitEvent(nullptr,finish_,0),"complete full output");
            }catch(...){cudaStreamSynchronize(side_);throw;}
        }else{
            main_.run(a,b,out,k,m,left,ld,nullptr);
            edge_.run(a,edge_start,edge_out,k,m,edge_cols_,ld,nullptr);
        }
        calls_+=m/4096;launches_+=2;
    }
};
}
