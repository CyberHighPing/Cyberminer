#pragma once
#include "fold16x16.cuh"
#include "rect_vector_fold.cuh"
#include "ordered_hits.hpp"
#include "paired_output.hpp"
#include <exception>

// Both complete raw planes are committed before any multiplication. Afterwards
// A stays resident and B is regenerated from the same XOF in global row order.
namespace column_stream {

struct DrainOnError {
    cudaStream_t fold;
    int exceptions=std::uncaught_exceptions();
    ~DrainOnError() {
        if(std::uncaught_exceptions()>exceptions) {
            cudaStreamSynchronize(nullptr);
            cudaStreamSynchronize(fold);
        }
    }
};

inline int run(const Options& o,WorkerBufferCache* pool) {
    constexpr int window_m=4096, window_n=16384;
    const bool direct=o.variant==pearl_fused_stream::name;
    const bool vendor=o.variant==verified_lt::name;
    const bool square=o.h==16 && o.w==16 && o.k==16384;
    const bool vector_rect=o.h==16 && o.w==32 && o.k==32768 && pearl_rect_vector::enabled();
    if(o.native_job.empty() || !o.resident || !o.deferred || (!direct && !vendor && o.variant!="cutlass120-coop-swizzle-pipeline") ||
       o.rows!=window_m || o.cols!=window_n || (o.k!=16384 && o.k!=32768) ||
       o.m%window_m || o.n%window_n || (!square && (o.h!=16 || o.w!=32)) || o.row_pad || o.k_major || o.fused ||
       !o.dump.empty() || !o.lanes.empty())
        throw std::invalid_argument("stream-B requires aligned native 4096x16384 windows, 16x32 K=16384/32768, or 16x16 K=16384");

    const auto start=Clock::now();
    using Side=pearl_jobgen::Side;
    native_inputs::Plan native(o.native_job,o.m,o.n,o.k,o.h,o.w);
    const int batch=4096;
    const size_t bytes_a=size_t(o.m)*o.k, bytes_b=size_t(o.n)*o.k;
    const size_t panel_b=size_t(window_n)*o.k;
    const size_t cells=size_t(window_m)*window_n;
    const size_t local_words=size_t(o.m/o.h)*(window_n/o.w)*16;
    const size_t all_words=size_t(o.m/o.h)*(o.n/o.w)*16;
    const size_t product_bytes=direct?0:2*cells*sizeof(word);
    const size_t live_bytes=bytes_a+panel_b+product_bytes+local_words*sizeof(word);
    const size_t capacity=o.hit_capacity;
    const size_t extra=capacity*sizeof(pearl_ticket::Hit)+sizeof(pearl_ticket::Count)+sizeof(unsigned);
    const size_t work_reserve=direct?0:verified_lt::offered_workspace;
    const size_t whole_scratch=std::max(native.scratch_bytes(batch),work_reserve);
    const bool panel_commit=pearl_panel_b::enabled(pearl_panel_b::selected(),std::max(bytes_b,live_bytes)+whole_scratch+extra);
    const size_t arena_bytes=std::max(panel_commit?native.panel_b_storage():bytes_b,live_bytes);
    const auto prep_mode=prep_inputs::selected();
    const bool save=prep_inputs::saved_rows(prep_mode);
    const size_t scratch_body=panel_commit?std::max(native.panel_b_scratch(batch),work_reserve):whole_scratch;
    const size_t meta_offset=(scratch_body+15)&~size_t(15);
    const size_t meta_bytes=save?(size_t(o.m)+o.n)*sizeof(prep_inputs::Row):0;
    const size_t scratch_bytes=save?meta_offset+meta_bytes:scratch_body;
    const size_t required=arena_bytes+scratch_bytes+extra;
    if(required>pearl_pool::ceiling)throw std::runtime_error("stream-B geometry exceeds the 7168 MiB allocation limit");

    pearl_pool::Meter meter;
    pearl_pool::Shape layout{o.device,{o.m,o.n,o.k,o.h,o.w,o.rows,o.cols,o.k,o.row_pad,o.hit_capacity},
                              {arena_bytes,0,scratch_bytes,0,0,capacity,1,1,0},(1u<<12)|(panel_commit?(1u<<13):0u),o.variant};
    if(pool){meter=pool->previous_idle_release();pool->match(&layout,meter);}
    CUDA(cudaSetDevice(o.device));
    cudaDeviceProp dev;CUDA(cudaGetDeviceProperties(&dev,o.device));
    if(std::string(dev.name).find("RTX 5060 Ti")==std::string::npos || dev.major!=12 || dev.minor!=0 ||
       dev.totalGlobalMem<7000ull*1024*1024 || dev.totalGlobalMem>9000ull*1024*1024)
        throw std::runtime_error("select an RTX 5060 Ti 8GB (SM120)");
    verified_lt::Plan lt;
    pearl_fused_stream::Plan fused;
    if(vendor){lt.prepare(window_m,window_n,o.k);if(!lt.ready())lt.clear();}
    pearl_cutlass::Candidate reference(o.device,dev.multiProcessorCount);
    size_t free_before,total;
    CUDA(cudaMemGetInfo(&free_before,&total));
    const size_t retained=pool?pool->retained():0;
    auto available=[&]{return pool?pool->admission_available(free_before,total):free_before;};
    if(!pearl_pool::admit(required,0,false,available()).fits)
        throw std::runtime_error("insufficient free VRAM for this streamed shape");
    std::unique_ptr<RetainedScanPointers> owner;
    if(pool && pool->reuse){meter.warm=pool->has();owner=meter.warm?pool->take():std::make_unique<RetainedScanPointers>();}
    auto take=[&](size_t i){return owner?owner->take(i):nullptr;};
    auto* measure=pool?&meter:nullptr;
    Buffer<byte> arena(arena_bytes,take(0),measure),scratch(scratch_bytes,take(2),measure);
    Buffer<pearl_ticket::Hit> hits(capacity,take(6),measure);
    Buffer<pearl_ticket::Count> count(1,take(7),measure);
    Buffer<unsigned> invalid(1,take(8),measure);
    FoldStream fold_stream(!direct);
    DrainOnError drain{fold_stream.value};
    size_t free_after;CUDA(cudaMemGetInfo(&free_after,&total));
    // The raw B plane or its panel and all global leaf CVs temporarily own the
    // arena. Commitment completes before original A generation can reuse it.
    auto* meta_a=save?reinterpret_cast<prep_inputs::Row*>(scratch.ptr+meta_offset):nullptr;
    auto* meta_b=save?meta_a+o.m:nullptr;
    CUDA(cudaMemset(invalid.ptr,0,sizeof(unsigned)));
    const auto before_commit=Clock::now();
    const auto hb=panel_commit?native.commit_panel_b(arena.ptr,arena_bytes,meta_b,invalid.ptr,prep_mode==prep_inputs::Mode::leaf||prep_mode==prep_inputs::Mode::leafvec):native.generate_and_commit(Side::B,arena.ptr,bytes_b,scratch.ptr,scratch_body,meta_b,invalid.ptr);
    const auto ha=native.generate_and_commit(Side::A,arena.ptr,bytes_a,scratch.ptr,scratch_body,meta_a,invalid.ptr);
    auto state=native.bind(ha,hb);
    const double commits=std::chrono::duration<double>(Clock::now()-before_commit).count();
    const auto ea=native_inputs::subkey(state.sa,"pearl/v4/FP8/noise-line");
    const auto fb=native_inputs::subkey(state.sb,"pearl/v4/FP8/noise-line");
    byte* const a=arena.ptr;
    byte* const b=a+bytes_a;
    word* const product=reinterpret_cast<word*>(b+panel_b);
    word* const folded=reinterpret_cast<word*>(b+panel_b+product_bytes);
    lt.bind_workspace(scratch.ptr,scratch_body);
    auto validate_rows=[&] {
        unsigned bad=0;CUDA(cudaMemcpy(&bad,invalid.ptr,sizeof(bad),cudaMemcpyDeviceToHost));
        if(bad)throw std::runtime_error("streamed operand row failed the row policy");
    };
    auto begin=Clock::now();
    const auto ek=pearl_noise::key_from_bytes(native_inputs::bytes(ea)),fk=pearl_noise::key_from_bytes(native_inputs::bytes(fb));
    if(prep_mode==prep_inputs::Mode::off){
        CUDA(pearl_noise::prepare_resident(reinterpret_cast<std::int8_t*>(a),o.m,o.k,0,native_inputs::bytes(ea),
             native_inputs::bytes(fb),batch,scratch.ptr,scratch_body,invalid.ptr,nullptr));
    }else{
        CUDA(prep_inputs::prepare(prep_mode,a,o.m,o.k,0,0,ek,fk,native.generator_seed(),batch,scratch.ptr,scratch_body,meta_a,invalid.ptr,true,nullptr));
    }
    validate_rows();
    double quantize_s=std::chrono::duration<double>(Clock::now()-begin).count(),regen_ms=0;
    paired_output::File output(o.out),roots(o.roots_out);
    std::istringstream header(native.ticket_header(state));
    pearl_ticket::SparseWire wire(header,o.m,o.n,o.k,o.h,o.w);
    pearl_ticket::Coverage coverage(o.m/o.h,o.n/o.w);
    wire.begin(output.file);
    const auto receipt=state.receipt();roots.file.write(reinterpret_cast<const char*>(receipt.data()),receipt.size());
    const int nr=o.m/window_m;
    std::vector<PhaseTime> phase(nr);
    std::vector<Event> fold_begin(nr);
    Event gen_begin,gen_end;
    double gemm_ms=0,fold_ms=0,filter_s=0;
    const bool global_order=all_words<=pearl_geometry::legacy_words;
    std::vector<pearl_ticket::Hit> small_grid_hits;
    std::optional<pearl_square::OrderedHits<pearl_ticket::Hit>> square_hits;
    if(square)square_hits.emplace(output.borrowed_staging_dir_fd(),o.n/window_n,o.n/o.w,window_n/o.w,all_words/16);
    for(int col0=0;col0<o.n;col0+=window_n) {
        // Only B's raw data is regenerated. Its row noise is still addressed
        // from the original full matrix's first row, including nonzero col0.
        begin=Clock::now();
        CUDA(cudaEventRecord(gen_begin.value));
        if(!prep_inputs::direct_b(prep_mode))native.regenerate_b(b,panel_b,col0,window_n);
        CUDA(cudaEventRecord(gen_end.value));
        if(prep_mode==prep_inputs::Mode::off){
            CUDA(pearl_noise::prepare_resident(reinterpret_cast<std::int8_t*>(b),window_n,o.k,1,native_inputs::bytes(fb),
                 native_inputs::bytes(fb),batch,scratch.ptr,scratch_body,invalid.ptr,nullptr,col0));
        }else{
            CUDA(prep_inputs::prepare(prep_mode,b,window_n,o.k,1,col0,fk,fk,native.generator_seed(),batch,scratch.ptr,scratch_body,
                 meta_b?meta_b+col0:nullptr,invalid.ptr,col0==0||!direct,nullptr));
        }
        validate_rows();
        float ms=0;CUDA(cudaEventElapsedTime(&ms,gen_begin.value,gen_end.value));regen_ms+=ms;
        quantize_s+=std::max(0.0,std::chrono::duration<double>(Clock::now()-begin).count()-ms/1000.0);
        const int window_group=direct?fused.group():1;
        for(int i=0;i<nr;i+=window_group) {
            if(direct){
                const byte* left=a+size_t(i)*window_m*o.k;
                word* local=folded+size_t(i)*(window_m/o.h)*(window_n/o.w)*16;
                CUDA(cudaEventRecord(phase[i].start.value));
                const int span=std::min(window_group,nr-i);
                fused.run_rows(left,b,local,o.k,span*window_m);
                CUDA(cudaEventRecord(phase[i].end.value));
                continue;
            }
            if(i>=2)CUDA(cudaStreamWaitEvent(nullptr,phase[i-2].end.value,0));
            word* dest=product+(i%2)*cells;
            const byte* left=a+size_t(i)*window_m*o.k;
            CUDA(cudaEventRecord(phase[i].start.value));
            if(!vendor || !lt.run(left,b,dest,window_m,window_n,o.k)){
                reference.run("cutlass120-coop",left,b,dest,window_m,window_n,o.k);
                if(vendor)lt.record_cutlass();
            }
            CUDA(cudaGetLastError());CUDA(cudaEventRecord(phase[i].mid.value));
            CUDA(cudaStreamWaitEvent(fold_stream.value,phase[i].mid.value,0));
            CUDA(cudaEventRecord(fold_begin[i].value,fold_stream.value));
            word* local=folded+size_t(i)*(window_m/o.h)*(window_n/o.w)*16;
            if(square)CUDA(pearl_square::launch(dest,window_m,window_n,local,window_n/o.w,fold_stream.value));
            else if(vector_rect)CUDA(pearl_rect_vector::launch(dest,window_m,window_n,local,window_n/o.w,fold_stream.value));
            else CUDA(pearl_fold::launch16x32_swizzled(dest,window_m,window_n,local,window_n/o.w,fold_stream.value));
            CUDA(cudaEventRecord(phase[i].end.value,fold_stream.value));
        }
        const int last=(nr-1)/window_group*window_group;
        CUDA(cudaEventSynchronize(phase[last].end.value));
        for(int i=0;i<nr;i+=window_group){
            if(direct){
                float elapsed=0;CUDA(cudaEventElapsedTime(&elapsed,phase[i].start.value,phase[i].end.value));
                gemm_ms+=elapsed;
            }else phase[i].add(gemm_ms,fold_ms,&fold_begin[i]);
        }
        const pearl_ticket::Slab slab{0,size_t(col0/o.w),size_t(o.n/o.w),size_t(o.m/o.h),size_t(window_n/o.w)};
        if(!coverage.append(slab))throw std::runtime_error("incomplete streamed column coverage");
        begin=Clock::now();
        CUDA(pearl_ticket::filter_slab(folded,slab,wire.job,hits.ptr,count.ptr,capacity,nullptr,
            [&](const pearl_ticket::Hit* rec,size_t n) {
                if(!n)return;
                if(square)square_hits->append(col0/window_n,rec,n);
                else if(global_order)small_grid_hits.insert(small_grid_hits.end(),rec,rec+n);
                else wire.append(output.file,rec,n);
            }));
        filter_s+=std::chrono::duration<double>(Clock::now()-begin).count();
    }
    if(square) {
        square_hits->emit([&](const auto* items,size_t n){wire.append(output.file,items,n);});
    } else if(global_order && !small_grid_hits.empty()){
        std::sort(small_grid_hits.begin(),small_grid_hits.end(),[](const auto& l,const auto& r){return l.tile<r.tile;});
        wire.append(output.file,small_grid_hits.data(),small_grid_hits.size());
    }
    wire.finish(output.file,coverage);
    paired_output::finish(roots,output);lt.clear();
    if(owner && !wire.hits()){
        owner->ptrs={arena.detach(),nullptr,scratch.detach(),nullptr,nullptr,nullptr,hits.detach(),count.detach(),invalid.detach(),nullptr};
        pool->keep(std::move(layout),false,required,std::move(owner));
    } else {
        // The proof opener may run as soon as this job is acknowledged.
        invalid.reset(); count.reset(); hits.reset(); scratch.reset(); arena.reset();
    }
    const double wall=std::chrono::duration<double>(Clock::now()-start).count();
    const auto macs=std::uint64_t(o.m)*o.n*o.k;
    std::cout<<std::setprecision(10)<<"{\"device\":\""<<dev.name<<"\",\"mode\":\"native_job\",\"variant\":\""<<o.variant
             <<"\",\"m\":"<<o.m<<",\"n\":"<<o.n<<",\"k\":"<<o.k<<",\"tile_h\":"<<o.h<<",\"tile_w\":"<<o.w
             <<",\"rows_per_pass\":"<<window_m<<",\"cols_per_pass\":"<<window_n<<",\"k_per_pass\":"<<o.k<<",\"row_pitch_bytes\":"<<o.k
             <<",\"resident_operands\":false,\"streamed_b\":true,\"sparse_filter\":true,\"deferred_filter\":true,\"overlapped_fold\":"<<(!direct&&nr>1?"true":"false")
             <<",\"coalesced_fold\":"<<(direct?"false":"true")<<",\"gemm_includes_fold\":"<<(direct?"true":"false")<<",\"threshold_hits\":"<<wire.hits()<<",\"tiles\":"<<coverage.scanned()
             <<",\"passes\":"<<size_t(nr)*(o.n/window_n)<<",\"buffer_pool_warm\":"<<(meter.warm?"true":"false")
             <<",\"buffer_pool_before_bytes\":"<<retained<<",\"explicit_gpu_bytes\":"<<required<<",\"fold_buffer_bytes\":"<<local_words*sizeof(word)
             <<",\"b_commit_mode\":\""<<(panel_commit?"panel":"whole")<<"\",\"b_commit_panel_bytes\":"<<(panel_commit?std::min<size_t>(bytes_b,pearl_panel_b::panel_limit):bytes_b)
             <<",\"b_commit_value_leaves\":"<<(1+(bytes_b-1)/pearl_panel_b::chunk)
             <<",\"free_before_buffers\":"<<free_before<<",\"free_after_buffers\":"<<free_after<<",\"device_total_bytes\":"<<total
             <<",\"generate_commit_s\":"<<commits<<",\"regenerate_b_s\":"<<regen_ms/1000<<",\"noise_quantize_s\":"<<quantize_s
             <<",\"preload_s\":"<<commits+quantize_s+regen_ms/1000<<",\"ticket_wall_s\":"<<filter_s
             <<",\"gemm_ms\":"<<gemm_ms<<",\"fold_ms\":"<<fold_ms<<",\"scan_wall_s\":"<<wall
             <<",\"kernel_tmac_s\":"<<double(macs)/gemm_ms/1e9<<",\"scan_equivalent_th_s\":"<<double(macs)/wall/1e12;
    if(direct)std::cout<<",\"fused_stream_calls\":"<<fused.calls()
        <<",\"cublaslt_calls\":0,\"cublaslt_attempts\":0,\"cublaslt_fallback_calls\":0,\"cutlass_fallback_calls\":0,\"native_fallback_calls\":0"
        <<",\"cublaslt_workspace_bytes\":0,\"raw_product_buffer_bytes\":"<<product_bytes
        <<",\"arena_bytes\":"<<arena_bytes<<",\"job_scratch_bytes\":"<<scratch_bytes<<",\"filter_counter_bytes\":"<<extra;
    if(direct)std::cout<<",\"physical_matrix_kernel_launches\":"<<fused.launches()<<",\"row_windows_per_group\":"<<fused.group();
    std::cout<<",\"prep_inputs\":\""<<prep_inputs::name(prep_mode)<<"\",\"prep_row_metadata_bytes\":"<<meta_bytes
             <<",\"direct_value_leaves\":"<<((prep_mode==prep_inputs::Mode::leaf||prep_mode==prep_inputs::Mode::leafvec)&&panel_commit?"true":"false")
             <<",\"regenerate_inside_quantize\":"<<(prep_inputs::direct_b(prep_mode)?"true":"false")
             <<",\"noise_f_tables_built\":"<<(prep_mode!=prep_inputs::Mode::off&&direct?2:1+o.n/window_n);
    if(vendor)lt.write(std::cout);
    if(!square)std::cout<<",\"rect_fold_kernel\":\"" << (direct?"fused":(vector_rect?"vector":"swizzled")) << "\"";
    if(square)std::cout<<",\"global_ticket_order\":true,\"fold_layout\":\"16x16-row-major\"";
    std::cout<<"}\n";
    return 0;
}

} // namespace column_stream
