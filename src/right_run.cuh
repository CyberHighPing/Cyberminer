#pragma once
#include "scan_depth.hpp"
#include "right_plan.cuh"
#include "panel_order.hpp"

namespace right_resident {
inline int run(const Options& o,WorkerBufferCache* pool) {
    constexpr int window_m=4096,window_n=16384,batch=4096;
    if(o.native_job.empty()||!o.resident||!o.deferred||o.variant!=pearl_fused_stream::name||
       o.rows!=window_m||o.cols!=window_n||o.m%window_m||o.n%window_n||o.h!=16||o.w!=32||
       !pearl_geometry::supported_scan_depth(o.k)||o.row_pad||o.k_major||o.fused||!o.dump.empty()||!o.lanes.empty()||
       prep_inputs::selected()!=prep_inputs::Mode::leafvec)
        throw std::invalid_argument("right residency requires a complete aligned native FP8 job and leafvec preparation");
    const auto start=Clock::now();
    const int strip=std::min(o.m,row_strip());
    const auto mode=prep_inputs::Mode::leafvec;
    native_inputs::Plan job(o.native_job,o.m,o.n,o.k,o.h,o.w);
    const size_t a_bytes=size_t(o.m)*o.k,b_bytes=size_t(o.n)*o.k,stage_bytes=size_t(strip)*o.k;
    const char* selection=std::getenv("PEARL_STREAM_A_COMMIT");
    const bool direct_a=selection&&!std::strcmp(selection,"leaves");
    if(selection&&!direct_a&&std::strcmp(selection,"whole"))throw std::invalid_argument("unknown A commitment implementation");
    const size_t commit_tail=job.panel_b_scratch(batch);
    const size_t a_commit_workspace=direct_a?job.panel_a_storage():a_bytes+commit_tail;
    const size_t folded_bytes=size_t(strip/16)*(o.n/32)*16*sizeof(Word);
    const size_t arena_bytes=std::max({job.panel_b_storage(),a_commit_workspace,b_bytes+stage_bytes+folded_bytes});
    const size_t noise_bytes=pearl_noise::scratch_bytes(o.k,batch);
    const size_t row_offset=(noise_bytes+15)&~size_t(15);
    const size_t meta_bytes=(size_t(o.m)+o.n)*sizeof(prep_inputs::Row);
    const size_t scratch_bytes=row_offset+meta_bytes;
    const size_t capacity=o.hit_capacity,extra=capacity*sizeof(pearl_ticket::Hit)+sizeof(pearl_ticket::Count)+sizeof(Word);
    const size_t explicit_bytes=arena_bytes+scratch_bytes+extra;
    if(explicit_bytes>pearl_pool::ceiling)throw std::runtime_error("right residency exceeds 7168 MiB");
    pearl_pool::Meter meter;
    pearl_pool::Shape layout{o.device,{o.m,o.n,o.k,o.h,o.w,o.rows,o.cols,o.k,o.row_pad,o.hit_capacity},
                             {arena_bytes,0,scratch_bytes,0,0,capacity,1,1,0},1u<<22,o.variant};
    if(pool){meter=pool->previous_idle_release();pool->match(&layout,meter);}
    CUDA(cudaSetDevice(o.device));
    cudaDeviceProp dev;CUDA(cudaGetDeviceProperties(&dev,o.device));
    if(std::string(dev.name).find("RTX 5060 Ti")==std::string::npos||dev.major!=12||dev.minor!=0||
       dev.totalGlobalMem<7000ull*1024*1024||dev.totalGlobalMem>9000ull*1024*1024)
        throw std::runtime_error("select an RTX 5060 Ti 8GB");
    size_t free_before,total;CUDA(cudaMemGetInfo(&free_before,&total));
    const size_t retained=pool?pool->retained():0;
    if(!pearl_pool::admit(explicit_bytes,0,false,pool?pool->admission_available(free_before,total):free_before).fits)
        throw std::runtime_error("too little free VRAM for the resident right operand");
    std::unique_ptr<RetainedScanPointers> owner;
    if(pool&&pool->reuse){meter.warm=pool->has();owner=meter.warm?pool->take():std::make_unique<RetainedScanPointers>();}
    auto take=[&](size_t i){return owner?owner->take(i):nullptr;};
    auto* measure=pool?&meter:nullptr;
    Buffer<Byte> arena(arena_bytes,take(0),measure),scratch(scratch_bytes,take(2),measure);
    Buffer<pearl_ticket::Hit> hits(capacity,take(6),measure);
    Buffer<pearl_ticket::Count> count(1,take(7),measure);
    Buffer<Word> invalid(1,take(8),measure);
    const bool packed=packed_selected();
    Plan matrix(packed);
    size_t free_after;CUDA(cudaMemGetInfo(&free_after,&total));
    auto* meta_a=reinterpret_cast<prep_inputs::Row*>(scratch.ptr+row_offset);
    auto* meta_b=meta_a+o.m;
    Byte* b=arena.ptr;
    Byte* a=b+b_bytes;
    auto* folded=reinterpret_cast<Word*>(a+stage_bytes);
    CUDA(cudaMemset(invalid.ptr,0,sizeof(Word)));
    const auto commit_begin=Clock::now();
    const auto hb=job.commit_panel_b(arena.ptr,arena_bytes,meta_b,invalid.ptr,true);
    const auto b_commit_end=Clock::now();
    const auto ha=direct_a?job.commit_stream_role(pearl_jobgen::Side::A,arena.ptr,arena_bytes,meta_a,invalid.ptr,true):job.generate_and_commit(pearl_jobgen::Side::A,arena.ptr,a_bytes,arena.ptr+a_bytes,commit_tail,meta_a,invalid.ptr);
    const auto a_commit_end=Clock::now();
    const auto state=job.bind(ha,hb);
    const auto desired_fold=native_fold::requested();
    matrix.certify_native(meta_a,size_t(o.m)+o.n,o.k,folded,desired_fold);
    const double commits=std::chrono::duration<double>(Clock::now()-commit_begin).count();
    const auto ea=native_inputs::subkey(state.sa,"pearl/v4/FP8/noise-line");
    const auto fb=native_inputs::subkey(state.sb,"pearl/v4/FP8/noise-line");
    const auto ek=pearl_noise::key_from_bytes(native_inputs::bytes(ea)),fk=pearl_noise::key_from_bytes(native_inputs::bytes(fb));
    auto verify_rows=[&] {
        Word bad=0;CUDA(cudaMemcpy(&bad,invalid.ptr,sizeof(bad),cudaMemcpyDeviceToHost));
        if(bad)throw std::runtime_error("regenerated source failed the row policy");
    };
    auto begin=Clock::now();
    const int body=body_columns(o.n);
    if(packed)CUDA(leaf_packed::prepare(b,o.n,o.k,0,body,fk,fk,job.generator_seed(),batch,scratch.ptr,noise_bytes,meta_b,invalid.ptr,true,nullptr));
    else CUDA(prep_inputs::prepare(mode,b,o.n,o.k,1,0,fk,fk,job.generator_seed(),batch,scratch.ptr,noise_bytes,meta_b,invalid.ptr,true,nullptr));
    verify_rows();
    double prepare_seconds=std::chrono::duration<double>(Clock::now()-begin).count();
    double regen_seconds=0,gemm_ms=0,filter_seconds=0;
    paired_output::File output(o.out),roots(o.roots_out);
    std::istringstream header(job.ticket_header(state));
    pearl_ticket::SparseWire wire(header,o.m,o.n,o.k,o.h,o.w);
    pearl_ticket::Coverage coverage(o.m/o.h,o.n/o.w,false);
    wire.begin(output.file);
    const auto receipt=state.receipt();roots.file.write(reinterpret_cast<const char*>(receipt.data()),receipt.size());
    const size_t tile_cols=o.n/o.w,all_tiles=size_t(o.m/o.h)*tile_cols;
    native_width::CanonicalOrder<pearl_ticket::Hit> order(output.borrowed_staging_dir_fd(),o.m/o.h,tile_cols,tile_cols,window_n/o.w,
                                                         all_tiles*16<=pearl_geometry::legacy_words);
    Event prep_start,raw_end,prep_end,gemm_end;
    int strips=0;
    size_t cache_bytes=0;
    {
        CacheWindow cache(a,stage_bytes,dev);cache_bytes=cache.bytes();
        column_stream::DrainOnError drain{nullptr};
        for(int first=0;first<o.m;first+=strip){
            const int rows=std::min(strip,o.m-first);
            CUDA(cudaEventRecord(prep_start.value));
            if(!packed){
                job.regenerate_a(a,stage_bytes,first,rows,nullptr);
                CUDA(cudaEventRecord(raw_end.value));
            }
            if(packed)CUDA(left_pack::prepare(128,false,a,stage_bytes,rows,o.k,first,ek,fk,job.generator_seed(),batch,nullptr,0,
                                              scratch.ptr,noise_bytes,meta_a+first,invalid.ptr,nullptr,first==0));
            else CUDA(prep_inputs::prepare(mode,a,rows,o.k,0,first,ek,fk,job.generator_seed(),batch,scratch.ptr,noise_bytes,meta_a+first,invalid.ptr,first==0,nullptr));
            CUDA(cudaEventRecord(prep_end.value));
            matrix.run(a,b,folded,o.k,rows,o.n);
            CUDA(cudaEventRecord(gemm_end.value));
            CUDA(cudaEventSynchronize(gemm_end.value));
            float ms=0;
            if(!packed){
                CUDA(cudaEventElapsedTime(&ms,prep_start.value,raw_end.value));regen_seconds+=ms/1000.;
            }
            CUDA(cudaEventElapsedTime(&ms,packed?prep_start.value:raw_end.value,prep_end.value));
            prepare_seconds+=ms/1000.;
            CUDA(cudaEventElapsedTime(&ms,prep_end.value,gemm_end.value));gemm_ms+=ms;
            const pearl_ticket::Slab slab{size_t(first/o.h),0,tile_cols,size_t(rows/o.h),tile_cols};
            if(!coverage.append(slab))throw std::runtime_error("overlapping or incomplete row coverage");
            begin=Clock::now();
            CUDA(pearl_ticket::filter_slab(folded,slab,wire.job,hits.ptr,count.ptr,capacity,nullptr,
                 [&](const pearl_ticket::Hit* rec,size_t n){if(n)order.append(0,rec,n);}));
            filter_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
            ++strips;
        }
        verify_rows();
    }
    if(!coverage.complete()||matrix.cells()!=std::uint64_t(o.m)*o.n)throw std::runtime_error("incomplete dense work");
    order.emit([&](const auto* items,size_t n){wire.append(output.file,items,n);});
    wire.finish(output.file,coverage);paired_output::finish(roots,output);
    if(owner&&!wire.hits()){
        owner->ptrs={arena.detach(),nullptr,scratch.detach(),nullptr,nullptr,nullptr,hits.detach(),count.detach(),invalid.detach(),nullptr};
        pool->keep(std::move(layout),false,explicit_bytes,std::move(owner));
    }else{invalid.reset();count.reset();hits.reset();scratch.reset();arena.reset();}
    const double wall=std::chrono::duration<double>(Clock::now()-start).count();
    const auto macs=matrix.cells()*o.k;
    const auto windows=matrix.cells()/(std::uint64_t(window_m)*window_n);
    std::cout<<std::setprecision(10)<<"{\"device\":\""<<dev.name<<"\",\"mode\":\"native_job\",\"variant\":\""<<o.variant
             <<"\",\"m\":"<<o.m<<",\"n\":"<<o.n<<",\"k\":"<<o.k<<",\"tile_h\":"<<o.h<<",\"tile_w\":"<<o.w
             <<",\"rows_per_pass\":"<<window_m<<",\"cols_per_pass\":"<<window_n<<",\"k_per_pass\":"<<o.k<<",\"row_pitch_bytes\":"<<o.k
             <<",\"resident_operands\":false,\"streamed_b\":false,\"streamed_a\":true,\"resident_operand\":\"B\",\"sparse_filter\":true,\"deferred_filter\":true"
             <<",\"overlapped_fold\":false,\"coalesced_fold\":false,\"gemm_includes_fold\":true,\"threshold_hits\":"<<wire.hits()<<",\"tiles\":"<<coverage.scanned()
             <<",\"passes\":"<<windows<<",\"fused_stream_calls\":"<<windows<<",\"buffer_pool_warm\":"<<(meter.warm?"true":"false")
             <<",\"buffer_pool_before_bytes\":"<<retained<<",\"explicit_gpu_bytes\":"<<explicit_bytes<<",\"fold_buffer_bytes\":"<<folded_bytes
             <<",\"b_commit_mode\":\"panel\",\"b_commit_panel_bytes\":"<<std::min<size_t>(b_bytes,pearl_panel_b::panel_limit)
             <<",\"b_commit_value_leaves\":"<<(1+(b_bytes-1)/pearl_panel_b::chunk)
             <<",\"free_before_buffers\":"<<free_before<<",\"free_after_buffers\":"<<free_after<<",\"device_total_bytes\":"<<total
             <<",\"generate_commit_s\":"<<commits<<",\"regenerate_b_s\":0,\"regenerate_a_s\":"<<regen_seconds<<",\"noise_quantize_s\":"<<prepare_seconds
             <<",\"preload_s\":"<<commits+prepare_seconds+regen_seconds<<",\"ticket_wall_s\":"<<filter_seconds
             <<",\"gemm_ms\":"<<gemm_ms<<",\"fold_ms\":0,\"scan_wall_s\":"<<wall<<",\"kernel_tmac_s\":"<<double(macs)/gemm_ms/1e9
             <<",\"scan_equivalent_th_s\":"<<double(macs)/wall/1e12<<",\"cublaslt_calls\":0,\"cublaslt_attempts\":0,\"cublaslt_fallback_calls\":0"
             <<",\"cutlass_fallback_calls\":0,\"native_fallback_calls\":0,\"cublaslt_workspace_bytes\":0,\"raw_product_buffer_bytes\":0"
             <<",\"arena_bytes\":"<<arena_bytes<<",\"job_scratch_bytes\":"<<scratch_bytes<<",\"filter_counter_bytes\":"<<extra
             <<",\"physical_matrix_kernel_launches\":"<<matrix.launches()<<",\"computed_output_cells\":"<<matrix.cells()
             <<",\"B_reuse_grid_N\":"<<direct_main::selected_grid()<<",\"A_strip_M_ctas\":"<<strip/256
             <<",\"A_streamed_value_leaves\":"<<(direct_a?"true":"false")<<",\"A_commit_workspace_bytes\":"<<a_commit_workspace
             <<",\"A_commit_s\":"<<std::chrono::duration<double>(a_commit_end-b_commit_end).count()
             <<",\"B_commit_s\":"<<std::chrono::duration<double>(b_commit_end-commit_begin).count()
             <<",\"physical_a_strip_rows\":"<<strip<<",\"physical_a_strips\":"<<strips<<",\"persisting_l2_bytes\":"<<cache_bytes
             <<",\"prep_inputs\":\"leafvec\",\"prep_row_metadata_bytes\":"<<meta_bytes<<",\"direct_value_leaves\":true,\"regenerate_inside_quantize\":true"
             <<",\"requested_native_fold\":\""<<native_fold::name(desired_fold)<<"\",\"actual_native_fold\":\""<<native_fold::name(matrix.fold())<<"\""
             <<",\"fold_domain_rows\":"<<(desired_fold==native_fold::Mode::canonical||!packed?0:size_t(o.m)+o.n)
             <<",\"resident_A_segment\":"<<(packed?128:0)<<",\"resident_B_segment\":"<<(packed?256:0)
             <<",\"resident_B_body_rows\":"<<body<<",\"resident_B_tail_rows\":"<<o.n-body
             <<",\"resident_pack_extra_device_bytes\":0,\"A_XOF_in_quantization\":"<<(packed?"true":"false")
             <<",\"noise_f_tables_built\":2,\"rect_fold_kernel\":\"fused\"}\n";
    return 0;
}
}
