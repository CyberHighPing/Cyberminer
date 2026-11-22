#pragma once

namespace panel_reference {
inline int run(const Options& o) {
    constexpr int H=4096,W=16384;
    if(o.native_job.empty()||!o.resident||!o.deferred||o.variant!=verified_lt::name||
       o.rows!=H||o.cols!=W||o.m%H||o.n%W||o.h!=16||o.w!=32||!verified_lt::eligible(H,W,o.k)||
       o.row_pad||o.k_major||o.fused||!o.dump.empty()||!o.lanes.empty())
        throw std::invalid_argument("reference needs canonical aligned native dense jobs");
    CUDA(cudaSetDevice(o.device));
    cudaDeviceProp dev{};CUDA(cudaGetDeviceProperties(&dev,o.device));
    using Byte=std::uint8_t;using Word=std::uint32_t;using Side=pearl_jobgen::Side;
    const auto start=Clock::now();
    native_inputs::Plan job(o.native_job,o.m,o.n,o.k,o.h,o.w);
    const size_t b_size=size_t(W)*o.k,a_size=size_t(H)*o.k,product_size=size_t(H)*W*sizeof(Word),fold_size=size_t(H)*W/32*sizeof(Word);
    const size_t arena_size=std::max({job.panel_a_storage(),job.panel_b_storage(),a_size+b_size+product_size+fold_size+verified_lt::offered_workspace});
    const size_t noise=pearl_noise::scratch_bytes(o.k,H),capacity=o.hit_capacity;
    const size_t explicit_bytes=arena_size+noise+capacity*sizeof(pearl_ticket::Hit)+sizeof(pearl_ticket::Count)+sizeof(Word);
    size_t free,total;CUDA(cudaMemGetInfo(&free,&total));
    if(explicit_bytes>pearl_pool::ceiling||explicit_bytes+size_t(96)*1024*1024>free)throw std::runtime_error("reference workspace cannot fit");
    Buffer<Byte> arena(arena_size),scratch(noise);
    Buffer<pearl_ticket::Hit> hits(capacity);Buffer<pearl_ticket::Count> count(1);Buffer<Word> invalid(1);
    column_stream::DrainOnError drain{nullptr};CUDA(cudaMemset(invalid.ptr,0,sizeof(Word)));
    const auto hb=job.commit_stream_role(Side::B,arena.ptr,arena_size,nullptr,nullptr,false);
    const auto ha=job.commit_stream_role(Side::A,arena.ptr,arena_size,nullptr,nullptr,false);
    const auto state=job.bind(ha,hb);
    const auto ad=native_inputs::subkey(state.sa,"pearl/v4/FP8/noise-line"),bd=native_inputs::subkey(state.sb,"pearl/v4/FP8/noise-line");
    const auto ak=pearl_noise::key_from_bytes(native_inputs::bytes(ad)),bk=pearl_noise::key_from_bytes(native_inputs::bytes(bd));
    const double commit_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    Byte* b=arena.ptr;Byte* a=b+b_size;
    auto* matrix=reinterpret_cast<Word*>(a+a_size);
    auto* folded=reinterpret_cast<Word*>(reinterpret_cast<Byte*>(matrix)+product_size);
    Byte* work=reinterpret_cast<Byte*>(folded)+fold_size;
    verified_lt::Plan plan;plan.prepare(H,W,o.k);
    if(!plan.ready())throw std::runtime_error("pinned cuBLASLt algorithm unavailable");
    plan.bind_workspace(work,verified_lt::offered_workspace);
    const size_t tile_cols=o.n/o.w,all_tiles=size_t(o.m/o.h)*tile_cols;
    paired_output::File output(o.out),roots(o.roots_out);
    std::istringstream header(job.ticket_header(state));pearl_ticket::SparseWire wire(header,o.m,o.n,o.k,o.h,o.w);
    pearl_ticket::Coverage coverage(o.m/o.h,o.n/o.w,true);
    native_width::CanonicalOrder<pearl_ticket::Hit> order(output.borrowed_staging_dir_fd(),o.m/o.h,tile_cols,W/o.w,W/o.w,all_tiles*16<=pearl_geometry::legacy_words);
    wire.begin(output.file);const auto receipt=state.receipt();roots.file.write(reinterpret_cast<const char*>(receipt.data()),receipt.size());
    auto original=[&](Byte* target,int rows,int first,bool right){
        const size_t size=size_t(rows)*o.k;
        if(right)job.regenerate_b(target,size,first,rows);else job.regenerate_a(target,size,first,rows,nullptr);
        const auto& key=right?bk:ak;
        CUDA(pearl_noise::prepare_resident(reinterpret_cast<int8_t*>(target),rows,o.k,right?1:0,reinterpret_cast<const Byte*>(key.word),
                 reinterpret_cast<const Byte*>(bk.word),H,scratch.ptr,noise,invalid.ptr,nullptr,first));
    };
    Event started,done,extracted;
    size_t verified_windows=0;double gemm_ms=0,fold_ms=0;
    for(int col=0;col<o.n;col+=W) {
        original(b,W,col,true);
        for(int row=0;row<o.m;row+=H) {
            original(a,H,row,false);
            CUDA(cudaEventRecord(started.value));
            if(!plan.run(a,b,matrix,H,W,o.k))throw std::runtime_error("cuBLASLt fallback in reference");
            CUDA(cudaEventRecord(done.value));
            CUDA(pearl_rect_vector::launch(matrix,H,W,folded,W/o.w,nullptr));
            CUDA(cudaEventRecord(extracted.value));
            const pearl_ticket::Slab slab{size_t(row/o.h),size_t(col/o.w),tile_cols,H/size_t(o.h),W/size_t(o.w)};
            if(!coverage.append(slab))throw std::runtime_error("invalid reference coverage");
            CUDA(pearl_ticket::filter_slab(folded,slab,wire.job,hits.ptr,count.ptr,capacity,nullptr,
                [&](const pearl_ticket::Hit* records,size_t size){if(size)order.append(col/W,records,size);}));
            float ms=0;CUDA(cudaEventElapsedTime(&ms,started.value,done.value));gemm_ms+=ms;
            CUDA(cudaEventElapsedTime(&ms,done.value,extracted.value));fold_ms+=ms;++verified_windows;
        }
    }
    Word bad=1;CUDA(cudaMemcpy(&bad,invalid.ptr,4,cudaMemcpyDeviceToHost));
    if(bad||!coverage.complete()||verified_windows!=size_t(o.m/H)*(o.n/W))throw std::runtime_error("incomplete or invalid independent dense job");
    order.emit([&](const pearl_ticket::Hit* records,size_t size){wire.append(output.file,records,size);});
    wire.finish(output.file,coverage);paired_output::finish(roots,output);
    std::cout<<std::setprecision(12)<<"{\"device\":\""<<dev.name<<"\",\"variant\":\""<<verified_lt::name<<"\",\"mode\":\"native_job\""
       <<",\"threshold_hits\":"<<wire.hits()<<",\"tiles\":"<<coverage.scanned()<<",\"computed_output_cells\":"<<std::uint64_t(o.m)*o.n
       <<",\"gemm_ms\":"<<gemm_ms<<",\"fold_ms\":"<<fold_ms<<",\"generate_commit_s\":"<<commit_seconds
       <<",\"reference_panel_windows\":"<<verified_windows<<",\"explicit_gpu_bytes\":"<<explicit_bytes
       <<",\"scan_wall_s\":"<<std::chrono::duration<double>(Clock::now()-start).count();
    plan.write(std::cout);std::cout<<"}\n";return 0;
}
}
