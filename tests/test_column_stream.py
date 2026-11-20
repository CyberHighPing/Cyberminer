"""Exercise public streamed-B host paths with synthetic controls and a CPU-only scanner double."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import cuda_worker
import gateway_miner
import host_miner
import native_job


ROWS,COLS=4096,16384
DIMS=(4*ROWS,2*COLS,32768,16,32)
TICKET_COUNT=(DIMS[0]//DIMS[3])*(DIMS[1]//DIMS[4])
WINDOW_COUNT=(DIMS[0]//ROWS)*(DIMS[1]//COLS)
VENDOR,REFERENCE,FUSED=cuda_worker.STREAM_VARIANTS


def args(**values):
    options=dict(m=DIMS[0],n=DIMS[1],k=DIMS[2],tile="16x32",stream_b=True,native_dense=True,
                 scanner=Path('synthetic-unused-scanner'),cpu_reference=False,command='mine',attempts=1,
                 variant=None,rows_per_pass=None,cols_per_pass=None,native_dir=None,job=None,
                 seed='13'*32,bits='03000001',ancestor_depth=1)
    return argparse.Namespace(**(options|values))


def make_report(dims, variant):
    m,n,k,h,w=dims
    tiles=(m//h)*(n//w)
    windows=(m//ROWS)*(n//COLS)
    value=dict(mode='native_job',variant=variant,m=m,n=n,k=k,tile_h=h,tile_w=w,rows_per_pass=ROWS,
               cols_per_pass=COLS,k_per_pass=k,row_pitch_bytes=k,streamed_b=True,resident_operands=False,
               sparse_filter=True,deferred_filter=True,overlapped_fold=m>ROWS,coalesced_fold=True,gemm_includes_fold=False,
               tiles=tiles,threshold_hits=0,passes=windows,scan_wall_s=31.0,generate_commit_s=1.0,
               noise_quantize_s=0.5,gemm_ms=29000.0,fold_ms=250.0,regenerate_b_s=.1,ticket_wall_s=.2,
               kernel_tmac_s=m*n*k/29000/1e9,scan_equivalent_th_s=m*n*k/31/1e12)
    if variant==VENDOR:
        value.update(cublaslt_calls=windows-1,cublaslt_attempts=windows,cublaslt_fallback_calls=1,
                     cutlass_fallback_calls=1,native_fallback_calls=0)
    return value


class FakeScanner:
    def __init__(self):
        self.flags=[]
        self.closed=False
        self.drop_ticket=False
    def close(self,graceful=True): self.closed=True
    def request(self,flags):
        self.flags=list(flags)
        def option(name): return flags[flags.index(name)+1]
        plan=native_job.read_control(Path(option('--native-job')))
        roots=native_job.derive(plan,b'A'*32,b'B'*32)
        Path(option('--roots-out')).write_bytes(native_job.ROOTS.pack(native_job.ROOTS_MAGIC,*roots))
        out=bytearray(native_job.params(plan,roots))
        report=make_report(plan.dims,option('--variant'))
        struct.pack_into('<QQ',out,48,report['tiles']-int(self.drop_ticket),0)
        Path(option('--out')).write_bytes(out)
        return report


class ColumnStreamTest(unittest.TestCase):
    def test_square_tickets_use_the_checked_depth(self):
        dims=(2*ROWS,2*COLS,16384,16,16)
        opt=args(m=dims[0],n=dims[1],k=dims[2],tile="16x16",variant=VENDOR)
        self.assertEqual(cuda_worker.streamed_variant(opt,dims),VENDOR)
        native_job.check_native_report(make_report(dims,VENDOR),argparse.Namespace(dims=dims),opt,0)
        with self.assertRaises(ValueError):
            cuda_worker.streamed_variant(opt,(dims[0],dims[1],32768,16,16))

    def test_fused_rectangular_stream_reports_each_launch_and_live_byte(self):
        for depth in (16384,32768):
            dims=(4*ROWS,2*COLS,depth,16,32); m,n,k,h,w=dims
            opt=args(k=depth,variant=FUSED);plan=argparse.Namespace(dims=dims)
            self.assertEqual(cuda_worker.streamed_variant(opt,dims),FUSED)
            report=make_report(dims,FUSED)
            fold_bytes=(m//h)*(COLS//w)*16*4
            arena=m*k+COLS*k+fold_bytes
            report.update(gemm_includes_fold=True,coalesced_fold=False,overlapped_fold=False,rect_fold_kernel="fused",
                          fold_ms=0.0,fused_stream_calls=report["passes"],fold_buffer_bytes=fold_bytes,
                          raw_product_buffer_bytes=0,cublaslt_calls=0,cublaslt_attempts=0,cublaslt_fallback_calls=0,
                          cutlass_fallback_calls=0,native_fallback_calls=0,cublaslt_workspace_bytes=0,
                          arena_bytes=arena,job_scratch_bytes=1024,filter_counter_bytes=2048,explicit_gpu_bytes=arena+3072)
            native_job.check_native_report(report,plan,opt,0)
            for key,value in {"fused_stream_calls":0,"cublaslt_calls":1,"raw_product_buffer_bytes":4,"fold_buffer_bytes":0,
                              "explicit_gpu_bytes":1,"fold_ms":1.0,"gemm_includes_fold":False}.items():
                with self.subTest(key=key), self.assertRaises(ValueError):
                    native_job.check_native_report(dict(report,**{key:value}),plan,opt,0)
        with self.assertRaises(ValueError):
            cuda_worker.streamed_variant(args(variant=FUSED),(2*ROWS,2*COLS,16384,16,16))
        with self.assertRaises(ValueError):
            cuda_worker.validate_stream_request(args(variant=FUSED,stream_b=False))

    def test_vendor_rejects_contradictory_or_noninteger_fused_counts(self):
        plan=argparse.Namespace(dims=DIMS);opt=args(variant=VENDOR)
        original=make_report(DIMS,VENDOR)
        native_job.check_native_report(original,plan,opt,0)
        native_job.check_native_report(dict(original,fused_stream_calls=0),plan,opt,0)
        for count in (1,-1,True,False,None,"0"):
            with self.subTest(count=count),self.assertRaises(ValueError):
                native_job.check_native_report(dict(original,fused_stream_calls=count),plan,opt,0)

    def test_one_row_window_has_no_overlapping_fold(self):
        dims=(ROWS,2*COLS,16384,16,32)
        report=make_report(dims,VENDOR)
        native_job.check_native_report(report,argparse.Namespace(dims=dims),args(variant=VENDOR),0)
        report["overlapped_fold"]=True
        with self.assertRaises(ValueError):
            native_job.check_native_report(report,argparse.Namespace(dims=dims),args(variant=VENDOR),0)

    def setUp(self):
        scratch=tempfile.TemporaryDirectory(prefix='fp8-stream-test-')
        self.addCleanup(scratch.cleanup)
        self.dir=Path(scratch.name)

    def test_worker_options_accept_both_variants_and_reject_incompatible_inputs(self):
        for variant in (None,'auto',VENDOR,REFERENCE):
            a=args(variant=variant)
            with cuda_worker.scanner_session(a) as worker: self.assertIsNone(worker)
            self.assertEqual(cuda_worker.streamed_variant(a,DIMS),REFERENCE if variant==REFERENCE else VENDOR)
        for wrong in (dict(native_dense=False),dict(cpu_reference=True),dict(scanner=None),dict(variant='row64'),
                      dict(m=4097),dict(n=12345),dict(k=65536),dict(tile='4x128'),dict(rows_per_pass=512),
                      dict(cols_per_pass=32768),dict(k_per_pass=1024),dict(row_pad=16),dict(fuse_fold=True)):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError): cuda_worker.scanner_session(args(**wrong))
        with self.assertRaises(ValueError): cuda_worker.scan_options(args(),None)
        self.assertEqual(cuda_worker.scan_options(args(stream_b=False),None),{})

    def test_public_cli_and_lower_k_geometry(self):
        dimensions=(8192,32768,16384,16,32)
        scanner=self.dir/'synthetic-scanner'
        scanner.touch()
        flags=['--scanner',str(scanner),'--native-dense','--stream-b','--m','8192','--n','32768',
               '--k','16384','--tile','16x32','--variant',REFERENCE]
        with patch.object(sys,'argv',['host_miner.py','mine',str(self.dir/'work'),*flags]):
            a=host_miner.arguments()
        self.assertTrue(a.stream_b)
        cuda_worker.validate_stream_request(a)
        native_job.check_native_report(make_report(dimensions,REFERENCE),argparse.Namespace(dims=dimensions),a,0)
        cmd=['gateway_miner.py','--work-dir',str(self.dir/'gateway-work'),'--socket',str(self.dir/'synthetic-socket'),*flags]
        with patch.object(sys,'argv',cmd), patch('host_miner.import_verifier') as verifier, \
             patch('gateway_miner.socket.socket') as socket_factory, patch('gateway_miner.run',return_value=0) as run:
            self.assertEqual(gateway_miner.main(),0)
            verifier.assert_called_once()
            socket_factory.return_value.connect.assert_called_once_with(str(self.dir/'synthetic-socket'))
            self.assertTrue(run.call_args.args[0].stream_b)
        cmd.remove('--native-dense')
        with patch.object(sys,'argv',cmd), patch('gateway_miner.socket.socket') as sockets, self.assertRaises(ValueError):
            try: gateway_miner.main()
            finally: sockets.assert_not_called()

    def test_full_work_signature_and_every_window_and_ticket_are_checked(self):
        plan=argparse.Namespace(dims=DIMS)
        for variant in (VENDOR,REFERENCE):
            report=make_report(DIMS,variant)
            native_job.check_native_report(report,plan,args(variant=variant),0)
            faults=dict(streamed_b=False,resident_operands=True,passes=WINDOW_COUNT-1,tiles=TICKET_COUNT-1,
                        threshold_hits=True,rows_per_pass=8192,cols_per_pass=4096,k_per_pass=16384,
                        kernel_tmac_s=1.0,scan_equivalent_th_s=1000.0,gemm_ms=0,
                        regenerate_b_s=float('nan'),overlapped_fold=False,coalesced_fold=False)
            if variant==VENDOR: faults.update(cublaslt_attempts=True,cutlass_fallback_calls=2,cublaslt_calls=0)
            for key,value in faults.items():
                with self.subTest(variant=variant,field=key), self.assertRaises(ValueError):
                    native_job.check_native_report(report|{key:value},plan,args(variant=variant),0)
            with self.assertRaises(ValueError): native_job.check_native_report(report,plan,args(stream_b=False),0)
            with self.assertRaises(ValueError): native_job.check_native_report(report|{'streamed_b':1},plan,args(),0)

    def test_host_native_scan_and_independent_inspect(self):
        a=args(directory=self.dir/'mine',variant=REFERENCE)
        fake=FakeScanner()
        with redirect_stdout(io.StringIO()): self.assertEqual(host_miner.execute(a,fake),2)
        self.assertEqual(fake.flags.count('--stream-b'),1)
        self.assertEqual(fake.flags.count('--defer-filter'),1)
        self.assertEqual(fake.flags[fake.flags.index('--rows-per-pass')+1],str(ROWS))
        self.assertEqual(fake.flags[fake.flags.index('--cols-per-pass')+1],str(COLS))
        stage=a.directory/'attempt-0000'
        command=['native_job.py','inspect',*(str(stage/n) for n in ('control.bin','roots.bin','hits.phits')),
                 '--stream-b','--scan-report',str(stage/'scan-report.json')]
        out=io.StringIO()
        with patch.object(sys,'argv',command), redirect_stdout(out): self.assertEqual(native_job.main(),0)
        result=json.loads(out.getvalue())
        self.assertEqual((result['covered_tickets'],result['covered_windows'],result['dense_macs']),
                         (TICKET_COUNT,WINDOW_COUNT,DIMS[0]*DIMS[1]*DIMS[2]))
        fail=command+['--variant',VENDOR]
        with patch.object(sys,'argv',fail), self.assertRaises(ValueError): native_job.main()

    def test_gateway_uses_streaming_and_preserves_no_hit_semantics(self):
        a=args(work_dir=self.dir)
        head,ancestors,_=host_miner.headers(a)
        job=gateway_miner.Job(head,1,tuple(ancestors))
        fake=FakeScanner()
        class NoSubmission:
            def call(self,*_args): raise AssertionError('zero-hit job attempted a submission')
        with redirect_stdout(io.StringIO()): summary=gateway_miner.mine_once(a,NoSubmission(),host_miner,job,1,fake)
        self.assertEqual(summary['status'],'no_difficulty_hit')
        self.assertTrue(summary['streamed_b'])
        self.assertEqual((summary['tickets'],summary['covered_windows']),(TICKET_COUNT,WINDOW_COUNT))
        self.assertEqual(fake.flags[fake.flags.index('--variant')+1],VENDOR)

    def test_one_shot_scanner_and_incomplete_file_fails_without_publishing(self):
        fake=FakeScanner()
        a=args()
        def command(argv,**kwargs):
            return argparse.Namespace(stdout=json.dumps(fake.request(argv[1:])))
        with patch('native_scan.subprocess.run',side_effect=command), redirect_stdout(io.StringIO()):
            result=native_job.attempt(a,a,self.dir/'one')
        self.assertEqual(result['tickets'],TICKET_COUNT)
        fake.drop_ticket=True
        with self.assertRaisesRegex(ValueError,'cover exactly'):
            native_job.attempt(a,a,self.dir/'bad',fake)
        self.assertFalse((self.dir/'bad').exists())
        self.assertTrue(fake.closed)

    def test_layout_independent_control_and_old_resident_receipts(self):
        common=['--m',str(DIMS[0]),'--n',str(DIMS[1]),'--k',str(DIMS[2]),'--tile','16x32','--seed','52'*32]
        for mode in ('old','new'):
            argv=['native_job.py','plan',str(self.dir/mode),*common]
            if mode=='new':argv.append('--stream-b')
            with patch.object(sys,'argv',argv), redirect_stdout(io.StringIO()): self.assertEqual(native_job.main(),0)
        self.assertEqual((self.dir/'old/control.bin').read_bytes(),(self.dir/'new/control.bin').read_bytes())
        r=make_report(DIMS,REFERENCE)
        r.pop('streamed_b'); r['resident_operands']=True
        native_job.check_native_report(r,argparse.Namespace(dims=DIMS),args(stream_b=False,variant=REFERENCE),0)
        meta=dict(zip(('m','n','k','h','w'),DIMS))
        r.update(mode='file_scan')
        cuda_worker.check_file_scan(r,meta,True,True,REFERENCE,0)
        with self.assertRaises(cuda_worker.WorkerError):
            cuda_worker.check_file_scan(r|{'streamed_b':True},meta,True,True,REFERENCE,0)


if __name__=='__main__': unittest.main()
