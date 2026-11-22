import hashlib
import io
import json
import os
from contextlib import nullcontext, redirect_stderr
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import fp8miner
import lottery
from cuda_worker import (FAST_GEOMETRY, FAST_SETTINGS, STREAM_COLS, STREAM_ROWS, TESTNET2_GEOMETRY,
                         scanner_environment, streamed_variant, validate_stream_request)
from native_scan import Plan, check_headers, check_native_report, synthetic_headers


class FastScanTest(unittest.TestCase):
    def test_visible_device_selects_benchmark_geometry(self):
        ordinary = fp8miner.arguments(["bench"])
        with patch("fp8miner.subprocess.check_output", return_value="testnet2 1280\n"):
            fp8miner.resolve_device_preset(ordinary)
            self.assertEqual(ordinary.strip_rows, fp8miner.RTX_5060_STRIP_ROWS)
            for command in (["bench"], ["mine", "--socket", "unused"]):
                standard = fp8miner.arguments([*command, "--preset", "testnet2"])
                fp8miner.resolve_device_preset(standard)
                self.assertEqual(standard.strip_rows, fp8miner.RTX_5060_STRIP_ROWS)
            self.assertEqual((ordinary.preset, ordinary.m, ordinary.n, ordinary.k),
                             ("testnet2", *TESTNET2_GEOMETRY[:3]))
            for command in (["bench"], ["mine", "--socket", "unused"]):
                explicit = fp8miner.arguments([*command, "--preset", "throughput"])
                with self.assertRaises(fp8miner.UnsupportedPreset):
                    fp8miner.resolve_device_preset(explicit)
        ti = fp8miner.arguments(["bench"])
        with patch("fp8miner.subprocess.check_output", return_value="throughput 1536\n"):
            fp8miner.resolve_device_preset(ti)
            self.assertEqual(ti.strip_rows, fp8miner.FAST_STRIP_ROWS)
        self.assertEqual((ti.preset, ti.m, ti.n, ti.k), ("throughput", *FAST_GEOMETRY[:3]))

    def test_standard_only_profiles_and_board_cap_diagnostics(self):
        for command in (["bench"], ["mine", "--socket", "unused"]):
            with patch("fp8miner.subprocess.check_output", return_value="testnet2 1536"):
                args = fp8miner.arguments([*command, "--preset", "testnet2"])
                fp8miner.resolve_device_preset(args)
                self.assertEqual((args.preset, args.strip_rows), ("testnet2", 1536))
        for raw in ("", "testnet2", "testnet2 13", "other 1536", "testnet2 16384"):
            with self.subTest(raw=raw), patch("fp8miner.subprocess.check_output", return_value=raw):
                with self.assertRaises(ValueError):
                    fp8miner.resolve_device_preset(fp8miner.arguments(["bench"]))
        failure = subprocess.CalledProcessError(78, ["scanner"])
        with patch("fp8miner.subprocess.check_output", side_effect=failure):
            with self.assertRaisesRegex(fp8miner.UnsupportedPreset, "250 W"):
                fp8miner.resolve_device_preset(fp8miner.arguments(["bench"]))

    def test_public_geometry_and_dense_controls(self):
        for command, flags, shape in (
                    ("bench", [], FAST_GEOMETRY),
                    ("bench", ["--preset", "testnet2"], TESTNET2_GEOMETRY),
                    ("mine", ["--socket", "gateway.sock"], TESTNET2_GEOMETRY),
                    ("mine", ["--socket", "gateway.sock", "--preset", "throughput"], FAST_GEOMETRY)):
            with self.subTest(command=command, flags=flags):
                args = fp8miner.arguments([command, *flags])
                self.assertEqual((args.m, args.n, args.k, *map(int, args.tile.split("x"))), shape)
                validate_stream_request(args)
                self.assertEqual(streamed_variant(args, shape), args.variant)
                header, chain = synthetic_headers(0x18010000)
                plan = Plan(shape, bytes(range(32)), header, tuple(chain))
                self.assertEqual(Plan.decode(plan.encode()), plan)
        self.assertEqual(TESTNET2_GEOMETRY, (131072, 131072, 16384, 16, 32))
        self.assertEqual(FAST_GEOMETRY, (1044480, 540672, 13312, 16, 32))

    def test_benchmark_uses_the_selected_shape_for_work_and_warmup(self):
        for preset, shape in (("throughput", FAST_GEOMETRY), ("testnet2", TESTNET2_GEOMETRY)):
            for quick in (False, True):
                with self.subTest(preset=preset, quick=quick):
                    args = fp8miner.arguments(["bench", "--preset", preset, "--jobs", "1"])
                    args.quick = quick
                    with patch.object(fp8miner.secrets, "token_bytes", side_effect=[bytes(32), bytes(range(32))]), \
                            patch.object(fp8miner, "scanner_session", return_value=nullcontext()), \
                            patch.object(fp8miner, "scan_plan", return_value=(None, 0, None)) as scan:
                        result = fp8miner.benchmark(args)
                    dims = (STREAM_ROWS, STREAM_COLS, *shape[2:]) if quick else shape
                    warm = (min(339968, dims[0]), *dims[1:])
                    self.assertEqual([call.args[0].dims for call in scan.call_args_list], [warm, dims])
                    self.assertNotEqual(scan.call_args_list[0].args[0].seed, scan.call_args_list[1].args[0].seed)
                    self.assertEqual(result["dense_macs"], dims[0]*dims[1]*dims[2])
                    self.assertEqual((result["preset"], result["geometry"]), (preset, dims))

    def test_submission_receipt_requires_the_gateway_ack(self):
        from gateway_miner import Job, mine_once
        header, ancestors = synthetic_headers(0x18010000)
        job = Job(header, 0x10000 << (8*(0x18-3)), tuple(ancestors))
        raw = b"synthetic verified payload for a transport test"
        def candidate(prep, args, directory, worker):
            self.assertEqual((prep.m, prep.n, prep.k, prep.tile), (131072, 131072, 16384, "16x32"))
            directory.mkdir()
            (directory / "proof.bin").write_bytes(raw)
            return dict(status="plain_verified", plain="proof.bin", proof_sha256=hashlib.sha256(raw).hexdigest(),
                        proof_bytes=len(raw), tickets=(prep.m//16)*(prep.n//32), hits_checked=1, threshold_hits=1,
                        cpu_recomputed_roots=False, streamed_b=False, streamed_a=True, covered_windows=256,
                        dense_macs=prep.m*prep.n*prep.k, scan_wall_s=1, scan_equivalent_th_s=0)
        miner = SimpleNamespace(native_attempt=candidate, check_headers=check_headers)
        for reply, status in (("submitted", "gateway_queued"), ("accepted", "submission_unconfirmed"),
                              (None, "submission_unconfirmed")):
            with self.subTest(reply=reply), tempfile.TemporaryDirectory() as stage:
                args = fp8miner.arguments(["mine", "--socket", "unused.sock", "--work-dir", stage])
                rpc = Mock()
                rpc.call.side_effect = (job.to_wire(), reply)
                result = mine_once(args, rpc, miner, job, 1)
                self.assertEqual([call.args[0] for call in rpc.call.call_args_list],
                                 ["getMiningInfo", "submitPlainProof"])
                self.assertEqual(result["status"], status)
                self.assertEqual(result["node_acceptance"], "unconfirmed")
                receipt = json.loads((Path(result["proof_dir"]) / "receipt.json").read_text())
                self.assertEqual(receipt, result)

    def test_fp8_inputs_and_rank32_match_the_independent_cpu_reference_at_both_depths(self):
        import numpy as np
        import fast_prepare
        import host_math
        for side, depth in enumerate((FAST_GEOMETRY[2], TESTNET2_GEOMETRY[2])):
            with self.subTest(depth=depth):
                codes = ((np.arange(2 * depth).reshape(2, depth) * 17) % 255 - 127).astype(np.int8)
                seed_e, seed_b = bytes(range(32)), bytes(reversed(range(32)))
                e = host_math.noise_lines(seed_e, side, 0, range(2))
                f = host_math.noise_lines(seed_b, side, 1, range(depth))
                reference = host_math.b200_mm(e, f)
                actual = fast_prepare.noise_mm(e, f)
                np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
                expected, _, _ = host_math.mix_codes(codes, reference, *host_math.unit_scale_norms(codes))
                np.testing.assert_array_equal(host_math.prepare_operand(codes, side, seed_e, seed_b), expected)

    def test_cli_and_controls_work_without_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            "import fp8miner, native_scan; "
            "args=fp8miner.arguments(['bench','--quick']); "
            "h,chain=native_scan.synthetic_headers(0x18010000); "
            "p=native_scan.Plan(fp8miner.FAST_GEOMETRY, bytes(range(32)), h, tuple(chain)); "
            "assert args.fast_profile and native_scan.Plan.decode(p.encode()) == p"
        )
        subprocess.run([sys.executable,"-S","-B","-c",code],cwd=root,check=True)

    def test_short_protocol_hashes_match_independent_blake3(self):
        import blake3
        key = bytes(range(32))
        with patch.object(lottery, "_native_blake3", None):
            for size in (0,1,32,63,64,65,76,83,108,127,128,129,1023,1024):
                data = bytes(i % 251 for i in range(size))
                with self.subTest(size=size):
                    self.assertEqual(lottery.small_digest(data), blake3.blake3(data).digest())
                    self.assertEqual(lottery.small_digest(data,key), blake3.blake3(data,key=key).digest())
            with self.assertRaises(ValueError):
                lottery.small_digest(bytes(1025))

    def test_preset_affects_only_the_child(self):
        args = fp8miner.arguments(["mine","--socket","gateway.sock"])
        with patch.dict(os.environ, {"PEARL_ROW_STRIP":"256"}):
            child = scanner_environment(args)
            self.assertEqual({k:child[k] for k in FAST_SETTINGS}, FAST_SETTINGS)
            self.assertEqual(os.environ["PEARL_ROW_STRIP"], "256")
        self.assertEqual(args.attempts, 0)

    def test_default_console_keeps_job_and_proof_details_local(self):
        from gateway_miner import progress
        receipt = dict(status="gateway_queued",attempt=1,tickets=42,proof_dir="example",
                       header_sha256="header",proof_sha256="proof",error="gateway detail")
        self.assertEqual(progress(receipt,True),dict(status="gateway_queued",attempt=1,tickets=42))
        self.assertEqual(progress(receipt,False),receipt)

    def test_default_errors_do_not_copy_gateway_or_subprocess_details(self):
        from gateway_miner import RpcError
        marker = "private-diagnostic-example"
        errors = (RpcError({"code":-1,"message":marker}),
                  subprocess.CalledProcessError(2,["prover",marker],stderr=marker),
                  FileNotFoundError(marker), RuntimeError(marker), Exception(marker))
        for error in errors:
            with self.subTest(kind=type(error).__name__), patch.object(fp8miner,"main",side_effect=error):
                stream=io.StringIO()
                with redirect_stderr(stream):
                    self.assertEqual(fp8miner.cli([]),1)
                self.assertTrue(stream.getvalue().startswith("error: "))
                self.assertNotIn(marker,stream.getvalue())

    def test_public_parser_hides_invalid_arguments_and_preserves_help(self):
        marker = "synthetic-user-argument"
        error = "error: invalid arguments. Use --help for usage.\n"
        for argv in ([marker], ["bench", "--jobs", marker], ["mine", "--socket", "unused.sock", "--" + marker]):
            with self.subTest(argv=argv):
                stream = io.StringIO()
                with redirect_stderr(stream), self.assertRaises(SystemExit) as failure:
                    fp8miner.cli(argv)
                self.assertEqual(failure.exception.code, 2)
                self.assertEqual(stream.getvalue(), error)
        script = Path(fp8miner.__file__)
        invalid = subprocess.run([sys.executable, "-B", str(script), "bench", "--" + marker],
                                 capture_output=True, text=True)
        self.assertEqual((invalid.returncode, invalid.stdout, invalid.stderr), (2, "", error))
        result = subprocess.run([sys.executable, "-B", str(script), "mine", "--help"], capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertIn("--preset {throughput,testnet2}", result.stdout)

    def test_right_resident_edge_coverage_and_fold_fallback(self):
        for dims, strips, launches, windows in (
                ((4096, 16384, *FAST_GEOMETRY[2:]), 3, 6, 1),
                (TESTNET2_GEOMETRY, 86, 172, 256),
                (FAST_GEOMETRY, 680, 680, 8415)):
            with self.subTest(dims=dims):
                self.check_right_resident_receipt(dims, strips, launches, windows)

    def test_standard_5060_row_strip_controls(self):
        for strip in (1024, 1536, 2048, 3072, 4096):
            with self.subTest(strip=strip):
                count = (TESTNET2_GEOMETRY[0]+strip-1)//strip
                self.check_right_resident_receipt(TESTNET2_GEOMETRY, count, 2*count, 256, strip)
        for invalid in (0, True, 255, 500, 8448):
            args = fp8miner.arguments(["bench", "--preset", "testnet2"])
            args.strip_rows = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scanner_environment(args)

    def check_right_resident_receipt(self, dims, strips, launches, windows, strip=1536):
        m,n,k,h,w = dims
        args = fp8miner.arguments(["bench","--quick"])
        header, chain = synthetic_headers(0x18010000)
        plan = Plan((m,n,k,h,w), bytes(range(32)), header, tuple(chain))
        args.strip_rows = strip
        self.assertEqual(scanner_environment(args)["PEARL_ROW_STRIP"], str(strip))
        macs=m*n*k; folds=(strip//h)*(n//w)*16*4
        arena=n*k+strip*k+folds
        row=dict(mode="native_job",m=m,n=n,k=k,tile_h=h,tile_w=w,variant=args.variant,
                 streamed_a=True,streamed_b=False,resident_operands=False,resident_operand="B",
                 sparse_filter=True,tiles=(m//h)*(n//w),threshold_hits=0,deferred_filter=True,
                 rows_per_pass=4096,cols_per_pass=16384,k_per_pass=k,row_pitch_bytes=k,passes=windows,
                 overlapped_fold=False,coalesced_fold=False,gemm_includes_fold=True,
                 scan_wall_s=2.,generate_commit_s=.1,noise_quantize_s=.2,kernel_tmac_s=macs/1e12,
                 scan_equivalent_th_s=macs/2e12,gemm_ms=1000.,fold_ms=0.,regenerate_b_s=0.,ticket_wall_s=.1,
                 computed_output_cells=m*n,physical_a_strip_rows=strip,physical_a_strips=strips,
                 physical_matrix_kernel_launches=launches,B_reuse_grid_N=1,A_streamed_value_leaves=True,
                 resident_A_segment=128,resident_B_segment=256,requested_native_fold="paired",
                 actual_native_fold="canonical",fold_domain_rows=m+n,fused_stream_calls=windows,
                 raw_product_buffer_bytes=0,cublaslt_calls=0,cublaslt_attempts=0,cublaslt_fallback_calls=0,
                 cutlass_fallback_calls=0,native_fallback_calls=0,cublaslt_workspace_bytes=0,
                 fold_buffer_bytes=folds,rect_fold_kernel="fused",arena_bytes=arena,
                 job_scratch_bytes=128,filter_counter_bytes=256,explicit_gpu_bytes=arena+384)
        check_native_report(row,plan,args,0)
        for field,value in (("physical_matrix_kernel_launches",3),("streamed_b",True),
                            ("computed_output_cells",m*n-1),("requested_native_fold","canonical"),
                            ("explicit_gpu_bytes",arena),("threshold_hits",False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_native_report({**row,field:value},plan,args,0)


if __name__=="__main__":
    unittest.main()
