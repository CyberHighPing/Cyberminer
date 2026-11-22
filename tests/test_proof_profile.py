"""Verify the public proof mode reaches both native children without changing the caller's environment."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cuda_worker
import fp8miner
import gpu_proof
import host_miner
import native_job
import native_scan


class ProofProfileTest(unittest.TestCase):
    def test_default_and_disabled_mode_reach_scanner_and_opening(self):
        key = cuda_worker.CHECKPOINT_SETTING
        for requested, expected in ((None, "1"), ("0", "0"), ("1", "1")):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
                os.environ.pop(key, None)
                if requested is not None:
                    os.environ[key] = requested
                args = fp8miner.arguments(["mine", "--socket", "example.sock", "--attempts", "1"])
                header, parents = native_scan.synthetic_headers(0x1a400000)
                observed = {}

                def scan(plan, options, stage, worker):
                    observed["scanner"] = cuda_worker.scanner_environment(options)[key]
                    roots = native_scan.derive(plan, bytes(32), bytes(32))
                    return roots, 1, {"streamed_b":False, "passes":256, "scan_wall_s":1, "scan_equivalent_th_s":0}

                def opening(command, **options):
                    observed["opener"] = options["env"][key]
                    Path(command[-1]).write_bytes(b"example")
                    return SimpleNamespace(stdout=json.dumps({"tile":0, "proof_bytes":7}))

                verifier = SimpleNamespace(__file__=__file__, IncompleteBlockHeader=SimpleNamespace(from_bytes=lambda raw:raw))
                prep = SimpleNamespace(m=args.m,n=args.n,k=args.k,tile=args.tile,seed=None)
                one_hit = lambda *unused: iter(((0,bytes(64),bytes(32)),))
                with patch.object(host_miner,"headers",return_value=(header,parents,None)), \
                     patch.object(host_miner,"ticket_records",side_effect=one_hit), \
                     patch.object(host_miner,"import_verifier",return_value=verifier), \
                     patch.object(gpu_proof,"verify_candidate",return_value=True), \
                     patch.object(native_job,"scan_plan",side_effect=scan), \
                     patch.object(gpu_proof.subprocess,"run",side_effect=opening):
                    result = native_job.attempt(prep,args,Path(tmp)/"result")
                self.assertEqual(result["status"],"plain_verified")
                self.assertEqual(observed, {"scanner":expected,"opener":expected})
                self.assertEqual(os.environ.get(key),requested)

    def test_bench_other_preset_and_advanced_defaults(self):
        with patch.dict(os.environ):
            os.environ.pop(cuda_worker.CHECKPOINT_SETTING,None)
            for command in (["bench","--preset","testnet2"],["mine","--socket","example.sock","--preset","throughput"]):
                args=fp8miner.arguments(command)
                self.assertEqual(args.proof_checkpoints,"0")
            self.assertIsNone(cuda_worker.proof_environment(SimpleNamespace()))
            self.assertIsNone(cuda_worker.scanner_environment(SimpleNamespace()))

    def test_invalid_override_exits_without_echoing_value(self):
        for value in ("", "invalid-setting"):
            with self.subTest(value=value),patch.dict(os.environ,{cuda_worker.CHECKPOINT_SETTING:value}):
                err=io.StringIO()
                with self.assertRaises(SystemExit) as exit_code,redirect_stderr(err):
                    fp8miner.arguments(["mine","--socket","example.sock"])
                self.assertEqual(exit_code.exception.code,2)
                self.assertEqual(err.getvalue(),"error: invalid arguments. Use --help for usage.\n")


if __name__=="__main__":
    unittest.main()
