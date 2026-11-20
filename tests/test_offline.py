"""Exercise the command-line interfaces using only data generated for this run."""

import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SMALL = ("--m", "8", "--n", "64", "--k", "1024", "--tile", "4x64")
GPU_SHAPE = ("--m", "256", "--n", "256", "--k", "1024", "--tile", "16x32")
VARIANT = "cutlass120-coop-swizzle-pipeline"
VENDOR_VARIANT = "cublaslt-verified-pipeline"


class OfflineMinerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fp8miner-test-")
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name)

    def invoke(self, script, *args, expected=0):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / script), *map(str, args)],
            cwd=self.scratch, text=True, capture_output=True, timeout=120,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, expected, result.stderr)
        return result

    def plan(self, name, shape=SMALL, bits="207fffff"):
        directory = self.scratch / name
        result = self.invoke("native_job.py", "plan", directory, *shape,
                             "--bits", bits, "--seed", secrets.token_hex(32))
        description = json.loads(result.stdout)
        self.assertEqual(description["status"], "planned")
        self.assertEqual(description["source"], "synthetic offline job")
        return directory / "control.bin"

    def test_roots_and_corruption(self):
        control = self.plan("input")
        reference = self.scratch / "reference"
        self.invoke("native_job.py", "reference", control, reference)
        roots = reference / "roots.bin"
        verified = self.scratch / "verified"
        result = self.invoke("native_job.py", "verify", control, roots, verified)
        self.assertTrue(json.loads(result.stdout)["cpu_recomputed_roots"])
        self.assertEqual(roots.read_bytes(), (verified / "roots.bin").read_bytes())
        damaged = bytearray(roots.read_bytes())
        damaged[8] ^= 1
        corrupt = self.scratch / "altered.bin"
        corrupt.write_bytes(damaged)
        rejected = self.scratch / "rejected"
        self.invoke("native_job.py", "verify", control, corrupt, rejected, expected=1)
        self.assertFalse(rejected.exists())

    def test_folds_and_filter(self):
        job = self.scratch / "case"
        self.invoke("host_miner.py", "prepare", job, *SMALL,
                    "--bits", "207fffff", "--seed", secrets.token_hex(32))
        self.invoke("host_miner.py", "scan", job, "--cpu-reference")
        self.assertEqual((job / "folds.bin").stat().st_size, 2 * 64)
        result = self.invoke("host_miner.py", "tickets", job)
        hits = json.loads(result.stdout)["hits"]
        self.assertEqual([item["tile"] for item in hits], [0, 1])
        self.assertTrue(all(len(bytes.fromhex(item["digest_le"])) == 32 for item in hits))

    def test_mine_command_reports_exhaustion(self):
        result = self.invoke("host_miner.py", "mine", self.scratch / "attempts", *SMALL,
                             "--bits", "03000001", "--cpu-reference", "--attempts", "1",
                             "--seed", secrets.token_hex(32), expected=2)
        receipt = json.loads(result.stdout)
        self.assertEqual((receipt["status"], receipt["tickets"], receipt["hits_checked"]),
                         ("no_difficulty_hit", 2, 0))

    @unittest.skipUnless(shutil.which("c++"), "requires a host C++17 compiler")
    def test_panel_boundaries_preserve_rows_and_merkle_leaves(self):
        program = self.scratch / ("panel.exe" if os.name == "nt" else "panel")
        subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(ROOT / "src"),
                        str(ROOT / "tests" / "panel_boundary.cpp"), "-o", str(program)],
                       check=True, timeout=60, cwd=self.scratch)
        subprocess.run([str(program)], check=True, timeout=30, cwd=self.scratch)

    @unittest.skipUnless(shutil.which("c++"), "requires a host C++17 compiler")
    def test_deferred_geometry_budget_and_order(self):
        program = self.scratch / ("geometry.exe" if os.name == "nt" else "geometry")
        subprocess.run(["c++", "-std=c++17", "-I", str(ROOT / "src"),
                        str(ROOT / "tests" / "geometry_offline.cpp"), "-o", str(program)],
                       check=True, timeout=60, cwd=self.scratch)
        subprocess.run([str(program)], check=True, timeout=10, cwd=self.scratch)

    @unittest.skipUnless(os.environ.get("FP8MINER_TEST_GPU") == "1", "requires explicit GPU opt-in")
    def test_device_commitments_and_pool_equivalence(self):
        from cuda_worker import CudaWorker

        scanner = ROOT / "build" / "pearl-fp8-scan"
        self.assertTrue(scanner.is_file(), "run bash build.sh first")
        controls = [self.plan(f"input-{i}", GPU_SHAPE, bits) for i, bits in
                    enumerate(("03000001", "03000001", "207fffff"))]
        outputs = {}
        for mode in ("off", "reuse"):
            with patch.dict(os.environ, PEARL_SCAN_BUFFER_POOL=mode), CudaWorker(scanner) as worker:
                for i, control in enumerate(controls):
                    roots = self.scratch / f"roots-{mode}-{i}.bin"
                    hits = self.scratch / f"hits-{mode}-{i}.bin"
                    arguments = ["--native-job", str(control), *GPU_SHAPE, "--variant", VARIANT,
                                 "--roots-out", str(roots), "--out", str(hits), "--defer-filter",
                                 "--rows-per-pass", "128", "--cols-per-pass", "128", "--k-per-pass", "0"]
                    reply = worker.request(arguments)
                    self.assertEqual(reply["threshold_hits"], 128 if i == 2 else 0)
                    if mode == "off":
                        self.assertNotIn("buffer_pool_mode", reply)
                    else:
                        self.assertEqual(reply["buffer_pool_mode"], "reuse")
                        if i == 2:
                            self.assertEqual(reply["buffer_pool_after_bytes"], 0)
                    outputs[mode, i] = (roots.read_bytes(), hits.read_bytes())
                    if mode == "off":
                        self.invoke("native_job.py", "verify", control, roots, self.scratch / f"verified-{i}")
                        audited = self.invoke("native_job.py", "inspect", control, roots, hits)
                        self.assertEqual(json.loads(audited.stdout)["checked_digest_records"], reply["threshold_hits"])
        for i in range(len(controls)):
            self.assertEqual(outputs["off", i], outputs["reuse", i])

    @unittest.skipUnless(os.environ.get("FP8MINER_TEST_CUBLASLT") == "1", "requires explicit cuBLASLt GPU opt-in")
    def test_vendor_window_against_cutlass(self):
        from cuda_worker import CudaWorker

        scanner = ROOT / "build" / "pearl-fp8-scan"
        self.assertTrue(scanner.is_file(), "run bash build.sh first")
        shape = ("--m", "4096", "--n", "16384", "--k", "16384", "--tile", "16x32")
        control = self.plan("vendor-input", shape, bits="1c080000")
        outputs = []
        with CudaWorker(scanner) as worker:
            for variant in (VARIANT, VENDOR_VARIANT):
                roots = self.scratch / f"{variant}.roots"
                hits = self.scratch / f"{variant}.tickets"
                reply = worker.request(["--native-job", str(control), *shape, "--variant", variant,
                                        "--roots-out", str(roots), "--out", str(hits), "--defer-filter",
                                        "--rows-per-pass", "4096", "--cols-per-pass", "16384", "--k-per-pass", "0"])
                if variant == VENDOR_VARIANT:
                    self.assertEqual(reply.get("cublaslt_calls"), 1, reply.get("cublaslt_plan"))
                    self.assertEqual(reply.get("cublaslt_fallback_calls"), 0)
                audited = self.invoke("native_job.py", "inspect", control, roots, hits)
                self.assertEqual(json.loads(audited.stdout)["checked_digest_records"], reply["threshold_hits"])
                outputs.append((roots.read_bytes(), hits.read_bytes()))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
