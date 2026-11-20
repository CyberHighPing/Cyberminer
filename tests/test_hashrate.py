import io
import json
import os
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import fp8miner
import gateway_miner
from hashrate import RateDisplay


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class HashrateTest(unittest.TestCase):
    def test_bench_excludes_warmup_and_preserves_stdout(self):
        for quiet in (False, True):
            with self.subTest(quiet=quiet), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "pearl-fp8-scan").touch()
                clock = Clock()
                durations = iter((40, 2, 6))

                def scan(*_):
                    clock.advance(next(durations))
                    return None, 0, None

                out, err = io.StringIO(), io.StringIO()
                flags = ["bench", "--jobs", "2"] + (["--no-status"] if quiet else [])
                with patch.object(fp8miner, "BUILD", root), patch("time.perf_counter", clock), \
                        patch.object(fp8miner, "scanner_session", return_value=nullcontext()), \
                        patch.object(fp8miner, "scan_plan", side_effect=scan), redirect_stdout(out), redirect_stderr(err):
                    self.assertEqual(fp8miner.main(flags), 0)
                result = json.loads(out.getvalue())
                self.assertEqual(result["wall_seconds"], 8)
                self.assertEqual(result["threshold_hits"], 0)
                work = result["dense_macs"] // 2
                self.assertEqual(result["full_scan_th_s"], 2 * work / 8 / 1e12)
                if quiet:
                    self.assertEqual(err.getvalue(), "")
                else:
                    lines = err.getvalue().splitlines()
                    self.assertIn("last scan -- TH/s", lines[0])
                    self.assertIn("warm-up", lines[0])
                    self.assertIn(f"last scan {work/6/1e12:.2f} TH/s", lines[-1])
                    self.assertIn(f"session avg {result['full_scan_th_s']:.2f} TH/s", lines[-1])
                    self.assertIn("scans 2 | elapsed 8.0s", lines[-1])

    def test_mine_counts_validated_scans_once_and_includes_wait_and_proof_time(self):
        import host_miner
        import native_job
        from native_scan import derive, synthetic_headers

        header, ancestors = synthetic_headers(0x18010000)
        job = gateway_miner.Job(header, 0x10000 << (8*(0x18-3)), tuple(ancestors))
        for scenario in ("no hit", "proof failed", "scan failed"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                args = fp8miner.arguments(["mine", "--socket", "example.sock", "--work-dir", temp, "--attempts", "1"])
                clock, err, out = Clock(), io.StringIO(), io.StringIO()

                def fetch(*_):
                    clock.advance(3)
                    return job

                def scan(plan, *_):
                    clock.advance(2)
                    if scenario == "scan failed":
                        raise ValueError("synthetic failed scan")
                    roots = derive(plan, bytes(32), bytes(32))
                    return roots, int(scenario == "proof failed"), dict(
                        streamed_b=True, passes=256, scan_wall_s=2, scan_equivalent_th_s=0)

                original_proof = native_job.proof_for_output

                def proof(*positional, **options):
                    if scenario == "proof failed":
                        self.assertEqual(args.rate_display.phase_name, "opening / verifying")
                        clock.advance(5)
                        raise ValueError("synthetic failed proof")
                    return original_proof(*positional, **options)

                with patch("time.perf_counter", clock), patch.object(gateway_miner.Job, "fetch", side_effect=fetch), \
                        patch.object(native_job, "scan_plan", side_effect=scan), \
                        patch.object(native_job, "proof_for_output", side_effect=proof), \
                        redirect_stdout(out), redirect_stderr(err):
                    if scenario == "no hit":
                        with RateDisplay() as args.rate_display:
                            self.assertEqual(gateway_miner.run_jobs(args, None, host_miner, None), 2)
                        self.assertEqual(json.loads(out.getvalue())["status"], "no_difficulty_hit")
                    else:
                        with self.assertRaises(ValueError), RateDisplay() as args.rate_display:
                            gateway_miner.run_jobs(args, None, host_miner, None)
                last = err.getvalue().splitlines()[-1]
                if scenario == "scan failed":
                    self.assertIn("last scan -- TH/s | session avg -- TH/s | scans 0", last)
                else:
                    work = args.m*args.n*args.k
                    self.assertIn(f"last scan {work/2/1e12:.2f} TH/s", last)
                    self.assertIn(f"session avg {work/clock.now/1e12:.2f} TH/s | scans 1", last)
                self.assertNotIn("example.sock", err.getvalue())
                self.assertNotIn("synthetic failed", err.getvalue())

    def test_status_output_failure_does_not_stop_work(self):
        closed = io.StringIO()
        closed.close()
        clock = Clock()
        with RateDisplay(stream=closed, clock=clock) as display:
            clock.advance(1)
            display.completed(10**12, 1, clock())
        self.assertEqual(display.scans, 1)

    def test_missing_stderr_keeps_stdout_clean(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(sys, "stderr", None):
            with RateDisplay() as display:
                display.completed(10**12, 1, display.clock())
            print('{"ok":true}')
        self.assertEqual(output.getvalue(), '{"ok":true}\n')

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux pipe semantics")
    def test_full_stderr_pipe_cannot_block_mining_or_exit(self):
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(write_fd, False)
            with self.assertRaises(BlockingIOError):
                while True:
                    os.write(write_fd, b"x"*4096)
            os.set_blocking(write_fd, True)
            script = ("from hashrate import RateDisplay\n"
                      "with RateDisplay() as display:\n"
                      "    for i in range(4096): display.completed(10**12, 1, display.clock())\n"
                      "print(display.scans)\n")
            p = subprocess.run([sys.executable, "-B", "-c", script],
                               cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
                               stderr=write_fd, text=True, timeout=4, check=True)
            self.assertEqual(p.stdout.strip(), "4096")
            self.assertTrue(os.get_blocking(write_fd))
        finally:
            os.close(write_fd)
            os.close(read_fd)


if __name__ == "__main__":
    unittest.main()
