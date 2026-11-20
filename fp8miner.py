"""Run the RTX 5060 Ti 8 GB FP8 scanner, offline or with a local Pearl V4 gateway."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time

from cuda_worker import FAST_GEOMETRY, FUSED_STREAM_VARIANT, GEOMETRIES, STREAM_COLS, STREAM_ROWS, scanner_session
from hashrate import RateDisplay
from native_scan import Plan, scan_plan, synthetic_headers

BUILD = Path(__file__).resolve().parent / "build"


class PublicArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse diagnostics may contain the operator's original input.
        self.exit(2, "error: invalid arguments. Use --help for usage.\n")


def arguments(argv=None):
    parser = PublicArgumentParser(prog="fp8miner.py", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("bench", help="measure complete dense synthetic scans on one visible GPU")
    bench.add_argument("--jobs", type=int, default=2, help="number of measured jobs after one warm-up")
    bench.add_argument("--quick", action="store_true", help="use one small complete window for a smoke test")
    mine = commands.add_parser("mine", help="mine and submit verified proofs via a local V4 gateway")
    mine.add_argument("--socket", required=True, type=Path, help="Unix socket of your configured gateway")
    mine.add_argument("--work-dir", type=Path, default=Path("runs"), help="local directory for verified proofs and receipts")
    mine.add_argument("--native-dir", type=Path, help="directory containing the official pearl_mining extension")
    mine.add_argument("--attempts", type=int, default=0, help="maximum jobs, or zero to continue")
    mine.add_argument("--ancestor-depth", type=int, choices=range(1,5), default=1)
    for command, default in ((bench, "throughput"), (mine, "testnet2")):
        command.add_argument("--no-status", action="store_true", help="hide the live rate lines on stderr")
        command.add_argument("--preset", choices=GEOMETRIES, default=default,
                             help=f"testnet2 selects 131072x131072x16384 for local solo work; "
                                  f"throughput selects 1044480x540672x13312 for large scans (default: {default})")
    parser.set_defaults(fast_profile=True, scanner=BUILD/"pearl-fp8-scan",
                        gpu_prover=BUILD/"pearl-fp8-open", persistent=True, native_dense=True, stream_b=True,
                        defer_filter=True, variant=FUSED_STREAM_VARIANT, k_per_pass=0, hit_capacity=256,
                        rows_per_pass=STREAM_ROWS, cols_per_pass=STREAM_COLS, cpu_reference=False,
                        native_full_reference=False, rpc_timeout=10.0, poll_seconds=2.0)
    args = parser.parse_args(argv)
    args.m, args.n, args.k, h, w = GEOMETRIES[args.preset]
    args.tile = f"{h}x{w}"
    if args.command=="bench" and args.jobs<1:
        parser.error("--jobs must be positive")
    if args.command=="mine" and args.attempts<0:
        parser.error("--attempts must be nonnegative")
    return args


def benchmark(args):
    preset = getattr(args, "preset", "throughput")
    measure = GEOMETRIES[preset]
    warm = (min(339968, measure[0]), *measure[1:])
    if args.quick:
        measure = warm = (STREAM_ROWS, STREAM_COLS, *measure[2:])
    # A synthetic compact target only controls the returned hit count.
    header, chain = synthetic_headers(0x18010000)
    work_per_job = measure[0]*measure[1]*measure[2]
    work = work_per_job*args.jobs
    display = getattr(args, "rate_display", None)
    with scanner_session(args) as worker:
        def scan(dims):
            plan = Plan(dims, secrets.token_bytes(32), header, tuple(chain))
            with tempfile.TemporaryDirectory(prefix="fp8-scan-") as directory:
                _, count, _ = scan_plan(plan, args, Path(directory), worker)
            return count
        scan(warm)
        start = time.perf_counter()
        if display is not None:
            display.after_warmup(start)
        hits = 0
        for _ in range(args.jobs):
            if display is not None:
                display.phase("scanning")
            scan_start = time.perf_counter()
            hits += scan(measure)
            end = time.perf_counter()
            if display is not None:
                display.completed(work_per_job, end-scan_start, end)
        elapsed = end-start
        if display is not None:
            display.end_benchmark()
    return dict(mode="offline_full_scan", preset=preset, geometry=measure, jobs=args.jobs, dense_macs=work,
                wall_seconds=elapsed, full_scan_th_s=work/elapsed/1e12, threshold_hits=hits)


def main(argv=None):
    args = arguments(argv)
    if not args.scanner.is_file():
        raise FileNotFoundError("build/pearl-fp8-scan is missing. Run bash build.sh first.")
    if args.command=="mine" and not args.gpu_prover.is_file():
        raise FileNotFoundError("build/pearl-fp8-open is missing. Run bash build.sh first.")
    status = nullcontext(None) if args.no_status else RateDisplay(benchmark=args.command=="bench")
    with status as args.rate_display:
        if args.command=="bench":
            print(json.dumps(benchmark(args)), flush=True)
            return 0
        from gateway_miner import connect
        return connect(args)


def cli(argv=None):
    try:
        return main(argv)
    except KeyboardInterrupt:
        return 130
    except ImportError:
        message = "a Python dependency or the official V4 verifier is missing"
    except FileNotFoundError:
        message = "a build artifact or the configured gateway socket is missing"
    except OSError:
        message = "could not access the local executable, working directory or gateway socket"
    except subprocess.SubprocessError:
        message = "the local GPU scanner or proof constructor failed"
    except (KeyError, TypeError, ValueError, RuntimeError):
        message = "the GPU job, proof or gateway reply failed validation"
    except Exception:
        message = "the local task could not be completed"
    # Exception messages can contain socket paths, ticket indices or gateway text.
    print("error: " + message, file=sys.stderr)
    return 1


if __name__=="__main__":
    raise SystemExit(cli())
