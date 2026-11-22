"""Sequential local PEARL-V4-SCAN/1 process client. Uses binary pipes and no shell."""

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import struct
import subprocess
import threading
import time


_MAGIC = b"PEARL-V4-SCAN/1\n"
_WORD = struct.Struct("<I")
_MAX_REPLY = 65536
_STDERR_TAIL = 8192


class WorkerError(RuntimeError):
    pass


def pack_arguments(args):
    words = list(args)
    if not 1 <= len(words) <= 96:
        raise ValueError("worker needs 1..96 scanner arguments")
    frame = bytearray(_WORD.pack(len(words)))
    size = 0
    for item in words:
        text = os.fspath(item)
        if not isinstance(text, str):
            raise ValueError("worker arguments must be text")
        value = text.encode("utf-8")
        size += len(value)
        if not 1 <= len(value) <= 16384 or size > 65536 or b"\x00" in value:
            raise ValueError("worker argument is empty, contains NUL, or exceeds the C++ frame limits")
        frame += _WORD.pack(len(value))
        frame += value
    return bytes(frame)


def _timeout(seconds, name):
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return seconds


class CudaWorker:
    """One ordinary child process. Calls are serialized; a failed call ends the session."""
    def __init__(self, scanner, *, startup_timeout=15.0, request_timeout=None, shutdown_timeout=2.0, env=None):
        # An explicit argv prefix also permits interpreter-backed local test scanners.
        cmd = list(scanner) if isinstance(scanner,(list,tuple)) else [scanner]
        if not cmd:
            raise ValueError("scanner executable is required")
        cmd = [os.fspath(x) for x in cmd]
        cmd[0] = str(Path(cmd[0]).resolve())
        if os.name == "nt" and Path(cmd[0]).suffix.lower() in (".bat",".cmd"):
            raise ValueError("persistent mode requires a direct executable")
        self._command = cmd + ["--worker"]
        self._env = env
        self._startup_timeout = _timeout(startup_timeout, "startup timeout")
        self._request_timeout = _timeout(request_timeout, "worker timeout")
        self._shutdown_timeout = _timeout(shutdown_timeout, "shutdown timeout")
        self._serial = threading.Lock()
        self._state = threading.Lock()
        self._tail_lock = threading.Lock()
        self._in = Queue()
        self._out = Queue()
        self._tail = bytearray()
        self._threads = []
        self._proc = None
        self._closed = False
        self._busy = False

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(graceful=kind is None)

    @property
    def pid(self):
        with self._state:
            return self._proc.pid if self._proc is not None else None

    @property
    def closed(self):
        with self._state:
            return self._closed

    def _drain_errors(self, pipe):
        try:
            while data := pipe.read1(4096):
                with self._tail_lock:
                    self._tail.extend(data)
                    del self._tail[:-_STDERR_TAIL]
        except (OSError,ValueError):
            pass

    def _exchange(self, proc):
        def line():
            data=proc.stdout.readline(_MAX_REPLY+1)
            if not data:
                raise WorkerError("scanner closed stdout before acknowledging the job")
            if len(data)>_MAX_REPLY or not data.endswith(b"\n"):
                raise WorkerError("scanner reply is incomplete or too long")
            return data
        try:
            self._out.put(line())
            while True:
                frame=self._in.get()
                if frame is None:
                    proc.stdin.write(_WORD.pack(0))
                    proc.stdin.flush()
                    proc.stdin.close()
                    return
                proc.stdin.write(frame)
                proc.stdin.flush()
                self._out.put(line())
        except Exception as exc:
            self._out.put(exc)

    def _start_locked(self):
        if self._closed:
            raise WorkerError("scanner session is closed")
        if self._proc is not None:
            return False
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen(self._command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,shell=False,close_fds=True,creationflags=flags,env=self._env)
        self._proc = proc
        for fn,arg in ((self._drain_errors,proc.stderr),(self._exchange,proc)):
            thread=threading.Thread(target=fn,args=(arg,),daemon=True)
            thread.start()
            self._threads.append(thread)
        return True

    def _reply(self, timeout, stage):
        deadline = None if timeout is None else time.monotonic()+timeout
        while True:
            left = 0.25 if deadline is None else min(0.25, deadline-time.monotonic())
            if left <= 0:
                raise WorkerError(f"scanner {stage} timed out")
            try:
                result=self._out.get(timeout=left)
            except Empty:
                if self.closed:
                    raise WorkerError("scanner session closed during a request")
                continue
            if isinstance(result,Exception):
                raise result
            return result

    def request(self,args):
        frame=pack_arguments(args)
        with self._serial:
            try:
                with self._state:
                    started=self._start_locked()
                    self._busy=True
                if started and self._reply(self._startup_timeout,"handshake") != _MAGIC:
                    raise WorkerError("incorrect scanner protocol greeting")
                self._in.put(frame)
                line=self._reply(self._request_timeout,"response")
                result=json.loads(line.decode("utf-8"),parse_constant=_invalid_json_constant)
                if not isinstance(result,dict):
                    raise WorkerError("scanner reply must be one JSON object")
                return result
            except BaseException as exc:
                self.close(graceful=False)
                if not isinstance(exc,Exception):
                    raise
                with self._tail_lock:
                    reason=self._tail.decode("utf-8",errors="replace").strip()
                raise WorkerError(f"{exc}" + (f" | scanner stderr: {reason}" if reason else "")) from exc
            finally:
                with self._state:
                    self._busy=False

    def close(self, *, graceful=True):
        with self._state:
            if self._closed:
                return
            self._closed=True
            proc=self._proc
            normal=graceful and not self._busy
        if proc is None:
            return
        # The I/O thread may be blocked on either pipe. Killing the child
        # before closing the pipes wakes both without waiting on Python I/O locks.
        self._in.put(None)
        exit_code=proc.poll()
        try:
            if normal and exit_code is None:
                try:
                    exit_code=proc.wait(timeout=self._shutdown_timeout)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # A keyboard interrupt during graceful shutdown must still reap the
            # scanner. No child or job that did not originate here is targeted.
            forced=exit_code is None
            if forced:
                try:
                    proc.terminate()
                except OSError:
                    if proc.poll() is None:
                        raise
                try:
                    proc.wait(timeout=self._shutdown_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            for thread in self._threads:
                thread.join(timeout=self._shutdown_timeout)
            for pipe in (proc.stdin,proc.stdout,proc.stderr):
                try:
                    pipe.close()
                except (OSError,ValueError):
                    pass
        if normal and (forced or exit_code != 0):
            with self._tail_lock:
                detail=self._tail.decode("utf-8",errors="replace").strip()
            reason="timed out shutting down" if forced else f"exited with status {exit_code}"
            raise WorkerError(f"scanner {reason}" + (f": {detail}" if detail else ""))


def _invalid_json_constant(text):
    raise ValueError(f"invalid JSON numeric token {text}")


def capacity_arg(text):
    try:
        value=int(text)
    except (TypeError,ValueError) as exc:
        raise argparse.ArgumentTypeError("hit capacity must be an integer in 1..65536") from exc
    if not 1 <= value <= 65536:
        raise argparse.ArgumentTypeError("hit capacity must be in 1..65536")
    return value


def row_pad_arg(text):
    try:
        value=int(text)
    except (TypeError,ValueError) as exc:
        raise argparse.ArgumentTypeError("row padding must be a multiple of 16 in 0..4096") from exc
    if not 0 <= value <= 4096 or value%16:
        raise argparse.ArgumentTypeError("row padding must be a multiple of 16 in 0..4096")
    return value


def add_worker_options(parser):
    parser.add_argument("--persistent",action="store_true",help="reuse one local CUDA scanner process across jobs")
    parser.add_argument("--worker-timeout",type=float,help="optional response timeout in seconds for each persistent scan")
    parser.add_argument("--variant",help="CUDA scanner kernel, including sm120-fused-stream with 16x32 streamed tickets")
    parser.add_argument("--resident",action="store_true",help="ask the scanner to keep operand panels resident during a scan")
    parser.add_argument("--stream-b",action="store_true",help="native dense 16x32 tickets: keep A and regenerate columns of B (4096x16384 windows)")
    parser.add_argument("--fuse-fold",action="store_true",help="request fused GEMM/fold on compatible geometry")
    parser.add_argument("--defer-filter",action="store_true",help="request deferred ticket filtering on compatible scans")
    parser.add_argument("--row-pad",type=row_pad_arg,default=0,help="extra bytes per GPU operand row, in units of 16")
    parser.add_argument("--k-per-pass",type=int,help="consecutive K elements per pass, a multiple of 128, or 0 for one pass")


def add_geometry_options(parser):
    for name, default in (("m", 32), ("n", 128), ("k", 4096)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--tile", default="4x128")


def add_scan_options(parser):
    parser.add_argument("--rows-per-pass", type=int)
    parser.add_argument("--cols-per-pass", type=int)
    parser.add_argument("--sparse", action="store_true", help="filter every ticket on the CUDA device")
    parser.add_argument("--hit-capacity", type=capacity_arg, default=256)
    add_worker_options(parser)


def add_native_options(parser):
    parser.add_argument("--native-dense", action="store_true", help="generate fresh dense operands and sparse results on the GPU")
    parser.add_argument("--native-full-reference", action="store_true", help="rebuild entire CPU FP8 planes when a native job returns candidates")
    parser.add_argument("--gpu-prover", type=Path, help="external dense V4 Merkle proof constructor for --native-dense")


STREAM_ROWS = 4096
STREAM_COLS = 16384
FUSED_STREAM_VARIANT = "sm120-fused-stream"
STREAM_VARIANTS = ("cublaslt-verified-pipeline", "cutlass120-coop-swizzle-pipeline", FUSED_STREAM_VARIANT)

# The scanner packs B once and streams A using the selected board profile.
# Only the scanner child receives these choices. Device clocks are set separately.
FAST_GEOMETRY = (1044480, 540672, 13312, 16, 32)
TESTNET2_GEOMETRY = (131072, 131072, 16384, 16, 32)
GEOMETRIES = {"throughput": FAST_GEOMETRY, "testnet2": TESTNET2_GEOMETRY}
FAST_STRIP_ROWS = 1536
RTX_5060_STRIP_ROWS = 1280
FAST_SETTINGS = {
    "PEARL_B_COMMIT": "panel", "PEARL_RECT_FOLD": "vector", "PEARL_SCAN_BUFFER_POOL": "reuse",
    "PEARL_TAIL_LAYOUT": "overlap256", "PEARL_PANEL_GROUP": "3", "PEARL_PREP_INPUTS": "leafvec",
    "PEARL_RESIDENT_AXIS": "right", "PEARL_RESIDENT_BYTES": "packed", "PEARL_STRIP_CACHE": "persist",
    "PEARL_B_REUSE_GRID": "columns", "PEARL_STREAM_A_COMMIT": "leaves",
    "PEARL_R10_FILTER_BATCH": "16", "PEARL_R10_BATCH_WAIT": "early85",
    "PEARL_ROW_STRIP": str(FAST_STRIP_ROWS), "PEARL_NATIVE_FINITE_FOLD": "paired",
}

CHECKPOINT_SETTING = "PEARL_PROOF_CHECKPOINTS"


def proof_environment(args):
    """Carry the public command's chosen proof mode to its children."""
    mode = getattr(args, "proof_checkpoints", None)
    if mode is None:
        return None
    return {**os.environ, CHECKPOINT_SETTING: mode}


def scanner_environment(args):
    if getattr(args, "fast_profile", False):
        rows = getattr(args, "strip_rows", FAST_STRIP_ROWS)
        if type(rows) is not int or not 256 <= rows <= 8192 or rows % 256:
            raise ValueError("row strip height must be a multiple of 256 in the range 256..8192")
        return {**(proof_environment(args) or os.environ), **FAST_SETTINGS, "PEARL_ROW_STRIP": str(rows)}
    return None


def streamed_variant(args, dims):
    m,n,k,h,w = dims
    if getattr(args, "fast_profile", False):
        if ((k,h,w) not in (FAST_GEOMETRY[2:], TESTNET2_GEOMETRY[2:])
                or m <= 0 or n <= 0 or m % STREAM_ROWS or n % STREAM_COLS):
            raise ValueError("the optimized presets require full 4096x16384 windows, K=13312 or 16384, and 16x32 tickets")
        return FUSED_STREAM_VARIANT
    tile_supported = (h,w) == (16,32) or ((h,w) == (16,16) and k == 16384)
    if (m < STREAM_ROWS or n < STREAM_COLS or m % STREAM_ROWS or n % STREAM_COLS
            or k not in (16384,32768) or not tile_supported):
        raise ValueError("--stream-b requires full 4096x16384 windows, K=16384 or 32768 with --tile 16x32, or K=16384 with --tile 16x16")
    if (getattr(args,"rows_per_pass",None) not in (None,STREAM_ROWS)
            or getattr(args,"cols_per_pass",None) not in (None,STREAM_COLS)
            or getattr(args,"k_per_pass",None) not in (None,0,k)
            or getattr(args,"row_pad",0) or getattr(args,"fuse_fold",False)):
        raise ValueError("--stream-b uses 4096x16384 windows, the entire K and unpadded rows; folding is selected by --variant")
    variant = getattr(args,"variant",None)
    if variant in (None,"auto"):
        return STREAM_VARIANTS[0]
    if variant == FUSED_STREAM_VARIANT and (h,w)!=(16,32):
        raise ValueError("sm120-fused-stream requires 16x32 tickets")
    if variant not in STREAM_VARIANTS:
        raise ValueError("--stream-b supports " + " or ".join(STREAM_VARIANTS))
    return variant


def validate_stream_request(args):
    if not getattr(args,"stream_b",False):
        if getattr(args,"variant",None)==FUSED_STREAM_VARIANT:
            raise ValueError("sm120-fused-stream requires --stream-b")
        return
    if not getattr(args,"native_dense",False) or not getattr(args,"scanner",None) or getattr(args,"cpu_reference",False):
        raise ValueError("--stream-b requires --native-dense and a CUDA --scanner")
    h,w = map(int,args.tile.lower().split("x"))
    streamed_variant(args,(args.m,args.n,args.k,h,w))


def scanner_session(args):
    validate_stream_request(args)
    persistent=getattr(args,"persistent",False)
    timeout=getattr(args,"worker_timeout",None)
    k_per_pass=getattr(args,"k_per_pass",None)
    if k_per_pass is not None and (not 0<=k_per_pass<=65536 or k_per_pass%128):
        raise ValueError("--k-per-pass must be 0 or a multiple of 128 up to 65536")
    if getattr(args,"cpu_reference",False) and (persistent or getattr(args,"variant",None) is not None
            or k_per_pass is not None or any(getattr(args,n,False) for n in ("sparse","resident","fuse_fold","defer_filter","row_pad"))):
        raise ValueError("CUDA tuning and persistent mode require --scanner")
    if getattr(args,"row_pad",0) and (getattr(args,"variant",None) or "").startswith("cutlass"):
        raise ValueError("CUTLASS variants require --row-pad 0")
    if timeout is not None and not persistent:
        raise ValueError("--worker-timeout requires --persistent")
    if persistent:
        if not getattr(args,"scanner",None) or getattr(args,"cpu_reference",False):
            raise ValueError("--persistent requires a CUDA scanner")
        return CudaWorker(args.scanner,request_timeout=timeout,env=scanner_environment(args))
    return nullcontext(None)


def scan_options(args,worker):
    if getattr(args,"stream_b",False):
        raise ValueError("--stream-b requires native dense inputs")
    options={}
    if getattr(args,"k_per_pass",None) is not None:
        options["k_per_pass"]=args.k_per_pass
    if getattr(args,"sparse",False):
        options.update(sparse=True,hit_capacity=getattr(args,"hit_capacity",256))
    if getattr(args,"variant",None) is not None:
        options["variant"]=args.variant
    if getattr(args,"resident",False):
        options["resident"]=True
    if getattr(args,"fuse_fold",False):
        options["fuse_fold"]=True
    if getattr(args,"defer_filter",False):
        options["defer_filter"]=True
    if getattr(args,"row_pad",0):
        options["row_pad"]=args.row_pad
    if worker is not None:
        options["worker"]=worker
    return options


@contextmanager
def abort_on_error(worker):
    try:
        yield
    except BaseException:
        if worker is not None:
            worker.close(graceful=False)
        raise


def check_file_scan(report,meta,sparse,resident,variant,row_pad):
    if report.get("streamed_b",False) is not False:
        raise WorkerError("file scanner acknowledged an unrequested streamed B layout")
    expected={"mode":"file_scan","m":meta["m"],"n":meta["n"],"k":meta["k"],
              "tile_h":meta["h"],"tile_w":meta["w"],"sparse_filter":bool(sparse),
              "resident_operands":bool(resident),
              "tiles":(meta["m"]//meta["h"])*(meta["n"]//meta["w"])}
    if variant and variant != "auto":
        expected["variant"]=variant
    if row_pad or "row_pitch_bytes" in report:
        expected["row_pitch_bytes"]=meta["k"]+row_pad
    for name,value in expected.items():
        if type(report.get(name)) is not type(value) or report[name] != value:
            raise WorkerError(f"scanner acknowledged the wrong {name}")
    if sparse:
        hits=report.get("threshold_hits")
        if type(hits) is not int or not 0 <= hits <= expected["tiles"]:
            raise WorkerError("scanner acknowledged an invalid hit count")
