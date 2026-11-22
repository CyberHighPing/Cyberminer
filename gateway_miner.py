"""Mine dense FP8 jobs from a local Pearl V4 gateway Unix socket.

The gateway queues submitPlainProof asynchronously. Its miner RPC does not
expose the subsequent node acceptance. A lost receipt is never retried.
"""

import argparse
import base64
import hashlib
import json
import math
import socket
import struct
import sys
import tempfile
import time
from contextlib import closing, ExitStack
from dataclasses import dataclass
from pathlib import Path
from cuda_worker import add_worker_options, capacity_arg, scanner_session, scan_options, validate_stream_request
from cuda_worker import add_geometry_options, add_native_options, add_scan_options


FRAME_LIMIT = 4 * 1024 * 1024  # d797c307 MinerRpcServer.READER_BUFFER_LIMIT


class RpcError(RuntimeError):
    def __init__(self, error):
        self.code = error.get("code")
        super().__init__(f"JSON-RPC {self.code}: {error.get('message', '')}")


class FrameTooLarge(ValueError):
    pass


class Gateway:
    def __init__(self, stream, reader):
        self.stream = stream
        self.reader = reader
        self.sequence = 0

    def call(self, method, params):
        self.sequence += 1
        envelope = {"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params}
        line = json.dumps(envelope, separators=(",", ":")).encode("ascii") + b"\n"
        if len(line) > FRAME_LIMIT:
            raise FrameTooLarge("request exceeds the official gateway's 4 MiB line limit")
        self.stream.sendall(line)
        reply = self.reader.readline(FRAME_LIMIT + 1)
        if len(reply) > FRAME_LIMIT:
            raise ValueError("oversized gateway reply")
        if not reply.endswith(b"\n"):
            raise ConnectionError("incomplete gateway reply")
        message = json.loads(reply)
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise ValueError("gateway returned an invalid JSON-RPC envelope")
        if type(message.get("id")) is not int or message["id"] != self.sequence:
            raise ValueError("gateway reply has the wrong request ID")
        if ("result" in message) == ("error" in message):
            raise ValueError("gateway must return exactly one result or error")
        if "error" in message:
            error = message["error"]
            if not isinstance(error, dict) or type(error.get("code")) is not int or not isinstance(error.get("message"), str):
                raise ValueError("malformed JSON-RPC error")
            raise RpcError(error)
        return message["result"]


class GatewaySession:
    """Reconnect once for a read that lost its response. A submission is only sent once."""
    def __init__(self, path, timeout):
        self.path = path
        self.timeout = timeout
        self.resources = ExitStack()
        self.rpc = None

    def open(self):
        with ExitStack() as pending:
            stream = pending.enter_context(closing(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)))
            stream.settimeout(self.timeout)
            stream.connect(str(self.path))
            reader = pending.enter_context(stream.makefile("rb"))
            self.rpc = Gateway(stream, reader)
            self.resources = pending.pop_all()

    def close(self):
        self.rpc = None
        self.resources.close()

    def call(self, method, params):
        attempts = 2 if method == "getMiningInfo" else 1
        for attempt in range(attempts):
            try:
                if self.rpc is None:
                    self.open()
                return self.rpc.call(method, params)
            except (ConnectionError, TimeoutError, FileNotFoundError):
                self.close()
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(0.2, self.timeout))


def decode_header(value, length):
    if not isinstance(value, str):
        raise ValueError("gateway header must be a base64 string")
    raw = base64.b64decode(value, validate=True)
    if len(raw) != length:
        raise ValueError(f"expected a {length}-byte header, received {len(raw)} bytes")
    return raw


@dataclass(frozen=True)
class Job:
    header: bytes
    target: int
    ancestors: tuple

    @classmethod
    def fetch(cls, rpc, miner):
        wire = rpc.call("getMiningInfo", {})
        if not isinstance(wire, dict) or type(wire.get("cert_version")) is not int or wire["cert_version"] != 4:
            raise ValueError("the gateway must explicitly advertise certificate version 4")
        if set(wire) != {"incomplete_header_bytes", "target", "cert_version", "ancestor_headers"}:
            raise ValueError("unexpected V4 mining job fields")
        header = decode_header(wire.get("incomplete_header_bytes"), 76)
        window = wire.get("ancestor_headers")
        if not isinstance(window, list) or not 1 <= len(window) <= 4:
            raise ValueError("V4 requires 1 to 4 complete ancestor headers, parent first")
        ancestors = tuple(decode_header(v, 108) for v in window)
        miner.check_headers(header, ancestors)
        target = wire.get("target")
        bits = struct.unpack_from("<I", header, 72)[0]
        exponent, mantissa = bits >> 24, bits & 0x7fffff
        if bits & 0x800000 or not mantissa or exponent < 3:
            raise ValueError("invalid positive compact target for this gateway version")
        implied = mantissa << (8 * (exponent - 3))
        if type(target) is not int or not 0 < target < (1 << 256) or target != implied:
            raise ValueError("gateway target and proposed header nBits disagree")
        return cls(header, target, ancestors)

    def to_wire(self):
        return {"incomplete_header_bytes": base64.b64encode(self.header).decode("ascii"), "target": self.target,
                "cert_version": 4, "ancestor_headers": [base64.b64encode(a).decode("ascii") for a in self.ancestors]}

    def for_host(self, depth):
        if depth > len(self.ancestors):
            raise ValueError(f"requested ancestor depth {depth}, but the job supplies {len(self.ancestors)}")
        return {"proposed_header_hex": self.header.hex(), "cert_version": 4,
                "ancestor_headers_hex": [a.hex() for a in self.ancestors[:depth]]}


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def mine_once(args, rpc, miner, job, attempt, worker=None):
    validate_stream_request(args)
    params = job.for_host(args.ancestor_depth)
    header_id = hashlib.sha256(job.header).hexdigest()
    with tempfile.TemporaryDirectory(dir=args.work_dir, prefix=".fp8-attempt-") as tmp:
        scratch = Path(tmp)
        header_file = scratch / "header.json"
        write_json(header_file, params)
        candidate = scratch / "candidate"
        # A blank seed asks host_miner to draw new random operands on every attempt.
        prep = argparse.Namespace(job=header_file, seed=None, m=args.m, n=args.n, k=args.k, tile=args.tile, progress=False)
        if getattr(args,"native_dense",False):
            outcome = miner.native_attempt(prep, args, candidate, worker)
        else:
            miner.prepare(prep, candidate)
            miner.scan(candidate, args.scanner, args.cpu_reference, args.rows_per_pass, args.cols_per_pass,
                       **scan_options(args,worker))
            result_file = candidate / ("hits.phits" if getattr(args,"sparse",False) else "folds.bin")
            outcome = miner.finish(candidate, result_file, args.native_dir)
        summary = {"attempt": attempt, "header_sha256": header_id, "ancestor_depth": args.ancestor_depth,
                   "tickets": outcome["tickets"], "hits_checked": outcome["hits_checked"]}
        if getattr(args,"native_dense",False):
            summary.update(native_dense=True,threshold_hits=outcome["threshold_hits"],
                           cpu_recomputed_roots=outcome["cpu_recomputed_roots"])
            if getattr(args,"stream_b",False):
                for key in ("streamed_b","dense_macs","covered_windows","scan_wall_s","scan_equivalent_th_s"):
                    summary[key]=outcome[key]
                if "streamed_a" in outcome:
                    summary["streamed_a"]=outcome["streamed_a"]
            if "proof_inputs" in outcome:
                summary["proof_inputs"]=outcome["proof_inputs"]
            if "cpu_verified_source_openings" in outcome:
                summary["cpu_verified_source_openings"]=outcome["cpu_verified_source_openings"]
        if outcome["status"] in ("no_admissible_hit", "no_difficulty_hit"):
            return {**summary, "status": outcome["status"]}
        if outcome["status"] != "plain_verified":
            raise ValueError(f"unsupported verifier result: {outcome['status']}")
        raw = (candidate / outcome["plain"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != outcome["proof_bytes"] or digest != outcome["proof_sha256"]:
            raise ValueError("serialized proof differs from the bytes verified by host_miner")

        saved = Path(tempfile.mkdtemp(dir=args.work_dir, prefix=f"proof-{digest[:16]}-"))
        (saved / "plain.bin").write_bytes(raw)
        write_json(saved / "mining_job.json", job.to_wire())
        summary.update(proof_sha256=digest, proof_dir=str(saved), node_acceptance="unconfirmed")
        receipt = saved / "receipt.json"
        write_json(receipt, {**summary, "status": "verified_locally"})
        display = getattr(args, "rate_display", None)
        if display is not None:
            display.phase("refreshing work")
        try:
            current = Job.fetch(rpc, miner)
        except (ValueError, OSError, RpcError) as error:
            status = "paused_before_submit" if isinstance(error, RpcError) and error.code == -32001 else "refresh_failed"
            result = {**summary, "status": status, "error": str(error)}
            write_json(receipt, result)
            return result
        if current != job:
            result = {**summary, "status": "stale_before_submit"}
        else:
            payload = {"plain_proof": base64.b64encode(raw).decode("ascii"), "mining_job": job.to_wire()}
            write_json(receipt, {**summary, "status": "submission_in_progress"})
            if display is not None:
                display.phase("submitting")
            try:
                response = rpc.call("submitPlainProof", payload)
                if response != "submitted":
                    raise ValueError("unrecognized submitPlainProof receipt")
                result = {**summary, "status": "gateway_queued", "gateway_reply": response}
            except RpcError as error:
                result = {**summary, "status": "gateway_rpc_error", "error": str(error)}
            except FrameTooLarge as error:
                result = {**summary, "status": "proof_exceeds_gateway_limit", "error": str(error)}
            except (ValueError, OSError) as error:
                result = {**summary, "status": "submission_unconfirmed", "error": str(error)}
        write_json(receipt, result)
        return result


def run(args, rpc, miner):
    with scanner_session(args) as worker:
        return run_jobs(args,rpc,miner,worker)

def progress(result, fast_profile):
    if not fast_profile:
        return result
    # The user's work directory holds the full submission receipt.
    fields=("status","attempt","tickets","threshold_hits","hits_checked","dense_macs","scan_wall_s","scan_equivalent_th_s")
    return {key:result[key] for key in fields if key in result}


def run_jobs(args,rpc,miner,worker):
    attempt = 0
    queued_header = None
    paused = False
    display = getattr(args, "rate_display", None)
    while args.attempts == 0 or attempt < args.attempts:
        if display is not None:
            display.phase("waiting for work")
        try:
            job = Job.fetch(rpc, miner)
        except RpcError as error:
            if error.code != -32001 or args.attempts != 0:
                raise
            if display is not None:
                display.phase("gateway paused")
            if not paused:
                print(json.dumps(progress({"status": "gateway_paused", "reason": str(error)},getattr(args,"fast_profile",False))), flush=True)
            paused = True
            time.sleep(args.poll_seconds)
            continue
        paused = False
        if job.header == queued_header:
            if display is not None:
                display.phase("waiting for new template")
            time.sleep(args.poll_seconds)
            continue
        attempt += 1
        result = mine_once(args, rpc, miner, job, attempt, worker)
        print(json.dumps(progress(result,getattr(args,"fast_profile",False))), flush=True)
        if result["status"] == "gateway_queued":
            if args.attempts:
                return 0
            queued_header = job.header
        if result["status"] in ("gateway_rpc_error", "submission_unconfirmed", "proof_exceeds_gateway_limit", "refresh_failed"):
            return 1
    return 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--socket", required=True, type=Path, help="Unix socket of your configured local V4 Pearl Gateway")
    p.add_argument("--work-dir", required=True, type=Path, help="scratch space and verified proofs with receipts")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scanner", type=Path, help="compiled pearl-fp8-scan executable")
    mode.add_argument("--cpu-reference", action="store_true", help="use CPU reference calculation for small jobs")
    p.add_argument("--native-dir", type=Path, help="directory of the official V4 pearl_mining extension")
    add_geometry_options(p)
    add_scan_options(p)
    add_native_options(p)
    p.add_argument("--ancestor-depth", type=int, choices=range(1, 5), default=1)
    p.add_argument("--attempts", type=int, default=1, help="maximum fresh matrix pairs; 0 runs until interrupted")
    p.add_argument("--poll-seconds", type=float, default=2, help="continuous mode wait for a new or resumed job")
    p.add_argument("--rpc-timeout", type=float, default=10)
    args = p.parse_args()
    validate_stream_request(args)
    if args.attempts < 0 or any(not math.isfinite(v) or v <= 0 for v in (args.poll_seconds, args.rpc_timeout)):
        p.error("attempts must be nonnegative and wait times positive and finite")
    if args.sparse and not args.scanner:
        p.error("--sparse requires --scanner")
    if args.native_dense and not args.scanner:
        p.error("--native-dense requires --scanner")
    if args.native_full_reference and not args.native_dense:
        p.error("--native-full-reference requires --native-dense")
    if args.gpu_prover is not None and not args.native_dense:
        p.error("--gpu-prover requires --native-dense")
    if args.scanner and not args.scanner.is_file():
        p.error("scanner file is missing")
    if any(v is not None and v < 1 for v in (args.rows_per_pass, args.cols_per_pass)):
        p.error("CUDA pass dimensions must be positive")
    return connect(args)


def connect(args):
    """Use the operator's Unix socket after loading the local V4 verifier."""
    import host_miner as miner
    h, w = map(int, args.tile.lower().split("x"))
    miner.geometry(args.m, args.n, args.k, h, w)
    pm = miner.import_verifier(args.native_dir)
    if not hasattr(pm, "verify_plain_proof_for_cert_version") or not hasattr(pm, "PlainProofV4"):
        raise ValueError("the supplied native verifier lacks the V4 interface")
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with closing(GatewaySession(args.socket, args.rpc_timeout)) as session:
        session.open()
        return run(args, session, miner)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
