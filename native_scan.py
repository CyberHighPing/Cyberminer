"""Wire format and checked GPU scanner client for dense Pearl V4 jobs."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

from cuda_worker import (FAST_STRIP_ROWS, FUSED_STREAM_VARIANT, STREAM_COLS, STREAM_ROWS,
                         STREAM_VARIANTS, abort_on_error, scanner_environment, streamed_variant)
from lottery import CONTROL_REVISION as PIN, difficulty_bound, labelled, small_digest
import sparse_tickets as sparse

CHUNK = 1024
MAX_WITNESS = 1 << 22
GENERATOR = "blake3-keyed-xof-triangular-v2"
XOF_DOMAIN = b"pearl-5060/dense-xof/v2/"
ROLE = b"pearl/v4/FP8/"
TAG_DOMAIN = b"pearl/v4/sparse-job/v1\0"
CONTROL_MAGIC = b"PRLNF8J1"
ROOTS_MAGIC = b"PRLNF8R1"
CONTROL = struct.Struct("<8s5I32s32s32s32s9s19s32s32sI")
ROOTS = struct.Struct("<8s32s32s32s32s")
TAG_PARAMS = struct.Struct("<6I")
ROLE_A = small_digest(ROLE+b"seed-A")
ROLE_B = small_digest(ROLE+b"seed-B")


def parameters(m, n, k, h, w):
    # Dense FP8 E4M3 prequant, Blackwell, rank 32, 1024-byte keyed tree leaves.
    p_a = struct.pack("<IBI", m, 3, h)
    p_b = struct.pack("<IHBBIBIH", k, 32, 0, 1, n, 3, w, 0)
    return p_a, p_b


def double_sha(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def check_headers(proposed, ancestors):
    if len(proposed) != 76 or not 1 <= len(ancestors) <= 4:
        raise ValueError("expected a 76-byte proposed header and 1 to 4 complete ancestors")
    previous = proposed[4:36]
    for ancestor in ancestors:
        if len(ancestor) != 108 or double_sha(ancestor) != previous:
            raise ValueError("complete ancestor header does not link to its child")
        previous = ancestor[4:36]


def witness_bytes(m, n, k, h, w, row=0, col=0):
    """Dense, consecutive tile: four opened runs plus three BLAKE3 message blocks."""
    blocks = 3
    for total, start, count in ((m, row, h), (n, col, w)):
        for stride in (k, k//4):
            tree_leaves = (total*stride + CHUNK-1)//CHUNK
            depth = (tree_leaves-1).bit_length()
            first = start*stride//CHUNK
            end = ((start+count)*stride + CHUNK-1)//CHUNK
            blocks += (end-first)*(CHUNK//64 + 1) + 2*(depth+1)
    return blocks*64


def geometry(m, n, k, h, w):
    if not (0 < m <= 2**32 - 1 and 0 < n <= 2**32 - 1 and 1024 <= k <= 65536 and k % 32 == 0):
        raise ValueError("positive uint32 M/N and a 32-aligned K in 1024..65536 are required")
    if h < 4 or w < 16 or not 256 <= h*w <= 2048 or (h*w) % 16:
        raise ValueError("tile must have H>=4, W>=16, and 256..2048 cells in multiples of 16")
    if m < h or n < w:
        raise ValueError("M and N must contain at least one complete tile")
    first = witness_bytes(m, n, k, h, w)
    if first > MAX_WITNESS:
        raise ValueError(f"every tile exceeds the official BLAKE3 witness cap: "
                         f"at least {first} bytes; maximum {MAX_WITNESS}")


def synthetic_headers(bits):
    ancestor = struct.pack("<I", 1) + bytes(64) + struct.pack("<II", 1, bits) + bytes(32)
    header = (struct.pack("<I", 1) + double_sha(ancestor)
              + hashlib.sha256(b"fp8miner synthetic candidate").digest()
              + struct.pack("<II", 2, bits))
    ancestors = [ancestor]
    return header, ancestors


@dataclass(frozen=True)
class Plan:
    dims: tuple
    seed: bytes
    header: bytes
    ancestors: tuple

    def __post_init__(self):
        if len(self.dims)!=5 or any(type(n) is not int or not 0<n<1<<32 for n in self.dims):
            raise ValueError("dimensions must be five positive uint32s")
        geometry(*self.dims)
        check_headers(self.header, self.ancestors)
        if len(self.seed)!=32 or not self.bound:
            raise ValueError("need a 32-byte fresh operand seed and positive difficulty bound")
        if sparse.PIN!=PIN:
            raise ValueError("existing host modules do not match this protocol revision")

    @property
    def nbits(self): return struct.unpack_from("<I",self.header,72)[0]

    @property
    def bound(self):
        _,_,k,h,w=self.dims
        return difficulty_bound(self.nbits,h,w,k)

    @property
    def keys(self): return labelled(b"key-A",self.header),labelled(b"key-B",self.ancestors[-1])

    def prefix(self):
        return (TAG_DOMAIN + bytes.fromhex(PIN) + TAG_PARAMS.pack(4,*self.dims) + self.header
                + bytes([len(self.ancestors)]) + b"".join(self.ancestors))

    def encode(self):
        prefix=self.prefix()
        return CONTROL.pack(CONTROL_MAGIC,*self.dims,self.seed,*self.keys,self.bound.to_bytes(32,"little"),
                            *parameters(*self.dims),ROLE_A,ROLE_B,len(prefix)) + prefix

    @classmethod
    def decode(cls, encoded):
        if len(encoded)<CONTROL.size: raise ValueError("truncated control")
        fields=CONTROL.unpack_from(encoded)
        prefix=encoded[CONTROL.size:]
        fixed=TAG_DOMAIN+bytes.fromhex(PIN)
        if fields[0]!=CONTROL_MAGIC or fields[-1]!=len(prefix) or not prefix.startswith(fixed):
            raise ValueError("control magic, prefix length or protocol revision mismatch")
        offset=len(fixed)
        if len(prefix)<offset+TAG_PARAMS.size+77: raise ValueError("truncated commitment context")
        version,*pdims=TAG_PARAMS.unpack_from(prefix,offset)
        if version!=4 or tuple(pdims)!=fields[1:6]:raise ValueError("geometry disagrees with sparse tag context")
        at=offset+TAG_PARAMS.size; header=prefix[at:at+76];depth=prefix[at+76];chain=prefix[at+77:]
        if not 1<=depth<=4 or len(chain)!=108*depth:raise ValueError("incorrect ancestor window")
        plan=cls(tuple(pdims),fields[6],header,tuple(chain[i:i+108] for i in range(0,len(chain),108)))
        if plan.encode()!=encoded:raise ValueError("control keys, parameters, bound or context are inconsistent")
        return plan

    def meta(self, roots):
        ha,hb,sa,sb=roots
        return dict(pin=PIN,cert_version=4,generator=GENERATOR,source="native dense control",seed_hex=self.seed.hex(),
                    **dict(zip(("m","n","k","h","w"),self.dims,strict=True)),nbits=self.nbits,
                    proposed_header_hex=self.header.hex(),ancestor_headers_hex=[v.hex() for v in self.ancestors],
                    hash_a=ha.hex(),hash_b=hb.hex(),seed_a=sa.hex(),seed_b=sb.hex())


def read_control(path):
    path=Path(path)
    return Plan.decode((path / "control.bin" if path.is_dir() else path).read_bytes())


def derive(plan,ha,hb):
    ka,kb=plan.keys;pa,pb=parameters(*plan.dims)
    sb=small_digest(hb+kb+pb,ROLE_B)
    sa=small_digest(ha+sb+ka+pa,ROLE_A)
    return ha,hb,sa,sb


def load_roots(plan, path):
    data=Path(path).read_bytes()
    if len(data)!=ROOTS.size:raise ValueError("roots.bin must be 136 bytes")
    magic,*rows=ROOTS.unpack(data)
    if magic!=ROOTS_MAGIC or tuple(rows)!=derive(plan,*rows[:2]):
        raise ValueError("device root record magic or derived seeds mismatch")
    return tuple(rows)


def params(plan,roots):
    raw=sparse.make_params(plan.meta(roots))
    if raw[128:]!=hashlib.sha256(plan.prefix()+b"".join(roots)).digest():
        raise ValueError("sparse tag serialization drift")
    return raw


def parse_hits(plan,roots,path):
    if not str(path).endswith(".phits"):raise ValueError("use the existing .phits sparse format")
    # fully validates the header, counts, duplicate indices, threshold and ticket hash
    return sum(1 for _ in sparse.verified_records(plan.meta(roots),path))


def check_native_report(report, plan, args, hits):
    if not isinstance(report, dict):
        raise ValueError("the native scanner must return one JSON object")
    m, n, k, h, w = plan.dims
    streaming = getattr(args,"stream_b",False)
    fast = getattr(args, "fast_profile", False)
    if report.get("streamed_b",False) is not (streaming and not fast) or report.get("streamed_a",False) is not fast:
        raise ValueError("native scanner acknowledged the wrong streamed operand")
    expected = {"mode":"native_job", "m":m, "n":n, "k":k, "tile_h":h, "tile_w":w,
                "resident_operands":not streaming, "sparse_filter":True, "tiles":(m//h)*(n//w), "threshold_hits":hits}
    variant = getattr(args, "variant", None) or "auto"
    if streaming:
        variant = streamed_variant(args,plan.dims)
        windows = (m//STREAM_ROWS)*(n//STREAM_COLS)
        direct = variant==FUSED_STREAM_VARIANT
        expected.update(rows_per_pass=STREAM_ROWS,cols_per_pass=STREAM_COLS,k_per_pass=k,row_pitch_bytes=k,
                        passes=windows,deferred_filter=True,overlapped_fold=not direct and m>STREAM_ROWS,
                        coalesced_fold=not direct,gemm_includes_fold=direct)
    if variant != "auto": expected["variant"] = variant
    if getattr(args, "row_pad", 0) or "row_pitch_bytes" in report:
        expected["row_pitch_bytes"] = k + getattr(args, "row_pad", 0)
    for field, value in expected.items():
        if type(report.get(field)) is not type(value) or report[field] != value:
            raise ValueError(f"native scanner returned incorrect {field}")
    if not isinstance(report.get("variant"),str) or not report["variant"]:
        raise ValueError("native scanner omitted its selected variant")
    for field in ("scan_wall_s", "generate_commit_s", "noise_quantize_s", "kernel_tmac_s", "scan_equivalent_th_s"):
        value = report.get(field)
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"native scanner returned invalid {field}")
    if streaming:
        for field in ("gemm_ms","fold_ms","regenerate_b_s","ticket_wall_s"):
            value=report.get(field)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"streamed scanner returned invalid {field}")
        for duration,rate,unit in (("scan_wall_s","scan_equivalent_th_s",1e12),("gemm_ms","kernel_tmac_s",1e9)):
            if report[duration]<=0 or not math.isclose(report[rate],m*n*k/report[duration]/unit,rel_tol=2e-8):
                raise ValueError(f"streamed scanner reported inconsistent {rate} for the complete dense work")
        if direct:
            if fast:
                strip = min(FAST_STRIP_ROWS, m)
                strips = (m + strip - 1) // strip
                edge = n % 192
                if edge == 64:
                    edge = 256
                fold_bytes = (strip // h) * (n // w) * 16 * 4
                body_bytes = n*k + strip*k
                geometry = {"resident_operand": "B", "computed_output_cells": m*n,
                            "physical_a_strip_rows": strip, "physical_a_strips": strips,
                            "physical_matrix_kernel_launches": strips*(int(n>edge)+int(edge>0)),
                            "B_reuse_grid_N": 1, "A_streamed_value_leaves": True,
                            "resident_A_segment": 128, "resident_B_segment": 256,
                            "requested_native_fold": "paired", "fold_domain_rows": m+n}
                if any(type(report.get(f)) is not type(v) or report[f]!=v for f,v in geometry.items()):
                    raise ValueError("the 8 GB scanner did not cover the requested strips and operands")
                if report.get("actual_native_fold") not in ("paired","canonical"):
                    raise ValueError("unknown native fold for checked row data")
            else:
                fold_bytes = (m//h)*(STREAM_COLS//w)*16*4
                body_bytes = m*k + STREAM_COLS*k
            exact={"fused_stream_calls":windows,"raw_product_buffer_bytes":0,"cublaslt_calls":0,"cublaslt_attempts":0,
                   "cublaslt_fallback_calls":0,"cutlass_fallback_calls":0,"native_fallback_calls":0,"cublaslt_workspace_bytes":0,
                   "fold_buffer_bytes":fold_bytes}
            if any(type(report.get(key)) is not int or report[key]!=value for key,value in exact.items()):
                raise ValueError("fused scanner reported inconsistent launches or buffers")
            allocations=[report.get(key) for key in ("arena_bytes","job_scratch_bytes","filter_counter_bytes","explicit_gpu_bytes")]
            if (report["fold_ms"]!=0 or report.get("rect_fold_kernel")!="fused"
                    or any(type(x) is not int or x<=0 for x in allocations) or sum(allocations[:3])!=allocations[3]
                    or allocations[0]<body_bytes+fold_bytes):
                raise ValueError("fused scanner returned inconsistent fold timing or GPU allocation")
        if variant==STREAM_VARIANTS[0]:
            if "fused_stream_calls" in report:
                fused_calls=report["fused_stream_calls"]
                if type(fused_calls) is not int or fused_calls!=0:
                    raise ValueError("vendor scanner reported an invalid fused launch count")
            counts = [report.get(f) for f in ("cublaslt_calls","cublaslt_attempts","cublaslt_fallback_calls",
                                               "cutlass_fallback_calls","native_fallback_calls")]
            if (any(type(v) is not int or not 0<=v<=windows for v in counts) or counts[0]+counts[2]!=windows
                    or counts[1]<counts[0] or counts[2]!=counts[3] or counts[4]!=0):
                raise ValueError("streamed scanner did not account for each dense multiplication window")


def scan_plan(plan, args, stage, worker=None):
    """Scan one encoded job and check its roots, complete hit file and work accounting."""
    m, n, k, h, w = plan.dims
    streaming = getattr(args,"stream_b",False)
    variant = streamed_variant(args,plan.dims) if streaming else (getattr(args,"variant",None) or "auto")
    control = stage / "control.bin"
    roots_file = stage / "roots.bin"
    hits_file = stage / "hits.phits"
    control.write_bytes(plan.encode())
    flags = ["--native-job", str(control), "--m", str(m), "--n", str(n), "--k", str(k),
             "--tile", f"{h}x{w}", "--variant", variant, "--roots-out", str(roots_file),
             "--out", str(hits_file), "--hit-capacity", str(getattr(args,"hit_capacity",256))]
    if streaming:
        flags.extend(("--stream-b","--rows-per-pass",str(STREAM_ROWS),"--cols-per-pass",str(STREAM_COLS),"--defer-filter"))
    else:
        for option in ("rows_per_pass", "cols_per_pass", "row_pad"):
            value = getattr(args, option, None)
            if value:
                flags.extend(("--" + option.replace("_","-"), str(value)))
        flags.extend("--" + option.replace("_","-") for option in ("fuse_fold", "defer_filter", "resident") if getattr(args,option,False))
    if getattr(args,"k_per_pass",None) is not None:
        flags.extend(("--k-per-pass",str(args.k_per_pass)))
    with abort_on_error(worker):
        if worker is None:
            result = subprocess.run([str(args.scanner.resolve()), *flags], check=True, stdout=subprocess.PIPE,
                                    text=True, env=scanner_environment(args))
            report = json.loads(result.stdout)
        else:
            report = worker.request(flags)
        roots = load_roots(plan, roots_file)
        count = parse_hits(plan, roots, hits_file)
        check_native_report(report, plan, args, count)
    return roots, count, report
