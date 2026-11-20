"""CPU planner, independent commitment checker and replay for native dense V4 jobs."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import secrets
import tempfile
import time

import numpy as np
from blake3 import blake3
from cuda_worker import STREAM_VARIANTS, abort_on_error, add_geometry_options, streamed_variant

import host_miner as host
from host_math import prepare_operand
import sparse_tickets as sparse


from native_scan import (PIN, GENERATOR, XOF_DOMAIN, ROLE, TAG_DOMAIN, CONTROL_MAGIC, ROOTS_MAGIC, CONTROL,
                         ROOTS, TAG_PARAMS, ROLE_A, ROLE_B, Plan, read_control, derive, load_roots, params,
                         parse_hits, check_native_report, scan_plan)
BLOCK_CELLS = 1 << 20
ONES = b"\x80\x3f" * (1 << 19)


def pretty(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2); out.write("\n")


def source_chunk(xof,start,count):
    if start<0 or count<0: raise ValueError("negative operand slice")
    bits=np.frombuffer(xof.digest(length=2*count,seek=2*start),dtype=np.uint8).reshape(-1,2)
    return ((bits[:,0]&127).astype(np.int16)+(bits[:,1]&127)-127).astype(np.int8)


def pad_hash(hasher,size):
    hasher.update(bytes(-size%1024));return hasher.digest()


def operand(seed,side,rows,k,key,save=None,raw_sha=None):
    cells=rows*k
    xs=blake3(XOF_DOMAIN+side,key=seed)
    vals=blake3(key=key); scales=blake3(key=key)
    with (save.open("xb") if save is not None else nullcontext(None)) as out:
        for i in range(0,cells,BLOCK_CELLS):
            codes=source_chunk(xs,i,min(BLOCK_CELLS,cells-i)); vals.update(codes)
            if out is not None:out.write(codes)
            if raw_sha is not None:raw_sha.update(codes)
    scale_size=cells//4
    for i in range(0,scale_size,len(ONES)):scales.update(ONES[:min(len(ONES),scale_size-i)])
    vr,sr=pad_hash(vals,cells),pad_hash(scales,scale_size)
    return blake3(vr+sr,key=key).digest(),vr,sr


def expected_roots(plan, save=None, hashes=None):
    m,n,k,h,w=plan.dims
    pieces={}
    for side,rows,key in zip((b"A",b"B"),(m,n),plan.keys,strict=True):
        who=side.decode("ascii")
        raw_sha = hashlib.sha256() if hashes is not None else None
        parts=operand(plan.seed,side,rows,k,key,(save/(who+".i8")) if save is not None else None,raw_sha)
        if raw_sha is not None:
            hashes[who+".i8"] = raw_sha.hexdigest()
        pieces[who]=dict(zip(("commitment","value_root","scale_root"),(x.hex() for x in parts),strict=True))
    roots=derive(plan,bytes.fromhex(pieces["A"]["commitment"]),bytes.fromhex(pieces["B"]["commitment"]))
    return roots,pieces


def publish(directory, create):
    directory=directory.resolve()
    if directory.exists():raise FileExistsError(directory)
    directory.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-job-",dir=directory.parent) as tmp:
        stage=Path(tmp)/"new";stage.mkdir()
        result=create(stage)
        stage.rename(directory)
    return result


def check_raw_context(directory, meta):
    """Require the on-disk proof job to encode the original control and root receipt."""
    plan = read_control(directory)
    roots = load_roots(plan, directory / "roots.bin")
    expected = plan.meta(roots)
    if any(meta.get(name) != value for name, value in expected.items()):
        raise ValueError("raw-row proof metadata differs from the committed GPU job")
    return plan


def candidate_panels(directory, meta, row, col):
    """Encode only opened rows, retaining their original global noise addresses."""
    k = meta["k"]
    seed_a = bytes.fromhex(meta["seed_a"])
    seed_b = bytes.fromhex(meta["seed_b"])
    panels = []
    for side, name, rows, first, count, e_seed in (
        (0, "A", meta["m"], row, meta["h"], seed_a),
        (1, "B", meta["n"], col, meta["w"], seed_b),
    ):
        if not 0 <= first <= rows - count:
            raise ValueError("opened rows exceed their committed operand")
        codes = np.memmap(directory / f"{name}.i8", mode="r", dtype=np.int8,
                          offset=first*k, shape=(count, k))
        try:
            panels.append(encode_global_rows(codes,side,first,e_seed,seed_b))
        finally:
            del codes
    return panels


def encode_global_rows(codes, side, first, e_seed, seed_b):
    from fast_prepare import Rank32Product
    from host_math import noise_lines, row_parameters, unit_scale_norms

    count,k=codes.shape
    product=Rank32Product(noise_lines(seed_b,side,1,range(k)))
    encoded=np.empty((count,k),dtype=np.uint8)
    for start in range(0,count,32):
        stop=min(count,start+32)
        block=codes[start:stop]
        e=noise_lines(e_seed,side,0,range(first+start,first+stop))
        product.write_mixed(e,block,*row_parameters(*unit_scale_norms(block)),encoded[start:stop])
    return encoded


def proof_for_output(plan, roots, hits, directory, native_dir, *, checked_count, full_reference=False, gpu_prover=None):
    """Skip host operand generation until a checked, witness-eligible record exists."""
    meta = plan.meta(roots)
    total = (meta["m"]//meta["h"])*(meta["n"]//meta["w"])
    first = next(host.ticket_records(directory, meta, hits), None) if checked_count else None
    if first is None:
        # verified_records exhaustively checks the complete sparse file before
        # yielding, including when every returned record exceeds the witness cap.
        pretty(directory / "job.json", meta)
        status = "no_admissible_hit" if checked_count else "no_difficulty_hit"
        return {"status":status, "tickets":total, "hits_checked":0,
                "cpu_recomputed_roots":False}
    if full_reference:
        build(plan, roots, directory, True)
    if gpu_prover is not None:
        from gpu_proof import prove_candidates
        if not full_reference:pretty(directory/"job.json",dict(meta,proof_inputs="gpu_prover"))
        return prove_candidates(plan,roots,hits,directory,Path(gpu_prover),native_dir,full_reference=full_reference)
    if not full_reference:
        hashes = {}
        regenerated, pieces = expected_roots(plan, directory, hashes)
        if regenerated != roots:
            raise ValueError("GPU roots differ from independently regenerated source operands")
        meta.update(proof_inputs="candidate_rows", sha256=hashes)
        pretty(directory / "commitments.json", pieces)
        pretty(directory / "job.json", meta)
    verified = host.finish(directory, hits, native_dir)
    verified.update(cpu_recomputed_roots=True, proof_inputs="full_planes" if full_reference else "candidate_rows")
    return verified


def attempt(prep, args, directory, worker=None):
    """Prepare a job, scan it and open eligible tickets using the same checked outputs."""
    if not args.scanner or getattr(args, "cpu_reference", False):
        raise ValueError("--native-dense requires a GPU scanner")
    header, ancestors, _ = host.headers(prep)
    h, w = map(int, prep.tile.lower().split("x"))
    plan = Plan((prep.m, prep.n, prep.k, h, w), bytes.fromhex(prep.seed) if prep.seed else secrets.token_bytes(32),
                header, tuple(ancestors))

    def launch(stage):
        display = getattr(args, "rate_display", None)
        if display is not None:
            display.phase("scanning")
            scan_start = time.perf_counter()
        roots, count, report = scan_plan(plan, args, stage, worker)
        if display is not None:
            end = time.perf_counter()
            display.completed(prep.m*prep.n*prep.k, end-scan_start, end,
                              "opening / verifying" if count else "scan complete")
        with abort_on_error(worker):
            (stage / "job.params").write_bytes(params(plan, roots))
            pretty(stage / "scan-report.json", report)
        if not getattr(args, "fast_profile", False):
            print(json.dumps(report), flush=True)
        outcome = proof_for_output(plan, roots, stage/"hits.phits", stage, args.native_dir, checked_count=count,
                                   full_reference=getattr(args,"native_full_reference",False),gpu_prover=getattr(args,"gpu_prover",None))
        outcome["threshold_hits"] = count
        if getattr(args,"stream_b",False):
            outcome.update(streamed_b=report["streamed_b"],dense_macs=prep.m*prep.n*prep.k,covered_windows=report["passes"],
                           scan_wall_s=report["scan_wall_s"],scan_equivalent_th_s=report["scan_equivalent_th_s"])
            if getattr(args, "fast_profile", False):
                outcome["streamed_a"] = True
        return outcome

    return publish(directory, launch)


def build(plan, reported, stage, materialize):
    roots,pieces=expected_roots(plan,stage if materialize else None)
    if reported is not None and roots!=reported: raise ValueError("GPU commitment differs from independently regenerated source operands")
    meta=plan.meta(roots)
    (stage/"control.bin").write_bytes(plan.encode());(stage/"roots.bin").write_bytes(ROOTS.pack(ROOTS_MAGIC,*roots))
    (stage/"job.params").write_bytes(params(plan,roots))
    pretty(stage/"commitments.json",pieces)
    if materialize:
        m,n,k,h,w=plan.dims
        for side,who,rows in ((0,"A",m),(1,"B",n)):
            raw=np.memmap(stage/(who+".i8"),mode="r",shape=(rows,k),dtype=np.int8)
            out=np.memmap(stage/(who+".e4m3"),mode="w+",shape=(rows,k),dtype=np.uint8)
            try:
                prepare_operand(raw,side,roots[2+side],roots[3],out=out);out.flush()
            finally:
                del raw,out
        np.arange(h*w,dtype="<u2").tofile(stage/"lanes.u16le")
        meta["sha256"]={n:host._stream_sha(stage/n) for n in ("A.i8","B.i8","A.e4m3","B.e4m3","lanes.u16le")}
    pretty(stage/"job.json",meta)
    return {"status":"materialized" if materialize else "roots_cpu_verified", "cpu_recomputed_roots":True,
            "hash_a":roots[0].hex(),"hash_b":roots[1].hex(),"tag":params(plan,roots)[128:].hex()}


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("plan",help="create a one-request GPU control with no source-matrix generation")
    a.add_argument("directory",type=Path);a.add_argument("--job",type=Path,help="V4 header+ancestor JSON; absent means a synthetic offline job")
    a.add_argument("--bits",default="1e0fffff");a.add_argument("--seed",help="32-byte hex seed (default fresh random bytes)")
    add_geometry_options(a)
    a.add_argument("--stream-b",action="store_true",help="check that the unchanged native control geometry supports streamed B")
    for mode in ("inspect","reference","verify","materialize","prove"):
        a=sub.add_parser(mode);a.add_argument("control",type=Path)
        if mode!="reference":a.add_argument("roots",type=Path,help="device roots.bin")
        if mode in ("inspect","prove"):a.add_argument("hits",type=Path,help="device .phits")
        if mode!="inspect":a.add_argument("directory",type=Path,help="new output directory")
        if mode=="reference":a.add_argument("--materialize",action="store_true")
        if mode=="inspect":
            a.add_argument("--scan-report",type=Path,help="also verify the scanner JSON against this control, the full coverage and the sparse file")
            a.add_argument("--stream-b",action="store_true",help="require a matching streamed-B scan report")
            a.add_argument("--variant",choices=STREAM_VARIANTS,help="require this kernel variant in the scan report")
        if mode=="prove":
            a.add_argument("--native-dir",type=Path,help="local pinned official pearl_mining module directory")
            a.add_argument("--full-reference",action="store_true",help="materialize complete CPU reference FP8 planes")
            a.add_argument("--gpu-prover",type=Path,help="external dense V4 Merkle proof constructor executable")
    a=sub.add_parser("check-proof",help="verify an existing dense proof against a specific returned sparse ticket; reads files only")
    a.add_argument("control",type=Path);a.add_argument("roots",type=Path);a.add_argument("hits",type=Path)
    a.add_argument("tile",type=int,help="global decimal tile index");a.add_argument("proof",type=Path,help="raw V4 proof from the remote GPU")
    a.add_argument("--native-dir",type=Path,help="directory containing the pinned official pearl_mining verifier")
    return p.parse_args()


def main():
    a=arguments()
    if a.command=="check-proof":
        from gpu_proof import check_existing_proof
        result=check_existing_proof(a.control,a.roots,a.hits,a.tile,a.proof,a.native_dir)
    elif a.command=="plan":
        header,ancestors,source=host.headers(a);h,w=map(int,a.tile.lower().split("x"))
        plan=Plan((a.m,a.n,a.k,h,w),bytes.fromhex(a.seed) if a.seed is not None else secrets.token_bytes(32),header,tuple(ancestors))
        if a.stream_b: streamed_variant(a,plan.dims)
        def create(d):
            encoded=plan.encode();(d/"control.bin").write_bytes(encoded)
            summary={"status":"planned","source":source,"generator":GENERATOR,"pin":PIN,"dimensions":plan.dims,
                     "source_seed_hex":plan.seed.hex(),"bytes":len(encoded),"sha256":hashlib.sha256(encoded).hexdigest()}
            if a.stream_b: summary["stream_b_compatible"]=True
            pretty(d/"plan.json",summary);return summary
        result=publish(a.directory,create)
    else:
        plan=read_control(a.control)
        roots=load_roots(plan,a.roots) if a.command!="reference" else None
        if a.command in ("reference","verify","materialize"):
            result=publish(a.directory,lambda d:build(plan,roots,d,a.command=="materialize" or getattr(a,"materialize",False)))
        else:
            count=parse_hits(plan,roots,a.hits)
            result={"status":"device_record_consistent" if count else "no_reported_difficulty_hit",
                    "cpu_recomputed_roots":False,"checked_digest_records":count,"tag":params(plan,roots)[128:].hex()}
            if a.command=="inspect":
                if (a.stream_b or a.variant) and not a.scan_report:
                    raise ValueError("--stream-b or --variant requires --scan-report when inspecting a job")
                if a.scan_report:
                    report=json.loads(a.scan_report.read_bytes())
                    if a.stream_b and a.variant is None and isinstance(report,dict): a.variant=report.get("variant")
                    check_native_report(report,plan,a,count)
                    result.update(scan_report_verified=True,streamed_b=a.stream_b,covered_tickets=report["tiles"])
                    if a.stream_b:
                        m,n,k,_,_=plan.dims
                        result.update(dense_macs=m*n*k,covered_windows=report["passes"])
            if a.command=="prove" and count:
                def prove(stage):
                    (stage/"control.bin").write_bytes(plan.encode())
                    (stage/"roots.bin").write_bytes(ROOTS.pack(ROOTS_MAGIC,*roots))
                    (stage/"job.params").write_bytes(params(plan,roots))
                    return proof_for_output(plan,roots,a.hits,stage,a.native_dir,checked_count=count,full_reference=a.full_reference,gpu_prover=a.gpu_prover)
                result=publish(a.directory,prove)
                result["checked_digest_records"]=count
    print(json.dumps(result,indent=2));return 0


if __name__=="__main__":
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError,ImportError) as error:raise SystemExit(f"error: {error}")
