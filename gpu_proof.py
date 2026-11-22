"""Bind an external dense V4 proof to its source seed, exact tile and official verifier."""

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
import struct
import subprocess
import tempfile
import time

import numpy as np
from blake3 import blake3

import host_miner as host
from host_math import b200_mm
from lottery import fold_cells, jackpot_key, meets_bound, ticket_digest
from native_job import XOF_DOMAIN, derive, encode_global_rows, load_roots, read_control, source_chunk
from sparse_tickets import verified_records


class Reader:
    def __init__(self, raw):
        self.data = memoryview(raw)
        self.offset = 0

    def take(self, size):
        stop = self.offset + size
        if stop > len(self.data):
            raise ValueError("truncated dense PlainProofV4")
        data = self.data[self.offset:stop]
        self.offset = stop
        return data

    def integer(self):
        return struct.unpack("<Q", self.take(8))[0]

    def count(self, expected, field):
        if self.integer() != expected:
            raise ValueError(f"wrong {field} count in dense V4 proof")

    def blob(self, expected, field):
        self.count(len(expected), field)
        if self.take(len(expected)) != expected:
            raise ValueError(f"dense V4 proof differs in {field}")

    def indices(self, expected, field):
        self.count(len(expected), field)
        for index in expected:
            if self.integer() != index:
                raise ValueError(f"wrong {field} index in dense V4 proof")


@dataclass(frozen=True)
class Opening:
    side: bytes
    rows: int
    first: int
    opened: int
    stride: int
    scales: bool

    @property
    def leaves(self):
        return range(self.first*self.stride//host.CHUNK,
                     ((self.first+self.opened)*self.stride + host.CHUNK-1)//host.CHUNK)

    @property
    def total_leaves(self):
        return (self.rows*self.stride + host.CHUNK-1)//host.CHUNK

    @property
    def max_siblings(self):
        # A contiguous interval exposes at most its two outside neighbours per tree level.
        return 2*(self.total_leaves-1).bit_length()

    def expected_leaves(self, seed):
        size = len(self.leaves)*host.CHUNK
        start = self.leaves.start*host.CHUNK
        actual = min(size, self.rows*self.stride-start)
        if self.scales:
            content = host.SCALE_ONE*(actual//2)
        else:
            xof = blake3(XOF_DOMAIN+self.side, key=seed)
            content = source_chunk(xof,start,actual).tobytes()
        return content+bytes(size-actual)

    def read(self, reader, seed):
        expected = self.expected_leaves(seed)
        reader.count(len(self.leaves), "opened leaves")
        plane = memoryview(expected)
        for off in range(0,len(plane),host.CHUNK):
            reader.blob(plane[off:off+host.CHUNK], "source leaf bytes")
        reader.indices(self.leaves, "leaf")
        if reader.integer()!=self.total_leaves:
            raise ValueError("wrong total leaf count in dense V4 proof")
        root = bytes(reader.take(32))
        siblings = reader.integer()
        if siblings>self.max_siblings:
            raise ValueError("too many Merkle siblings for this contiguous opening")
        reader.take(32*siblings)  # The official verifier authenticates the path.
        reader.indices(range(self.first,self.first+self.opened), "opened row")
        if self.scales:
            return root, None
        offset = self.first*self.stride - self.leaves.start*host.CHUNK
        rows = np.frombuffer(expected, dtype=np.int8, offset=offset, count=self.opened*self.stride)
        return root, rows.reshape(self.opened,self.stride)

    def max_wire_bytes(self):
        return 72 + len(self.leaves)*(host.CHUNK+16) + 32*self.max_siblings + 8*self.opened


def tile_openings(plan, tile):
    m,n,k,h,w=plan.dims
    total=(m//h)*(n//w)
    if type(tile) is not int or not 0<=tile<total:
        raise ValueError("candidate tile is out of range")
    row=(tile//(n//w))*h
    col=(tile%(n//w))*w
    if host.witness_bytes(m,n,k,h,w,row,col)>host.MAX_WITNESS:
        raise ValueError("candidate exceeds the official witness cap")
    openings = (Opening(b"A",m,row,h,k,False), Opening(b"B",n,col,w,k,False),
                Opening(b"A",m,row,h,k//4,True), Opening(b"B",n,col,w,k//4,True))
    return row,col,openings


def max_proof_bytes(plan, openings):
    pa,pb=host.parameters(*plan.dims)
    statements=16+len(plan.ancestors)*108+len(pa)+len(pb)+1
    return statements + sum(p.max_wire_bytes() for p in openings)


def check_openings(raw, plan, roots, openings):
    reader=Reader(raw)
    pa,pb=host.parameters(*plan.dims)
    reader.blob(plan.ancestors[-1]+pb+pa, "ancestor and committed parameters")
    reader.blob(b"".join(plan.ancestors[:-1]), "complete ancestor chain")
    parsed=[p.read(reader, plan.seed) for p in openings]
    if reader.take(1)!=b"\0" or reader.offset!=len(reader.data):
        raise ValueError("expected a complete dense proof without an MoE extension or trailing bytes")
    key_a,key_b=plan.keys
    hash_a = blake3(parsed[0][0]+parsed[2][0],key=key_a).digest()
    hash_b = blake3(parsed[1][0]+parsed[3][0],key=key_b).digest()
    if derive(plan,hash_a,hash_b)!=roots:
        raise ValueError("the four opened roots do not reproduce the scanner commitments and noise seeds")
    return parsed[0][1],parsed[1][1]


def check_fold(plan,roots,row,col,source_rows,fold,digest):
    _,_,_,h,w=plan.dims
    a=encode_global_rows(source_rows[0],0,row,roots[2],roots[3])
    b=encode_global_rows(source_rows[1],1,col,roots[3],roots[3])
    actual_fold=fold_cells(b200_mm(a,b).view(np.uint32).ravel(),h,w)
    actual_digest=ticket_digest(jackpot_key(roots[2]),actual_fold)
    if actual_fold!=fold or actual_digest!=digest or not meets_bound(actual_digest,plan.bound):
        raise ValueError("CPU reconstruction of the selected source rows differs from this sparse ticket")


@contextmanager
def timed(timings, field):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[field] = timings.get(field,0.0) + (time.perf_counter()-started)


def read_plain_file(path, limit, declared_size=None):
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=limit:
        raise ValueError("provided proof is not a regular file of bounded size")
    if declared_size is not None and (type(declared_size) is not int or declared_size!=info.st_size):
        raise ValueError("prover output size differs from its acknowledgement")
    with path.open("rb") as src:
        raw=src.read(limit+1)
    if len(raw)!=info.st_size:
        raise ValueError("proof changed while being read")
    return raw,hashlib.sha256(raw).hexdigest()


def verify_candidate(pm, header, plan, roots, tile, location, fold, digest, raw, timings, *, full_directory=None):
    """Use the same exact verification for newly generated and already transferred bytes."""
    row,col,openings=location
    with timed(timings,"opened_source_s"):
        source_rows=check_openings(raw,plan,roots,openings)
    with timed(timings,"official_decode_verify_s"):
        proof=pm.PlainProofV4.from_base64(base64.b64encode(raw).decode("ascii"))
        valid,detail=pm.verify_plain_proof_for_cert_version(4,header,proof)
    if not valid:
        if str(detail)=="The jackpot is not admissible":
            return False
        raise ValueError(f"official V4 verification failed: {detail}")
    with timed(timings,"candidate_fold_digest_s"):
        check_fold(plan,roots,row,col,source_rows,fold,digest)
    if full_directory is not None:
        with timed(timings,"full_reference_tile_s"):
            host.check_tile(full_directory,plan.meta(roots),tile,digest,expected_fold=fold)
    with timed(timings,"official_serialize_replay_s"):
        serialized=proof.to_base64()
        if base64.b64decode(serialized,validate=True)!=raw:
            raise ValueError("official serialization changed the candidate proof")
        replay=pm.PlainProofV4.from_base64(serialized)
        valid,detail=pm.verify_plain_proof_for_cert_version(4,header,replay)
        if not valid:
            raise ValueError(f"independent official serialized replay failed: {detail}")
    return True


def check_existing_proof(control_file, roots_file, hits_file, tile, proof_file, native_dir):
    """Read five existing inputs and verify the selected ticket without writing a file."""
    start=time.perf_counter()
    timings={}
    with timed(timings,"control_roots_s"):
        plan=read_control(control_file)
        roots=load_roots(plan,roots_file)
        location=tile_openings(plan,tile)
    with timed(timings,"sparse_records_s"):
        chosen=None
        count=0
        for idx,fold,digest in verified_records(plan.meta(roots),hits_file):
            count+=1
            if idx==tile:
                chosen=fold,digest
        if chosen is None:
            raise ValueError(f"tile {tile} does not appear among the checked difficulty hits")
    with timed(timings,"read_proof_sha256_s"):
        raw,sha=read_plain_file(Path(proof_file),max_proof_bytes(plan,location[2]))
    with timed(timings,"load_official_verifier_s"):
        pm=host.import_verifier(native_dir)
        header=pm.IncompleteBlockHeader.from_bytes(plan.header)
    fold,digest=chosen
    if not verify_candidate(pm,header,plan,roots,tile,location,fold,digest,raw,timings):
        raise ValueError("official V4 verification failed: The jackpot is not admissible")
    with timed(timings,"native_module_sha256_s"):
        module_sha=host._stream_sha(Path(pm.__file__))
    timings["total_s"]=time.perf_counter()-start
    m,n,_,h,w=plan.dims
    return {"status":"plain_verified","protocol":host.PIN,"submission":"offline","proof_inputs":"provided_file",
            "tickets":(m//h)*(n//w),"checked_digest_records":count,"tile":tile,"row":location[0],"col":location[1],
            "fold_hex":fold.hex(),"target_digest_le":digest.hex(),"proof_bytes":len(raw),"proof_sha256":sha,
            "native_module_sha256":module_sha,"cpu_recomputed_roots":False,"cpu_verified_source_openings":True,
            "timings":timings}


def external_proof(program, directory, tile, out, limit, *, env=None):
    if out.exists(): raise FileExistsError(out)
    cmd=[str(program.resolve()), "--control", str((directory/"control.bin").resolve()),
         "--roots", str((directory/"roots.bin").resolve()), "--tile-index",str(tile),"--out",str(out.resolve())]
    reply=subprocess.run(cmd,check=True,text=True,capture_output=True,env=env).stdout
    if len(reply)>65536:
        raise ValueError("oversized prover acknowledgement")
    receipt=json.loads(reply)
    if not isinstance(receipt,dict) or type(receipt.get("tile")) is not int or receipt["tile"]!=tile:
        raise ValueError("prover acknowledged the wrong tile")
    if type(receipt.get("proof_bytes")) is not int:
        raise ValueError("prover must acknowledge an integer proof size")
    data,sha=read_plain_file(out,limit,receipt["proof_bytes"])
    if "proof_sha256" in receipt and receipt["proof_sha256"]!=sha:
        raise ValueError("prover SHA-256 acknowledgement does not match the proof bytes")
    return data,sha


def prove_candidates(plan,roots,hits_file,directory,program,native_dir,*,full_reference=False,prover_env=None):
    meta=plan.meta(roots)
    m,n,_,h,w=plan.dims
    total=(m//h)*(n//w)
    pm=host.import_verifier(native_dir)
    header=pm.IncompleteBlockHeader.from_bytes(plan.header)
    checked=0
    build_s=0.0
    timings={}
    for tile,fold,digest in host.ticket_records(directory,meta,hits_file):
        location=tile_openings(plan,tile)
        row,col,openings=location
        checked+=1
        with tempfile.TemporaryDirectory(prefix=".gpu-proof-",dir=directory) as tmp:
            started=time.monotonic()
            raw,sha=external_proof(program,directory,tile,Path(tmp)/"candidate.plain.bin",max_proof_bytes(plan,openings),
                                   env=prover_env)
            build_s+=time.monotonic()-started
        if not verify_candidate(pm,header,plan,roots,tile,location,fold,digest,raw,timings,
                                full_directory=directory if full_reference else None):
            continue
        result_file=directory/f"tile-{tile}.plain.bin"
        with result_file.open("xb") as out:
            out.write(raw)
        return {"status":"plain_verified", "tickets":total, "hits_checked":checked, "tile":tile, "row":row, "col":col,
                "proof_inputs":"gpu_prover", "gpu_prover_s":build_s,"verification_timings":timings, "cpu_recomputed_roots":full_reference,
                "cpu_verified_source_openings":True,"target_digest_le":digest.hex(), "plain":result_file.name,
                "proof_bytes":len(raw),"proof_sha256":sha,"native_module_sha256":host._stream_sha(Path(pm.__file__))}
    return {"status":"no_admissible_hit","tickets":total,"hits_checked":checked,"gpu_prover_s":build_s,
            "proof_inputs":"gpu_prover","cpu_recomputed_roots":full_reference}
