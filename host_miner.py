"""Offline dense Pearl certificate-v4 miner for the pinned fp8-scheme protocol."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
from blake3 import blake3

from cuda_worker import abort_on_error, add_worker_options, capacity_arg, check_file_scan, scanner_session, scan_options, validate_stream_request
from cuda_worker import add_geometry_options, add_native_options, add_scan_options
from fast_prepare import MappedPages
from host_math import b200_mm, labelled, prepare_operand


from native_scan import CHUNK, MAX_WITNESS, PIN, parameters, geometry, check_headers, double_sha, synthetic_headers, witness_bytes
SCALE_ONE = b"\x80\x3f"
GENERATOR_CELLS = 1 << 22


def padded_plane(data):
    return data + bytes(-len(data) % CHUNK)


def scale_plane(rows, k):
    return SCALE_ONE * (rows * (k // 8))


def operand_digest(codes, key):
    # The native 1024-byte Merkle tree root equals keyed BLAKE3 over
    # the identically padded stream. Keep original rows file-backed.
    raw = memoryview(codes).cast("B")
    values = blake3(key=key)
    release = MappedPages(codes)
    for start in range(0, len(raw), 1 << 20):
        end = min(len(raw), start + (1 << 20))
        values.update(raw[start:end])
        release(end)
    values.update(bytes(-len(raw) % CHUNK))
    scale_size = len(raw)//4
    scales = blake3(key=key)
    ones = SCALE_ONE * (1 << 19)
    for start in range(0, scale_size, len(ones)):
        scales.update(ones[:min(len(ones), scale_size-start)])
    scales.update(bytes(-scale_size % CHUNK))
    return blake3(values.digest() + scales.digest(), key=key).digest()


def write_codes(path, seed, which, rows, k):
    """Two independent 7-bit draws per entry, as in the official CPU miner."""
    domain = b"pearl-5060/dense-random/v1\x00" + which + seed
    with path.open("xb") as out:
        for start in range(0, rows*k, GENERATOR_CELLS):
            count = min(rows*k - start, GENERATOR_CELLS)
            stream = domain if start == 0 else domain + struct.pack("<Q", start)
            raw = np.frombuffer(hashlib.shake_256(stream).digest(2*count), dtype=np.uint8).reshape(count, 2)
            codes = ((raw[:, 0] & 127).astype(np.int16) + (raw[:, 1] & 127) - 127).astype(np.int8)
            out.write(codes)


def headers(args):
    if args.job:
        external = json.loads(args.job.read_text(encoding="utf-8"))
        if external["cert_version"] != 4:
            raise ValueError("the job must explicitly advertise certificate version 4")
        header = bytes.fromhex(external["proposed_header_hex"])
        ancestors = [bytes.fromhex(x) for x in external["ancestor_headers_hex"]]
        source = "provided job"
    else:
        # A synthetic, hash-linked parent and candidate. No network lookup or submit.
        bits = int(args.bits, 16)
        header, ancestors = synthetic_headers(bits)
        source = "synthetic offline job"
    check_headers(header, ancestors)
    return header, ancestors, source


def read_meta(directory):
    meta = json.loads((directory / "job.json").read_text(encoding="utf-8"))
    if meta["pin"] != PIN:
        raise ValueError("job protocol revision mismatch")
    geometry(*[meta[x] for x in ("m", "n", "k", "h", "w")])
    check_headers(bytes.fromhex(meta["proposed_header_hex"]),
                  [bytes.fromhex(x) for x in meta["ancestor_headers_hex"]])
    return meta


def _stream_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(args, directory):
    h, w = map(int, args.tile.lower().split("x"))
    m, n, k = args.m, args.n, args.k
    geometry(m, n, k, h, w)
    header, ancestors, source = headers(args)
    if directory.exists():
        raise FileExistsError(f"choose an unused job directory: {directory}")
    seed = bytes.fromhex(args.seed) if args.seed else secrets.token_bytes(32)
    if len(seed) != 32:
        raise ValueError("seed must contain exactly 32 bytes")
    from lottery import difficulty_bound
    nbits = struct.unpack_from("<I", header, 72)[0]
    if not difficulty_bound(nbits, h, w, k):
        raise ValueError("job has no positive decoded target")
    keys = (labelled(b"key-A", header), labelled(b"key-B", ancestors[-1]))
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling directory and publish all the inputs together.
    with tempfile.TemporaryDirectory(prefix=".pearl-prep-", dir=directory.parent) as temporary:
        stage = Path(temporary) / "job"
        stage.mkdir()
        roots = []
        for name, rows, key in zip("AB", (m, n), keys, strict=True):
            path = stage / f"{name}.i8"
            write_codes(path, seed, name.encode("ascii"), rows, k)
            codes = np.memmap(path, mode="r", dtype=np.int8, shape=(rows, k))
            roots.append(operand_digest(codes, key))
            del codes
        ha, hb = roots
        pa, pb = parameters(m, n, k, h, w)
        seed_b = labelled(b"seed-B", hb + keys[1] + pb)
        seed_a = labelled(b"seed-A", ha + seed_b + keys[0] + pa)
        for name, rows, side, e_seed in (("A", m, 0, seed_a), ("B", n, 1, seed_b)):
            codes = np.memmap(stage/f"{name}.i8", mode="r", dtype=np.int8, shape=(rows, k))
            output = np.memmap(stage/f"{name}.e4m3", mode="w+", dtype=np.uint8, shape=(rows, k))
            last = time.monotonic()
            def progress(completed):
                nonlocal last
                now = time.monotonic()
                if completed == rows or now-last >= 20:
                    print(f"{name} FP8 rows {completed}/{rows}", file=sys.stderr, flush=True)
                    last = now
            prepare_operand(codes, side, e_seed, seed_b, out=output,
                            on_rows=progress if getattr(args, "progress", False) else None)
            output.flush()
            del codes, output
        np.arange(h*w, dtype="<u2").tofile(stage / "lanes.u16le")
        checksums = {name: _stream_sha(stage / name) for name in
                     ("A.i8", "B.i8", "A.e4m3", "B.e4m3", "lanes.u16le")}
        meta = dict(pin=PIN, source=source, cert_version=4, generator="shake256-triangular-4M",
                    m=m, n=n, k=k, h=h, w=w,
                    nbits=nbits, seed_hex=seed.hex(), proposed_header_hex=header.hex(),
                    ancestor_headers_hex=[v.hex() for v in ancestors], hash_a=ha.hex(), hash_b=hb.hex(),
                    seed_a=seed_a.hex(), seed_b=seed_b.hex(), sha256=checksums)
        (stage / "job.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        stage.rename(directory)
    return meta


def checked_inputs(directory, meta, names):
    for name in names:
        if _stream_sha(directory / name) != meta["sha256"][name]:
            raise ValueError(f"file no longer matches the prepared job: {name}")


def scan(directory, scanner=None, cpu=False, rows=None, cols=None, *, sparse=False, hit_capacity=256,
         worker=None, variant=None, resident=False, fuse_fold=False, defer_filter=False, row_pad=0, k_per_pass=None):
    from lottery import fold_cells
    with abort_on_error(worker), tempfile.TemporaryDirectory(prefix=".pearl-scan-", dir=directory) as temporary:
        directory=directory.resolve()
        meta = read_meta(directory)
        m, n, k, h, w = [meta[x] for x in ("m", "n", "k", "h", "w")]
        if type(hit_capacity) is not int or not 1 <= hit_capacity <= 65536:
            raise ValueError("hit capacity must be in 1..65536")
        if type(row_pad) is not int or not 0 <= row_pad <= 4096 or row_pad%16:
            raise ValueError("row padding must be a multiple of 16 in 0..4096")
        if row_pad and (variant or "").startswith("cutlass"):
            raise ValueError("CUTLASS variants require --row-pad 0")
        if k_per_pass is not None and (type(k_per_pass) is not int or not 0<=k_per_pass<=65536 or k_per_pass%128):
            raise ValueError("K pass size must be 0 or a multiple of 128 up to 65536")
        if cpu and (sparse or worker is not None or variant is not None or resident or fuse_fold or defer_filter or row_pad or k_per_pass is not None):
            raise ValueError("CUDA scanner options require a scanner")
        checked_inputs(directory, meta, ("A.e4m3", "B.e4m3", "lanes.u16le"))
        report=None
        name = "hits.phits" if sparse else "folds.bin"
        path = Path(temporary).resolve() / name
        if cpu:
            a = np.memmap(directory / "A.e4m3", dtype=np.uint8, mode="r", shape=(m, k))
            b = np.memmap(directory / "B.e4m3", dtype=np.uint8, mode="r", shape=(n, k))
            with path.open("xb") as f:
                for i in range(0, m-h+1, h):
                    for j in range(0, n-w+1, w):
                        acc = b200_mm(a[i:i+h], b[j:j+w])
                        f.write(fold_cells(acc.view(np.uint32).ravel(), h, w))
        else:
            if not scanner:
                raise ValueError("a built CUDA scanner or --cpu-reference is required")
            cmd = [str(scanner.resolve()), "--a", str(directory/"A.e4m3"), "--b", str(directory/"B.e4m3"),
                   "--m", str(m), "--n", str(n), "--k", str(k), "--tile", f"{h}x{w}",
                   "--lanes", str(directory/"lanes.u16le"), "--out", str(path)]
            if sparse:
                from sparse_tickets import make_params
                params = path.parent / "job.params"
                params.write_bytes(make_params(meta))
                cmd.extend(("--ticket-params", str(params), "--hit-capacity", str(hit_capacity)))
            if rows is not None:
                cmd.extend(("--rows-per-pass", str(rows)))
            if cols is not None:
                cmd.extend(("--cols-per-pass", str(cols)))
            if k_per_pass is not None:
                cmd.extend(("--k-per-pass",str(k_per_pass)))
            if variant is not None:
                cmd.extend(("--variant",variant))
            if resident:
                cmd.append("--resident")
            if fuse_fold:
                cmd.append("--fuse-fold")
            if defer_filter:
                cmd.append("--defer-filter")
            if row_pad:
                cmd.extend(("--row-pad",str(row_pad)))
            if worker is None:
                subprocess.run(cmd, check=True)
            else:
                report=worker.request(cmd[1:])
                check_file_scan(report,meta,sparse,resident,variant,row_pad)
        if sparse:
            from sparse_tickets import verified_hits
            count=sum(1 for _ in verified_hits(meta, path))
            if report is not None and count!=report["threshold_hits"]:
                raise ValueError("scanner acknowledged a different hit count than the checked sparse file")
        elif path.stat().st_size != (m//h)*(n//w)*64:
            raise ValueError("scanner returned the wrong number of 64-byte folds")
        os.replace(path, directory / name)
        if report is not None:
            print(json.dumps(report),flush=True)
    return meta


def ticket_records(directory, meta, folds_path):
    from lottery import jackpot_key, difficulty_bound, winning_tickets
    m, n, k, h, w = [meta[x] for x in ("m", "n", "k", "h", "w")]
    if Path(folds_path).suffix == ".phits":
        from sparse_tickets import verified_records
        for idx, folded, digest in verified_records(meta, folds_path):
            row, col = (idx//(n//w))*h, (idx % (n//w))*w
            if witness_bytes(m, n, k, h, w, row, col) <= MAX_WITNESS:
                yield idx, folded, digest
        return
    bound = difficulty_bound(meta["nbits"], h, w, k)
    key = jackpot_key(bytes.fromhex(meta["seed_a"]))
    expected = (m//h)*(n//w)
    if folds_path.stat().st_size != expected*64:
        raise ValueError("folds file size does not match this job")
    with folds_path.open("rb") as f:
        for start in range(0, expected, 4096):
            batch = f.read(min(4096, expected-start)*64)
            for local, digest in winning_tickets(batch, key, bound):
                idx = start + local
                row, col = (idx//(n//w))*h, (idx % (n//w))*w
                if witness_bytes(m, n, k, h, w, row, col) <= MAX_WITNESS:
                    yield idx, batch[64*local:64*(local+1)], digest


def ticket_results(directory, meta, folds_path):
    for idx, _fold, digest in ticket_records(directory, meta, folds_path):
        yield idx, digest


def import_verifier(native_dir):
    if native_dir:
        sys.path.insert(0, str(native_dir.resolve()))
    import pearl_mining as pm
    if not hasattr(pm, "BlockHeader") or not hasattr(pm, "verify_plain_proof_v4"):
        raise ValueError("load the native pearl_mining from the pinned certificate-v4 checkout")
    return pm


def native_operands(pm, directory, meta, keys):
    m, n, k = (meta[x] for x in ("m", "n", "k"))
    checked_inputs(directory, meta, ("A.i8", "B.i8"))
    trees = []
    for side, rows, key in zip("AB", (m, n), keys, strict=True):
        values = (directory / f"{side}.i8").read_bytes()
        if len(values) != rows * k:
            raise ValueError("prequant values size mismatch")
        parts = tuple(pm.MerkleTree(data=padded_plane(data), key=key, chunk_len=CHUNK)
                      for data in (values, scale_plane(rows, k)))
        digest = blake3(bytes(parts[0].root) + bytes(parts[1].root), key=key).hexdigest()
        if digest != meta[f"hash_{side.lower()}"]:
            raise ValueError("the official Merkle root differs from the prepared commitment")
        trees.append(parts)
    return trees


def open_rows(pm, tree, start, count, rows, row_bytes):
    indices = list(range(start, start+count))
    leaves = pm.MerkleTree.compute_leaf_indices_from_rows(indices, (rows, row_bytes), CHUNK)
    return pm.MatrixMerkleProof(tree.get_multileaf_proof(leaves), indices)


def check_proof_inputs(directory, meta):
    if meta.get("proof_inputs") == "candidate_rows":
        from native_job import check_raw_context
        check_raw_context(directory, meta)
    else:
        checked_inputs(directory, meta, ("A.e4m3", "B.e4m3"))


def pack_ticket(directory, meta, folds, tile_index, merkle_module, dest):
    """Export a candidate for a separately built Rust verifier on another host."""
    if dest.exists():
        raise FileExistsError(f"proof output already exists: {dest}")
    hit = next((record for record in ticket_records(directory, meta, folds) if record[0] == tile_index), None)
    if hit is None:
        raise ValueError("requested tile does not meet the job difficulty")
    check_proof_inputs(directory, meta)
    row, col = check_tile(directory, meta, tile_index, hit[2], expected_fold=hit[1])
    proposed = bytes.fromhex(meta["proposed_header_hex"])
    ancestors = [bytes.fromhex(v) for v in meta["ancestor_headers_hex"]]
    dims = [meta[x] for x in ("m", "n", "k", "h", "w")]
    m, n, k, h, w = dims
    p_a, p_b = parameters(*dims)
    keys = (labelled(b"key-A", proposed), labelled(b"key-B", ancestors[-1]))
    a, b = native_operands(merkle_module, directory, meta, keys)

    def u64(value):
        return struct.pack("<Q", value)

    def blob(value):
        return u64(len(value)) + value

    def integers(items):
        return u64(len(items)) + b"".join(u64(i) for i in items)

    def opening(tree, start, count, total, row_bytes):
        indices = list(range(start, start+count))
        leaves = merkle_module.MerkleTree.compute_leaf_indices_from_rows(indices, (total, row_bytes), CHUNK)
        proof = tree.get_multileaf_proof(leaves)
        return (u64(len(proof.leaf_data)) + b"".join(blob(bytes(leaf)) for leaf in proof.leaf_data)
                + integers(proof.leaf_indices) + u64(proof.total_leaves) + bytes(proof.root)
                + u64(len(proof.siblings)) + b"".join(bytes(s) for s in proof.siblings) + integers(indices))

    raw = (blob(ancestors[-1] + p_b + p_a) + blob(b"".join(ancestors[:-1]))
           + opening(a[0], row, h, m, k) + opening(b[0], col, w, n, k)
           + opening(a[1], row, h, m, k//4) + opening(b[1], col, w, n, k//4) + b"\x00")
    with dest.open("xb") as f:
        f.write(raw)
    return dict(status="candidate_exported", tile=tile_index, bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), digest_le=hit[2].hex())


def check_tile(directory, meta, index, digest, *, expected_fold=None):
    from lottery import fold_cells, jackpot_key, ticket_digest
    m, n, k, h, w = [meta[x] for x in ("m", "n", "k", "h", "w")]
    row, col = (index // (n//w))*h, (index % (n//w))*w
    if meta.get("proof_inputs") == "candidate_rows":
        from native_job import candidate_panels
        a, b = candidate_panels(directory, meta, row, col)
    else:
        a = np.memmap(directory/"A.e4m3", dtype=np.uint8, mode="r", shape=(m, k))[row:row+h]
        b = np.memmap(directory/"B.e4m3", dtype=np.uint8, mode="r", shape=(n, k))[col:col+w]
    c = b200_mm(a, b)
    replay = fold_cells(c.view(np.uint32).ravel(), h, w)
    if expected_fold is not None and replay != expected_fold:
        raise ValueError("claimed 64-byte fold differs from CPU recomputation of this tile")
    if ticket_digest(jackpot_key(bytes.fromhex(meta["seed_a"])), replay) != digest:
        raise ValueError("scanner hit differs from CPU recomputation of this tile")
    return row, col


def finish(directory, folds_path, native_dir):
    meta = read_meta(directory)
    m, n, k, h, w = [meta[x] for x in ("m", "n", "k", "h", "w")]
    tickets = iter(ticket_records(directory, meta, folds_path))
    first = next(tickets, None)
    if first is None:
        return {"status": "no_difficulty_hit", "tickets": (m//h)*(n//w), "hits_checked": 0}
    import itertools
    pm = import_verifier(native_dir)
    check_proof_inputs(directory, meta)
    header_bytes = bytes.fromhex(meta["proposed_header_hex"])
    header = pm.IncompleteBlockHeader.from_bytes(header_bytes)
    ancestor_bytes = [bytes.fromhex(v) for v in meta["ancestor_headers_hex"]]
    ancestor = pm.BlockHeader.from_bytes(ancestor_bytes[-1])
    middle = [pm.BlockHeader.from_bytes(v) for v in ancestor_bytes[:-1]]
    keys = (labelled(b"key-A", header_bytes), labelled(b"key-B", ancestor_bytes[-1]))
    ta, tb = native_operands(pm, directory, meta, keys)
    hash_id = pm.HashId.Blake3Chunk1024
    hits = 0
    for index, folded, digest in itertools.chain((first,), tickets):
        hits += 1
        row, col = (index // (n//w))*h, (index % (n//w))*w
        proof = pm.PlainProofV4(
            ancestor_header=ancestor, common=pm.CommonParams(k, 32, pm.Quant.Fp8E4M3Prequant, pm.Device.B200),
            a=pm.OperandParams(m, hash_id, h), b=pm.OperandParams(n, hash_id, w),
            values_a=open_rows(pm, ta[0], row, h, m, k), values_b=open_rows(pm, tb[0], col, w, n, k),
            scales_a=open_rows(pm, ta[1], row, h, m, k//4), scales_b=open_rows(pm, tb[1], col, w, n, k//4),
            moe=None, moe_witness=None, ancestor_chain=middle)
        ok, message = pm.verify_plain_proof_for_cert_version(4, header, proof)
        if not ok:
            if str(message) == "The jackpot is not admissible":
                continue
            raise ValueError(f"upstream verification rejected this candidate: {message}")
        check_tile(directory, meta, index, digest, expected_fold=folded)
        encoded = proof.to_base64()
        raw = base64.b64decode(encoded, validate=True)
        decoded = pm.PlainProofV4.from_base64(encoded)
        valid, replay = pm.verify_plain_proof_for_cert_version(4, header, decoded)
        if not valid:
            raise ValueError(f"serialized proof failed an independent replay: {replay}")
        name = f"tile-{index}.plain.bin"
        with tempfile.TemporaryDirectory(prefix=".pearl-proof-", dir=directory) as temporary:
            path = Path(temporary) / name
            path.write_bytes(raw)
            os.replace(path, directory/name)
        return dict(status="plain_verified", tickets=(m//h)*(n//w), hits_checked=hits, tile=index,
                    row=row, col=col, target_digest_le=digest.hex(), plain=name, proof_bytes=len(raw),
                    proof_sha256=hashlib.sha256(raw).hexdigest(), native_module_sha256=_stream_sha(Path(pm.__file__)))
    return dict(status="no_admissible_hit", tickets=(m//h)*(n//w), hits_checked=hits)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "scan", "finish", "mine", "tickets", "pack", "replay"):
        s = sub.add_parser(command)
        s.add_argument("directory", type=Path)
        if command in ("mine", "prepare"):
            s.add_argument("--job", type=Path, help="local JSON with a proposed header and complete ancestors")
            s.add_argument("--bits", default="1e0fffff", help="compact target for synthetic jobs only")
            s.add_argument("--seed", help="32 bytes of hex for repeatable tests, otherwise freshly generated")
            add_geometry_options(s)
            s.add_argument("--progress", action="store_true", help="report rows processed about every 20 seconds")
        if command in ("mine", "scan"):
            g = s.add_mutually_exclusive_group(required=True)
            g.add_argument("--scanner", type=Path)
            g.add_argument("--cpu-reference", action="store_true")
            add_scan_options(s)
        if command in ("mine", "finish", "pack", "replay"):
            s.add_argument("--native-dir", type=Path, help="directory containing pinned self-built pearl_mining")
        if command in ("finish", "tickets", "pack"):
            s.add_argument("--folds", type=Path, help="GPU output from another machine")
        if command == "pack":
            s.add_argument("--tile-index", type=int, required=True)
            s.add_argument("--out", type=Path, required=True)
        if command == "replay":
            s.add_argument("--proof", type=Path, required=True)
        if command == "mine":
            s.add_argument("--attempts", type=int, default=8, help="maximum fresh operands per job")
            add_native_options(s)
    return p.parse_args()


def main():
    args = arguments()
    with scanner_session(args) as worker:
        return execute(args,worker)


def execute(args,worker=None):
    validate_stream_request(args)
    directory = args.directory.resolve()
    if args.command == "mine":
        if args.attempts < 1:
            raise ValueError("attempts must be positive")
        native = getattr(args,"native_dense",False)
        if native and (not args.scanner or args.cpu_reference):
            raise ValueError("--native-dense requires --scanner")
        if getattr(args,"native_full_reference",False) and not native:
            raise ValueError("--native-full-reference requires --native-dense")
        if getattr(args,"gpu_prover",None) is not None and not native:
            raise ValueError("--gpu-prover requires --native-dense")
        if args.seed:
            base_seed = bytes.fromhex(args.seed)
            if len(base_seed) != 32:
                raise ValueError("seed must contain exactly 32 bytes")
        else:
            base_seed = secrets.token_bytes(32)
        directory.mkdir(parents=True, exist_ok=False)
        for i in range(args.attempts):
            args.seed = hashlib.sha256(base_seed + i.to_bytes(8, "little")).hexdigest()
            candidate = directory / f"attempt-{i:04d}"
            if native:
                result = native_attempt(args, args, candidate, worker)
            else:
                prepare(args, candidate)
                scan(candidate, args.scanner, args.cpu_reference, args.rows_per_pass, args.cols_per_pass,
                     **scan_options(args,worker))
                name = "hits.phits" if args.sparse else "folds.bin"
                result = finish(candidate, candidate/name, args.native_dir)
            print(json.dumps(dict(attempt=i, **result)), flush=True)
            if result["status"] == "plain_verified":
                return 0
        return 2
    if args.command == "prepare":
        meta = prepare(args, directory)
        result = {k: meta[k] for k in ("pin", "source", "m", "n", "k", "h", "w", "sha256")}
    elif args.command == "scan":
        scan(directory, args.scanner, args.cpu_reference, args.rows_per_pass, args.cols_per_pass,
             **scan_options(args,worker))
        name = "hits.phits" if args.sparse else "folds.bin"
        result = {"hits" if args.sparse else "folds": name, "sha256": _stream_sha(directory/name)}
    elif args.command == "tickets":
        meta = read_meta(directory)
        hits = ticket_results(directory, meta, args.folds or directory/"folds.bin")
        result = {"status": "difficulty_hits_only", "hits": [{"tile": i, "digest_le": v.hex()} for i, v in hits]}
    elif args.command == "pack":
        if args.native_dir:
            sys.path.insert(0, str(args.native_dir.resolve()))
        import pearl_mining
        result = pack_ticket(directory, read_meta(directory), args.folds or directory/"folds.bin",
                             args.tile_index, pearl_mining, args.out)
    elif args.command == "replay":
        pm = import_verifier(args.native_dir)
        meta = read_meta(directory)
        blob = args.proof.read_bytes()
        proof = pm.PlainProofV4.from_base64(base64.b64encode(blob).decode())
        header = pm.IncompleteBlockHeader.from_bytes(bytes.fromhex(meta["proposed_header_hex"]))
        valid, detail = pm.verify_plain_proof_for_cert_version(4, header, proof)
        if not valid:
            raise ValueError(f"official PlainProofV4 verification failed: {detail}")
        result = dict(status="plain_verified", sha256=hashlib.sha256(blob).hexdigest(),
                      native_module_sha256=_stream_sha(Path(pm.__file__)))
    else:
        result = finish(directory, args.folds or directory/"folds.bin", args.native_dir)
    print(json.dumps(result, indent=2))
    return 0


def native_attempt(prep, args, directory, worker=None):
    from native_job import attempt
    return attempt(prep, args, directory, worker)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
