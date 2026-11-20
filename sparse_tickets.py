"""Pearl V4 native ticket filter parameters and checked, sparse scanner output."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from lottery import CONTROL_REVISION as PIN, difficulty_bound, jackpot_key, meets_bound, ticket_digest


MAGIC = b"PRLFP8S1"
HEADER = struct.Struct("<8s8I3Q32s32s32s")
RECORD = struct.Struct("<Q64s32s")
COUNTS = struct.Struct("<QQ")


def _hex(value, length, name):
    if not isinstance(value, str) or len(value) != 2*length:
        raise ValueError(f"invalid {name}")
    data = bytes.fromhex(value)
    if len(data) != length:
        raise ValueError(f"invalid {name}")
    return data


def make_params(meta):
    if meta.get("pin") != PIN or meta.get("cert_version") != 4:
        raise ValueError("expected the supported FP8 certificate-v4 protocol revision")
    dims = tuple(meta[x] for x in ("m","n","k","h","w"))
    if any(type(x) is not int or not 1 <= x <= 0xFFFFFFFF for x in dims):
        raise ValueError("invalid task dimensions")
    m,n,k,h,w = dims
    if m < h or n < w or not 1024 <= k <= 65536 or k % 32:
        raise ValueError("invalid complete-tile geometry")
    proposed = _hex(meta["proposed_header_hex"],76,"proposed header")
    nbits = struct.unpack_from("<I", proposed, 72)[0]
    if nbits != meta.get("nbits"):
        raise ValueError("nBits differs from the job header")
    ancestors = meta["ancestor_headers_hex"]
    if not isinstance(ancestors, list) or not 1 <= len(ancestors) <= 4:
        raise ValueError("expected one to four complete ancestor headers")
    ancestors = [_hex(v,108,"ancestor header") for v in ancestors]
    prev = proposed[4:36]
    for ancestor in ancestors:
        if hashlib.sha256(hashlib.sha256(ancestor).digest()).digest() != prev:
            raise ValueError("ancestor chain does not connect to the proposed header")
        prev = ancestor[4:36]
    ha, hb, sa, sb = (_hex(meta[x],32,x) for x in ("hash_a","hash_b","seed_a","seed_b"))
    key = jackpot_key(sa)
    bound = difficulty_bound(nbits,h,w,k)
    if not bound:
        raise ValueError("job has a zero difficulty bound")
    tag = hashlib.sha256(b"pearl/v4/sparse-job/v1\0" + bytes.fromhex(PIN)
                         + struct.pack("<I5I",4,*dims) + proposed
                         + bytes([len(ancestors)]) + b"".join(ancestors) + ha + hb + sa + sb).digest()
    total = (m//h)*(n//w)
    return HEADER.pack(MAGIC,1,HEADER.size,RECORD.size,*dims,total,0,0,key,bound.to_bytes(32,"little"),tag)


def _remember_factory(total, claimed):
    # Typical jobs keep a few hundred candidates. For a dense result, the bitset
    # avoids allocating one Python integer object per candidate.
    size = (total+7)//8
    if size <= 64 << 20 and size < 40*claimed:
        seen = bytearray(size)
        def remember(i):
            p, bit = divmod(i,8)
            mask = 1 << bit
            fresh = not (seen[p] & mask)
            seen[p] |= mask
            return fresh
    else:
        seen = set()
        def remember(i):
            if i in seen:
                return False
            seen.add(i)
            return True
    return remember


def verified_records(meta, path):
    """Yield (global tile, original fold, digest) after the full file passes."""
    expected = make_params(meta)
    total = struct.unpack_from("<Q",expected,40)[0]
    key, bound = expected[64:96], int.from_bytes(expected[96:128],"little")
    with Path(path).open("rb") as src, tempfile.SpooledTemporaryFile(max_size=1<<20) as verified:
        head = src.read(HEADER.size)
        if len(head) != HEADER.size or head[:48] != expected[:48] or head[64:] != expected[64:]:
            raise ValueError("sparse header is truncated or bound to a different job")
        scanned, claimed = COUNTS.unpack_from(head,48)
        if scanned != total or claimed > total:
            raise ValueError("scanner did not cover exactly all complete job tiles")
        if os.fstat(src.fileno()).st_size != HEADER.size + RECORD.size*claimed:
            raise ValueError("sparse file is truncated, has extra bytes, or has the wrong hit count")
        remember = _remember_factory(total,claimed)
        remaining = claimed
        while remaining:
            take = min(4096, remaining)
            records = src.read(take*RECORD.size)
            if len(records) != take*RECORD.size:
                raise ValueError("truncated sparse record")
            for tile, folded, claimed_digest in RECORD.iter_unpack(records):
                if tile >= total or not remember(tile):
                    raise ValueError(f"tile {tile} is out of range or repeated")
                digest = ticket_digest(key,folded)
                if claimed_digest != digest:
                    raise ValueError(f"ticket digest mismatch at tile {tile}")
                if not meets_bound(digest,bound):
                    raise ValueError(f"tile {tile} misses the job difficulty")
                verified.write(RECORD.pack(tile,folded,digest))
            remaining -= take
        if src.read(1):
            raise ValueError("unexpected data after the last sparse hit")
        verified.seek(0)
        while chunk := verified.read(4096*RECORD.size):
            yield from RECORD.iter_unpack(chunk)


def verified_hits(meta, path):
    """Yield the same (global tile, digest) values as a checked dense scan."""
    for tile, _fold, digest in verified_records(meta,path):
        yield tile,digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("params","check"):
        cmd = sub.add_parser(mode)
        cmd.add_argument("job",type=Path,help="trusted job directory or its job.json")
        cmd.add_argument("file",type=Path)
        if mode == "check":
            cmd.add_argument("--list",action="store_true")
    args = parser.parse_args()
    source = args.job/"job.json" if args.job.is_dir() else args.job
    meta = json.loads(source.read_text(encoding="utf-8"))
    if args.mode == "params":
        data = make_params(meta)
        with args.file.open("xb") as target:
            target.write(data)
        print(json.dumps({"bytes":len(data),"tag":data[128:].hex(),
                          "complete_tiles":struct.unpack_from("<Q",data,40)[0]}))
    else:
        count = 0
        for tile,digest in verified_hits(meta,args.file):
            count += 1
            if args.list:
                print(json.dumps({"tile":tile,"digest_le":digest.hex()}))
        print(json.dumps({"verified_difficulty_hits":count}))


if __name__ == "__main__":
    try:
        main()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit(f"error: {exc}")
