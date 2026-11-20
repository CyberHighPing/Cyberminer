"""Pearl V4 ticket extraction and target filter (upstream d797c307).

Records contain sixteen little-endian uint32 lanes per lottery tile, in tile
order. A passing digest still requires the opened proof's policy and verifier.
"""

from __future__ import annotations

import operator
import struct
from collections.abc import Iterator, Sequence

try:
    from blake3 import blake3 as _native_blake3
except ImportError:
    _native_blake3 = None


_U32 = (1 << 32) - 1
_U256 = (1 << 256) - 1
_LABEL = b"pearl/v4/FP8/jackpot"
CONTROL_REVISION = "d797c3070a17e1d0fc1b30346a073a7de2b0eb23"
_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_KEY_WORDS = struct.Struct("<8I")
_RECORD = struct.Struct("<16I")


def _right(value, count):
    return ((value << (32 - count)) & _U32) | (value >> count)


def _mix(state, a, b, c, d, x, y):
    state[a] = (state[a] + state[b] + x) & _U32
    state[d] = _right(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _U32
    state[b] = _right(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + y) & _U32
    state[d] = _right(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _U32
    state[b] = _right(state[b] ^ state[c], 7)


def _compress(key_words, words, length, flags):
    state = [*key_words, *_IV[:4], 0, 0, length, flags]
    for round_index in range(7):
        _mix(state, 0, 4, 8, 12, words[0], words[1])
        _mix(state, 1, 5, 9, 13, words[2], words[3])
        _mix(state, 2, 6, 10, 14, words[4], words[5])
        _mix(state, 3, 7, 11, 15, words[6], words[7])
        _mix(state, 0, 5, 10, 15, words[8], words[9])
        _mix(state, 1, 6, 11, 12, words[10], words[11])
        _mix(state, 2, 7, 8, 13, words[12], words[13])
        _mix(state, 3, 4, 9, 14, words[14], words[15])
        if round_index < 6:
            words = [words[i] for i in _PERM]
    return [state[i] ^ state[i + 8] for i in range(8)]


def small_digest(data: bytes, key: bytes | None = None) -> bytes:
    """Hash labels, headers and ticket records. All fit in one BLAKE3 chunk."""
    if key is not None and len(key) != 32:
        raise ValueError("BLAKE3 keys must be 32 bytes")
    if _native_blake3 is not None:
        return (_native_blake3(data) if key is None else _native_blake3(data, key=key)).digest()
    if len(data) > 1024:
        raise ValueError("install blake3 to hash messages larger than one chunk")
    cv = _IV if key is None else _KEY_WORDS.unpack(key)
    keyed = 0 if key is None else 16
    for offset in range(0, max(1, len(data)), 64):
        block = data[offset:offset+64]
        flags = keyed | (1 if offset == 0 else 0)  # CHUNK_START
        if offset + 64 >= len(data):
            flags |= 2 | 8  # CHUNK_END | ROOT
        cv = _compress(cv, _RECORD.unpack(block.ljust(64, b"\x00")), len(block), flags)
    return _KEY_WORDS.pack(*cv)


def labelled(label, message, parent=None):
    return small_digest(message, small_digest(b"pearl/v4/FP8/" + label, parent))


def jackpot_key(seed_a: bytes) -> bytes:
    """Derive the jackpot subkey from the job's A-side noise seed."""
    return small_digest(_LABEL, seed_a)


def ticket_digest(key: bytes, folded: bytes) -> bytes:
    """Digest an already-folded tile with the already-derived jackpot key."""
    if len(folded) != 64:
        raise ValueError("one folded lottery tile is exactly 64 bytes")
    return small_digest(folded, key)


def _check_tile(h: int, w: int) -> None:
    if h < 4 or w < 16 or not 256 <= h * w <= 2048 or (h * w) % 16:
        raise ValueError("tile is outside the V4 lottery layout")


def fold_cells(cell_bits: Sequence[int], h: int, w: int) -> bytes:
    """Fold raw FP32 bit patterns in compact tile row-major order."""
    _check_tile(h, w)
    if len(cell_bits) != h * w:
        raise ValueError("wrong number of accumulator cells for this tile")
    span = h * w // 16
    lanes = []
    for start in range(0, h * w, span):
        lane = 0
        for i in range(start, start + span):
            word = operator.index(cell_bits[i])
            if not 0 <= word <= _U32:
                raise ValueError("accumulator bits must be an unsigned 32-bit word")
            mixed = (lane * 0x9E3779B1 + word) & _U32
            lane = ((mixed << 13) & _U32) | (mixed >> 19)
        lanes.append(lane)
    return _RECORD.pack(*lanes)


def difficulty_bound(nbits: int, h: int, w: int, k: int) -> int:
    """Decode compact nBits and scale by the committed lottery work."""
    _check_tile(h, w)
    if not (0 <= nbits <= _U32 and 0 < k <= _U32):
        raise ValueError("nBits and positive k must fit uint32")
    exponent, mantissa = nbits >> 24, nbits & 0xFFFFFF
    if not exponent or (mantissa & 0x800000):
        return 0
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = (mantissa << (8 * (exponent - 3))) & _U256
    # Rust saturates the u32 work adjustment and then the U256 product.
    return min(_U256, target * min(_U32, h * w * k))


def meets_bound(digest: bytes, bound: int) -> bool:
    """Pearl compares the entire little-endian digest, including equality."""
    if len(digest) != 32 or not 0 <= bound <= _U256:
        raise ValueError("digest and bound must be 256 bits")
    return int.from_bytes(digest, "little") <= bound


def winning_tickets(folded_records: bytes, key: bytes, bound: int) -> Iterator[tuple[int, bytes]]:
    """Yield (tile index, digest) for candidates, in input order."""
    data = memoryview(folded_records).cast("B")
    if len(key) != 32 or data.nbytes % 64 or not 0 <= bound <= _U256:
        raise ValueError("invalid jackpot key, incomplete tile record, or bound")
    if _native_blake3 is None and data.nbytes >= 64 * 64:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            # Bound temporary storage even when the caller passes a whole device run.
            key_words = [np.array([v], dtype=np.uint32) for v in _KEY_WORDS.unpack(key)]
            limit = [(bound >> (32 * i)) & _U32 for i in range(8)]
            for start in range(0, data.nbytes, 4096 * 64):
                batch = data[start : start + 4096 * 64]
                words = np.frombuffer(batch, dtype="<u4").reshape(-1, 16).T
                result = _compress(key_words, words, 64, 1 | 2 | 8 | 16)
                # Build lexicographic <= starting at the low word. A higher
                # unequal word overrides every lower word.
                accepted = result[0] <= limit[0]
                for i in range(1, 8):
                    accepted = (result[i] < limit[i]) | ((result[i] == limit[i]) & accepted)
                for index in np.flatnonzero(accepted):
                    digest = _KEY_WORDS.pack(*(int(v[index]) for v in result))
                    yield start // 64 + int(index), digest
            return
    for offset in range(0, data.nbytes, 64):
        digest = ticket_digest(key, bytes(data[offset : offset + 64]))
        if meets_bound(digest, bound):
            yield offset // 64, digest
