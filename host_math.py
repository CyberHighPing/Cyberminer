"""CPU reference for dense rank-32 Blackwell FP8 inputs. Only NumPy and BLAKE3."""

import math
import struct

import numpy as np
from blake3 import blake3
from lottery import labelled
from fast_prepare import MappedPages, Rank32Product, encode_f32


def bf16_bits(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def bf16(x):
    return from_bf16(bf16_bits(x))


def decode_fp8(raw):
    raw = np.asarray(raw, dtype=np.uint8)
    exp = ((raw >> 3) & 15).astype(np.int16)
    mant = (raw & 7).astype(np.float64)
    magnitude = np.ldexp(np.where(exp == 0, mant, mant + 8), np.maximum(exp, 1) - 10)
    return np.where(raw & 128, -magnitude, magnitude)


_LEVELS = decode_fp8(np.arange(127, dtype=np.uint8))
_MID = (_LEVELS[:-1] + _LEVELS[1:]) * 0.5


def encode_fp8(x):
    x = np.asarray(x)
    if x.dtype == np.float32:
        return encode_f32(x)
    if not np.isfinite(x).all():
        raise ValueError("non-finite E4M3 input")
    mag = np.abs(x)
    code = np.searchsorted(_MID, mag, side="left")
    ties_up = (code < 126) & (mag == _MID[np.minimum(code, 125)]) & ((code & 1) != 0)
    return ((code + ties_up) | (np.signbit(x).astype(np.int32) << 7)).astype(np.uint8)


def noise_lines(seed, side, factor, indices):
    key = blake3(b"pearl/v4/FP8/noise-line", key=seed).digest()
    address = struct.Struct("<BBI58x")
    raw = b"".join(blake3(address.pack(side, factor, i), key=key).digest() for i in indices)
    u = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 32).astype(np.int32)
    x = (1 - 2 * (u >> 7)) * ((u & 127) + 1)
    sums = np.sum(x * x, axis=1, dtype=np.int64)
    denom = bf16(np.fromiter((math.isqrt(int(v) * 1024) for v in sums), dtype=np.int64))
    gain = bf16(np.float32(8192) / denom)
    return encode_fp8(bf16(x * gain[:, None]))


def _fp8_stored_exponents(raw):
    exp = np.maximum(((raw >> 3) & 15).astype(np.int16), 1) - 7
    return np.where((raw & 127) == 0, -512, exp)


def _atom(a, ae, b, be, carry):
    # Every operation preceding the final FP32 cast is exact in binary64:
    # each aligned magnitude has at most 27 bits, and there are 33 terms.
    product = a[:, None, :] * b[None, :, :]
    term_exp = ae[:, None, :] + be[None, :, :]
    carry_exp = np.where(carry == 0, -512, np.frexp(carry)[1] - 1)
    anchor = np.maximum(term_exp.max(axis=2), carry_exp)
    grid = np.maximum(anchor, -126) - 25
    aligned = np.trunc(np.ldexp(product, -grid[:, :, None])).sum(axis=2)
    aligned += np.trunc(np.ldexp(carry.astype(np.float64), -grid))
    exact = np.ldexp(aligned, grid)
    nearest = exact.astype(np.float32)
    return np.where(np.abs(nearest) > np.abs(exact),
                    np.nextafter(nearest, np.float32(0)), nearest)


def b200_mm(a, b):
    """Correctly ordered E4M3 B200 MMA with 32 products per instruction."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] % 32:
        raise ValueError("A[M,K] and B[N,K] need the same 32-aligned K")
    if np.any((a & 127) == 127) or np.any((b & 127) == 127):
        raise ValueError("NaN is not a valid mining input")
    av, bv = decode_fp8(a), decode_fp8(b)
    ae, be = _fp8_stored_exponents(a), _fp8_stored_exponents(b)
    out = np.empty((len(a), len(b)), dtype=np.float32)
    for i in range(0, len(a), 16):
        for j in range(0, len(b), 256):
            row = slice(i, min(i + 16, len(a)))
            col = slice(j, min(j + 256, len(b)))
            carry = np.zeros(out[row, col].shape, dtype=np.float32)
            for k in range(0, a.shape[1], 32):
                atom = slice(k, k + 32)
                carry = _atom(av[row, atom], ae[row, atom], bv[col, atom], be[col, atom], carry)
            out[row, col] = carry
    return out


def unit_scale_norms(codes):
    """Exact RNE BF16 square root for the committed unit-scale int8 rows."""
    sq = np.square(codes.astype(np.int64)).sum(axis=1)
    k = codes.shape[1]
    root = np.sqrt(sq.astype(np.float64) / k)
    lower = (root.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    # A rounded FP32 intermediate can have crossed a BF16 representable value.
    low = from_bf16(lower).astype(np.float64)
    lower -= (low * low * k > sq).astype(np.uint16)
    mid = (from_bf16(lower).astype(np.float64) + from_bf16(lower + 1)) * 0.5
    square = mid * mid * k
    rounded = lower + ((sq > square) | ((sq == square) & ((lower & 1) == 1)))
    rms = from_bf16((rounded + 2) & np.uint16(0xFFFC))
    peak = np.max(np.abs(codes.astype(np.int16)), axis=1).astype(np.float32)
    return rms, peak


def row_parameters(rms, peak):
    """Committed unit BF16 scales, rank 32, Blackwell delta = 1/2."""
    rms = np.maximum(rms, np.float32(2**-32))
    peak = np.maximum(peak, np.float32(2**-32))
    delta_r = bf16(math.sqrt(32.0) * 0.5)
    delta_std = bf16(math.sqrt(32.0) * 0.5 / 65536.0)
    alpha = bf16(np.float32(448) / bf16(delta_r.astype(np.float64) * rms + peak))
    beta = bf16(bf16(alpha * rms) * delta_std)
    scaled_rms = alpha * rms
    if np.any(scaled_rms < 4):
        raise ValueError("input row below the protocol's scaled RMS floor")
    high = (np.frexp(scaled_rms)[1] - 1 - 9 + 7).astype(np.int16)
    strip_exp = ((high << 3) & 0x78).astype(np.uint8)
    g = np.where(scaled_rms < 8, np.float32(0.1), np.float32(0.08))
    return alpha, beta, strip_exp, encode_fp8(g * scaled_rms)


def mix_codes(codes, noise, rms, peak):
    """Independent readable reference for the fused rank-32 input preparation."""
    alpha, beta, strip_exp, threshold = row_parameters(rms, peak)
    signal = alpha[:, None] * codes.astype(np.float32)
    # Unit scales plus int8 codes keep the signal < 16 exact significant bits.
    # beta is BF16 and the bounded 32-term noise has <=24 significant bits.
    mixed = (noise.astype(np.float64) * beta[:, None] + signal).astype(np.float32)
    raw = encode_fp8(mixed)
    safe = (raw & 0x80) | strip_exp[:, None] | ((raw & 7) | 1)
    below = (raw & 127) < threshold[:, None]
    coin = ((mixed[:, ::4].copy().view(np.uint32) >> 8) & 127) < 112
    return np.where(below & np.repeat(coin, 4, axis=1), safe, raw), alpha, beta


def prepare_operand(codes, side, e_seed, b_seed, out=None, on_rows=None):
    k = codes.shape[1]
    product = Rank32Product(noise_lines(b_seed, side, 1, range(k)))
    if out is None:
        out = np.empty(codes.shape, dtype=np.uint8)
    release = MappedPages(codes, out)
    for start in range(0, len(codes), 32):
        stop = min(start + 32, len(codes))
        batch = codes[start:stop]
        e = noise_lines(e_seed, side, 0, range(start, stop))
        params = row_parameters(*unit_scale_norms(batch))
        product.write_mixed(e, batch, *params, out[start:stop])
        release(stop*k)
        if on_rows is not None:
            on_rows(stop)
    return out
