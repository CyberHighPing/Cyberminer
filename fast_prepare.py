"""Memory-bounded, bit-exact noise MMA for a single rank-32 Blackwell group."""

import ctypes
import errno
import mmap
import os
import platform
import sys
from pathlib import Path

import numpy as np


def _load_engine():
    # An architecture suffix keeps a package shared between ARM and x86 usable.
    ext = "dll" if os.name == "nt" else "so"
    root = Path(__file__).resolve().parent
    name = platform.machine().lower()
    name = {"amd64": "x86_64", "arm64": "aarch64"}.get(name, name)
    for path in (root / "build" / f"fast_prepare.{name}.{ext}",
                 root / "build" / f"fast_prepare.native.{ext}"):
        if not path.is_file():
            continue
        try:
            library = ctypes.CDLL(str(path))
            engine = library.pearl_rank32_product
        except (OSError, AttributeError):
            continue
        ptr = ctypes.c_void_p
        engine.argtypes = (ptr, ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_size_t,
                           ptr, ctypes.c_int, ctypes.c_int)
        engine.restype = None
        mix = getattr(library, "pearl_rank32_quantize", None)
        encode = getattr(library, "pearl_encode_f32", None)
        if mix is not None:
            mix.argtypes = (ptr,) * 9 + (ctypes.c_size_t, ctypes.c_size_t, ptr, ctypes.c_int, ctypes.c_int)
            mix.restype = ctypes.c_int
        if encode is not None:
            encode.argtypes = (ptr, ptr, ctypes.c_size_t)
            encode.restype = ctypes.c_int
        return engine, mix, encode
    return None, None, None


_ENGINE, _MIX, _ENCODE = _load_engine()
_ABS = np.arange(128, dtype=np.int32)
_FIELD = _ABS >> 3
_EXP = np.maximum(_FIELD, 1) - 7
_MAG = np.where(_FIELD == 0, _ABS & 7, 8 + (_ABS & 7)) << (_EXP + 6)
_SIGNED = np.concatenate((_MAG, -_MAG)).astype(np.int32)
_EXP[0] = -64
_EXP = np.tile(_EXP, 2).astype(np.int8)


class MappedPages:
    """Release consumed shared Linux file pages in bounded batches during sequential scans."""
    def __init__(self, *arrays):
        self._maps = []
        self._last = 0
        self._size = 0
        if sys.platform != "linux" or not hasattr(mmap.mmap, "madvise"):
            return
        for arr in arrays:
            if not isinstance(arr, np.memmap) or arr.mode not in ("r", "r+", "w+") or not arr.flags.c_contiguous:
                continue
            root = arr
            while isinstance(root.base, np.memmap):
                root = root.base
            mapping = root.base
            if not isinstance(mapping, mmap.mmap):
                continue
            offset = root.offset % mmap.ALLOCATIONGRANULARITY + arr.ctypes.data - root.ctypes.data
            if self._maps and arr.nbytes != self._size:
                raise ValueError("tracked maps must share their byte length")
            self._size = arr.nbytes
            self._maps.append((mapping, offset, arr.mode != "r"))

    def __call__(self, consumed):
        if not self._maps or (consumed - self._last < 32 << 20 and consumed != self._size):
            return
        if not self._last <= consumed <= self._size:
            raise ValueError("mapped file consumption must advance within the array")
        page = mmap.PAGESIZE
        for mapping, offset, writable in self._maps:
            # Keep a partially consumed page live across calls. For an unaligned
            # mapped view, also keep the initial page shared with earlier bytes.
            initial = ((offset + page - 1) // page) * page
            begin = max(initial, ((offset + self._last) // page) * page)
            end = ((offset + consumed) // page) * page
            if end > begin:
                if writable:
                    mapping.flush(begin, end - begin)
                try:
                    mapping.madvise(mmap.MADV_DONTNEED, begin, end - begin)
                except OSError as exc:
                    if exc.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                        raise
        self._last = consumed


def encode_f32(values):
    """Round finite FP32 to E4M3FN, ties-to-even, saturating at magnitude 448."""
    values = np.asarray(values)
    if values.dtype != np.float32:
        raise ValueError("FP32 values required")
    if _ENCODE is not None:
        output = np.empty(values.shape, dtype=np.uint8)
        contiguous = np.ascontiguousarray(values)
        if not _ENCODE(contiguous.ctypes.data, output.ctypes.data, values.size):
            raise ValueError("non-finite E4M3 input")
        return output
    bits = values.view(np.uint32)
    mag = bits & np.uint32(0x7fffffff)
    if np.any(mag >= np.uint32(0x7f800000)):
        raise ValueError("non-finite E4M3 input")
    # The normal values retain three fraction bits. Add half an ulp minus
    # one plus the retained tie bit before shifting. FP32/E4M3 biases differ by 120.
    rounded = mag + np.uint32(0x7ffff) + ((mag >> 20) & np.uint32(1))
    normal = np.minimum((rounded >> 20).astype(np.int32) - 960, 126)
    # Below 2^-6, E4M3 uses a uniform 2^-9 grid. The FP32 ulp at 2^14
    # is exactly 2^-9; RNE addition yields that small integer in its low bits.
    small = (mag.view(np.float32) + np.float32(16384)).view(np.uint32) - np.uint32(0x46800000)
    encoded = np.where(mag < np.uint32(0x3c800000), small, normal).astype(np.uint8)
    return encoded | ((bits >> 24).astype(np.uint8) & np.uint8(0x80))


class _Factor:
    def __init__(self, codes):
        codes = np.asarray(codes)
        if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != 32:
            raise ValueError("factor must be a [rows,32] uint8 E4M3 plane")
        if np.any((codes & 127) == 127):
            raise ValueError("NaN E4M3 code")
        self.units = np.ascontiguousarray(_SIGNED[codes])
        self.exp = np.ascontiguousarray(_EXP[codes])
        self.max_exp = int(self.exp.max(initial=-64))
        # Every value is an integer multiple of this many low binary places.
        mask = int(np.bitwise_or.reduce(np.abs(self.units).ravel(), initial=0))
        self.zeros = (mask & -mask).bit_length() - 1 if mask else 64


def _rtz_scaled(totals):
    exact = np.ldexp(totals.astype(np.float64, copy=False), -18)
    rounded = exact.astype(np.float32)
    down = np.abs(rounded) > np.abs(exact)
    return (rounded.view(np.uint32) - down.astype(np.uint32)).view(np.float32)


def _numpy_product(a, b, plain, cached_b):
    m, n = len(a.units), len(b.units)
    out = np.empty((m, n), dtype=np.float32)
    # Each product of two FP8 values in common 2^-18 units is integral;
    # all 32 terms together need under 42 bits, so binary64 addition is exact.
    if plain:
        av = a.units.astype(np.float64)
        for start in range(0, m, 32):
            out[start:start + 32] = _rtz_scaled(av[start:start + 32] @ cached_b.T)
        return out
    # For broad exponent spreads the exact truncation uses a different grid
    # per output cell. Only 2-D slabs are live, even for arbitrary E4M3 inputs.
    for i in range(0, m, 32):
        ar = slice(i, i + 32)
        for j in range(0, n, 512):
            br = slice(j, j + 512)
            av, bv = a.units[ar], b.units[br]
            ax, bx = a.exp[ar], b.exp[br]
            anchor = np.full((len(av), len(bv)), -128, dtype=np.int16)
            for p in range(32):
                np.maximum(anchor, ax[:, p, None].astype(np.int16) + bx[None, :, p], out=anchor)
            mask = (np.int64(1) << np.maximum(anchor - 7, 0).astype(np.int64)) - 1
            totals = np.zeros(anchor.shape, dtype=np.int64)
            for p in range(32):
                product = av[:, p, None].astype(np.int64) * bv[None, :, p]
                totals += (product + ((product < 0) * mask)) & ~mask
            out[ar, br] = _rtz_scaled(totals)
    return out


class Rank32Product:
    """Prepare F once and multiply batches of E against F-transpose."""
    def __init__(self, f, threads=4):
        self._b = _Factor(f)
        if not 1 <= threads <= 256:
            raise ValueError("threads must be in 1..256")
        self._threads = int(threads)
        self._float_b = None

    def __call__(self, e):
        a, b = _Factor(e), self._b
        # The highest possible anchor determines the coarsest possible grid.
        # When all products are divisible by that grid, no term loses bits.
        plain = a.zeros + b.zeros >= max(0, a.max_exp + b.max_exp - 7)
        if _ENGINE is None:
            if plain and self._float_b is None:
                self._float_b = b.units.astype(np.float64)
            return _numpy_product(a, b, plain, self._float_b)
        out = np.empty((len(a.units), len(b.units)), dtype=np.float32)
        _ENGINE(a.units.ctypes.data, a.exp.ctypes.data, b.units.ctypes.data, b.exp.ctypes.data,
                len(a.units), len(b.units), out.ctypes.data, plain, self._threads)
        return out

    def write_mixed(self, e, codes, alpha, beta, strip_exp, threshold, out):
        """Write one batch of final E4M3 codes using its official per-row parameters."""
        codes = np.asarray(codes)
        m, n = len(e), len(self._b.units)
        if codes.dtype != np.int8 or codes.shape != (m, n):
            raise ValueError("committed codes must be a matching int8 plane")
        if not isinstance(out, np.ndarray) or out.dtype != np.uint8 or out.shape != (m, n) or not out.flags.writeable:
            raise ValueError("output must be a writable uint8 plane of the same shape")
        row_args = []
        for row, dtype in ((alpha, np.float32), (beta, np.float32), (strip_exp, np.uint8), (threshold, np.uint8)):
            row = np.asarray(row)
            if row.shape != (m,) or row.dtype != dtype:
                raise ValueError("invalid row parameter dtype or shape")
            row_args.append(np.ascontiguousarray(row))
        alpha, beta, strip_exp, threshold = row_args
        if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta)):
            raise ValueError("row factors must be finite")
        if _MIX is not None:
            a, b = _Factor(e), self._b
            plain = a.zeros + b.zeros >= max(0, a.max_exp + b.max_exp - 7)
            codes = np.ascontiguousarray(codes)
            dest = out if out.flags.c_contiguous else np.empty((m, n), dtype=np.uint8)
            finite = _MIX(a.units.ctypes.data, a.exp.ctypes.data, b.units.ctypes.data, b.exp.ctypes.data,
                          codes.ctypes.data, alpha.ctypes.data, beta.ctypes.data, strip_exp.ctypes.data,
                          threshold.ctypes.data, m, n, dest.ctypes.data, plain, self._threads)
            if not finite:
                raise ValueError("non-finite E4M3 input")
            if dest is not out:
                out[:] = dest
        else:
            noise = self(e)
            signal = alpha[:, None] * codes.astype(np.float32)
            mixed = (noise.astype(np.float64) * beta[:, None] + signal).astype(np.float32)
            raw = encode_f32(mixed)
            safe = (raw & 128) | strip_exp[:, None] | (raw & 7) | 1
            coin = ((mixed[:, ::4].copy().view(np.uint32) >> 8) & 127) < 112
            below = (raw & 127) < threshold[:, None]
            out[:] = np.where(below & np.repeat(coin, 4, axis=1)[:, :n], safe, raw)
        return out


def noise_mm(e, f, threads=4):
    """Return the raw FP32 output of the rank-32 B200 noise product."""
    return Rank32Product(f, threads)(e)
