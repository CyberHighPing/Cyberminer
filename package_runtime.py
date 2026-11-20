"""Package a local Linux x86_64 build with fixed ZIP membership and permissions."""
import argparse
import os
from pathlib import Path
import stat
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent
PROGRAMS = {"build/pearl-fp8-scan", "build/pearl-fp8-open"}
LIBRARY = "build/fast_prepare.native.so"
MEMBERS = sorted(PROGRAMS | {
    LIBRARY, "LICENSE", "README.md", "THIRD_PARTY.md", "requirements.txt",
    "licenses/CUTLASS.txt", "licenses/NVIDIA-CUDA.txt", "licenses/Pearl.txt",
    "cuda_worker.py", "fast_prepare.py", "fp8miner.py", "gateway_miner.py", "hashrate.py",
    "gpu_proof.py", "host_math.py", "host_miner.py", "lottery.py",
    "native_job.py", "native_scan.py", "sparse_tickets.py",
})


def package(output):
    with tempfile.TemporaryDirectory(dir=output.parent) as stage:
        candidate = Path(stage) / "archive.zip"
        with ZipFile(candidate, "w") as archive:
            for name in MEMBERS:
                path = ROOT / name
                if path.is_symlink() or not path.is_file() or path.resolve() != path:
                    raise ValueError(f"missing or indirect package member: {name}")
                raw = path.read_bytes()
                if name.startswith("build/") and (raw[:6] != b"\x7fELF\x02\x01" or raw[18:20] != b">\0"):
                    raise ValueError(f"expected a little-endian x86_64 ELF: {name}")
                info = ZipInfo("Cyberminer/" + name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | (0o755 if name in PROGRAMS else 0o644)) << 16
                archive.writestr(info, raw, compress_type=ZIP_DEFLATED, compresslevel=9)
        os.link(candidate, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("Cyberminer-linux-x86_64.zip"),
                        help="new ZIP path")
    package(parser.parse_args().output)
