# Cyberminer v0.2.0

Pearl testnet2 FP8 矿工。**0% 开发费，矿工源码完全公开，MIT 许可。**

Open-source FP8 miner for Pearl testnet2. **0% developer fee, MIT licensed.** [LICENSE](LICENSE) · [Third-party notices](THIRD_PARTY.md)

**本包版本 / This build**　`v0.2.0` · linux-x86_64

## 选择运行包 / Choose a build

相较 v0.1.0，桌面版新增 RTX 5060、5070、5070 Ti 和 5080，继续支持并优化 RTX 5060 Ti 8 GB。DGX Spark GB10 使用独立 ARM64 版。

Since v0.1.0, the desktop miner adds the RTX 5060, 5070, 5070 Ti and 5080 and improves the RTX 5060 Ti 8 GB. DGX Spark GB10 has a separate ARM64 build.

| 设备和显存 / Hardware and VRAM | Releases 中的包名前缀 / Release asset prefix | Linux |
| --- | --- | --- |
| RTX 5060 / 5060 Ti **8 GB** · RTX 5070 **12 GB** · RTX 5070 Ti / 5080 **16 GB** | `Cyberminer-desktop-v0.2.0` | x86_64 |
| RTX 5060 Ti **8 GB**，复现 r10.3 大矩阵成绩 / historical throughput runs | `Cyberminer-5060Ti-v0.2.0-r10.3` | x86_64 |
| NVIDIA DGX Spark GB10 | `Cyberminer-Spark-v0.2.0-gb10.2` | aarch64 |

每版都有 `-source.zip` 源码和 `-linux-<arch>.zip` 预编译包。仓库根目录是桌面通用版源码。

Each variant ships as source (`-source.zip`) and a prebuilt runtime (`-linux-<arch>.zip`). The repository root contains the desktop source.

## 运行与测速 / Run a benchmark

将所选 ZIP 和 `SHA256SUMS.txt` 放在同一目录，核验后解压该 ZIP。下面以本版预编译包为例。多卡机器可用 `CUDA_VISIBLE_DEVICES` 选择一张卡。需要 Python 3.11+、兼容 CUDA 13.0 的 NVIDIA 驱动、系统 glibc、GNU C++/OpenMP 运行库和 OpenSSL 3。  
Save the chosen ZIP and `SHA256SUMS.txt` together. Verify the ZIP, then follow the prebuilt example below. On a multi-GPU host, set `CUDA_VISIBLE_DEVICES` to select one card. Use Python 3.11+, an NVIDIA driver compatible with CUDA 13.0, system glibc, GNU C++/OpenMP runtimes and OpenSSL 3.

```sh
sha256sum --check --ignore-missing SHA256SUMS.txt &&
python3 -m zipfile -e Cyberminer-desktop-v0.2.0-linux-x86_64.zip . &&
cd Cyberminer &&
chmod +x build/pearl-fp8-scan build/pearl-fp8-open &&
python3 -m venv .venv && . .venv/bin/activate &&
python3 -m pip install -r requirements.txt &&
python3 -B fp8miner.py --version
```

源码编译另需 CUDA Toolkit 13.0、GNU C++17、OpenMP、binutils 和 OpenSSL 3 开发文件。桌面版先将 `CUTLASS_ROOT` 指向 CUTLASS 4.4.2 源码目录，然后运行 `bash build.sh`。Spark 在 ARM64 上直接运行 `bash build.sh`。使用预编译包可直接测速。  
To build from source, install CUDA Toolkit 13.0, GNU C++17 with OpenMP, binutils and OpenSSL 3 development headers. On desktop, set `CUTLASS_ROOT` to a CUTLASS 4.4.2 source tree and run `bash build.sh`. On Spark, run `bash build.sh` on ARM64. The prebuilt archive is ready to benchmark after Python setup.

RTX 5080 需先将核心偏移设为零、板卡功耗上限设为 250 W。默认配置的核心上限为 1,500 MHz。复现数据表内的性能档用 2,400 MHz 上限，并设置 `CYBERMINER_5080_PROFILE=performance`。其他型号直接使用下面的标准命令。  
On RTX 5080, configure zero core offset and a 250 W board limit. Use a 1,500 MHz core ceiling for the default profile. To reproduce the performance row, use a 2,400 MHz ceiling and set `CYBERMINER_5080_PROFILE=performance`. For the other GPUs, run the standard command below.

```sh
python3 -B fp8miner.py bench --preset testnet2 --jobs 8
```

RTX 5060 Ti 8 GB 可单独复测大矩阵。 / Optional large-matrix benchmark for the RTX 5060 Ti 8 GB.

```sh
python3 -B fp8miner.py bench --preset throughput --jobs 2
```

在 RTX 5060 Ti 8 GB 上，通用版复用当前作业的 A 矩阵输入。标准 testnet2 尺寸、40 个配对新种子，完整扫描从测试基线 `v0.1.2-dev.2` 的 **98.950** 提升至 **99.064 TH/s**。r10.3 专版在两份历史大矩阵作业上合计 **100.839 TH/s**。

On the 5060 Ti 8 GB, the desktop build reuses the current job’s A-matrix input. Full scans on the standard testnet2 shape rose from **98.950** on the `v0.1.2-dev.2` baseline to **99.064 TH/s** across 40 matched seeds. The r10.3 build measured **100.839 TH/s** across two historical large-matrix jobs. [实测与出块 / Benchmarks and block validation](BENCHMARKS.md)

## Testnet2 solo

将本地 Pearl 网关连到已同步的 testnet2 节点，在网关中设置自己的测试网收款地址。将 `PEARL_GATEWAY_SOCKET` 指向网关 Unix 套接字，将 `PEARL_NATIVE_DIR` 指向含 `pearl_mining` 的目录。使用下方固定提交构建官方 V4 验证扩展。  
Connect your local Pearl gateway to a synced testnet2 node and configure your own testnet payout address on the gateway. Set `PEARL_GATEWAY_SOCKET` to its Unix socket and `PEARL_NATIVE_DIR` to the directory containing the official `pearl_mining` V4 verifier built from the pinned revision. [Pearl `11956cb`](https://github.com/pearl-research-labs/pearl/commit/11956cb9c42a1b22ba6301a724b6dace9a1384fd)

```sh
python3 -B fp8miner.py mine --preset testnet2 \
  --socket "$PEARL_GATEWAY_SOCKET" --native-dir "$PEARL_NATIVE_DIR" --work-dir runs
```

程序在本机开叶、复算候选票并验证证明。`gateway_queued` 表示网关已排队，出块看节点主链。  
The miner opens Merkle leaves locally, recomputes the candidate ticket and checks the proof. `gateway_queued` acknowledges queueing. Check the node’s active chain for block acceptance. [Testnet2 FP8 激活块 / activation block 150,283](https://blockbook.testnet2.pearlnetwork.net/block/150283)

## 自愿打赏 / Optional donations

如果 Cyberminer 对你有帮助，欢迎使用 **Pearl 主网 PRL** 自愿打赏。qwq

If Cyberminer has been useful to you, you can send an optional tip in **PRL on the Pearl mainnet**.

`prl1pw0sxd2j7s7hjfsyrh4pq4hmwsd8w00c6wqcqyjggxj4dd6vykmsqve30mm`
