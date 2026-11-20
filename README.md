# Cyberminer

适用于 RTX 5060 Ti 8 GB 的 Pearl V4 FP8 矿工。**0% 开发费，矿工源码完全公开。**

Pearl V4 FP8 miner for the RTX 5060 Ti 8 GB. **Fully open source, with a 0% developer fee.** [MIT](LICENSE) · [Third-party notices](THIRD_PARTY.md)

## 编译与试跑 / Build and run

在 Linux x86_64 上准备 Python 3.11+、CUDA Toolkit 13.0 及兼容驱动、CUTLASS 4.4.2、支持 C++17 和 OpenMP 的 GNU 编译器、GNU binutils 和 OpenSSL 3 开发文件。从仓库根目录执行以下命令。将 `CUTLASS_ROOT` 指向已有的 CUTLASS 源码目录，`CUDA_VISIBLE_DEVICES` 选择显卡。

On Linux x86_64, install Python 3.11+, CUDA Toolkit 13.0 with a compatible driver, CUTLASS 4.4.2, GNU C++17 with OpenMP, GNU binutils, and the OpenSSL 3 development files. From the repository root, point `CUTLASS_ROOT` at your CUTLASS source tree and select your GPU with `CUDA_VISIBLE_DEVICES`.

```sh
export CUTLASS_ROOT="$PWD/../cutlass"
export CUDA_VISIBLE_DEVICES=0
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install -r requirements.txt
bash build.sh
python3 -B fp8miner.py bench --quick
```

## 复测速率 / Reproduce the benchmark

```sh
python3 -B fp8miner.py bench --preset throughput --jobs 2
python3 -B fp8miner.py bench --preset testnet2 --jobs 8
```

单张 RTX 5060 Ti 8 GB，功耗上限 **180 W**，负载下驱动报告的显存频率 **7,001 MHz**。用 `nvidia-smi -q -d CLOCK,POWER` 查看实际频率和功耗。矿工沿用已有设置。`--quick` 只用于安装自检。

Single RTX 5060 Ti 8 GB, **180 W** power limit and **7,001 MHz** memory clock as reported under load. Check yours with `nvidia-smi -q -d CLOCK,POWER`. The miner uses your existing clock settings. Use `--quick` for an installation check.

| 预设 / Preset | M × N × K | 预热后任务 / Measured jobs | 扫描时间 / Scan time | 本地扫描 / Local scan |
| --- | --- | ---: | ---: | ---: |
| `throughput` | 1,044,480 × 540,672 × 13,312 | 2 | 149.92 s | 100.29 TH/s |
| `testnet2` | 131,072 × 131,072 × 16,384 | 8 | 22.80 s | 98.76 TH/s |

每次稠密乘加计一次。`last scan` 是最近一次完整扫描的速率，`session avg` 按已完成工作量和整段用时计算。矿池接受份额算力尚未实测。加 `--no-status` 可隐藏终端状态行。

One dense multiply-accumulate counts as one unit of work. `last scan` measures the latest completed scan. `session avg` divides completed work by elapsed time. Pool-side accepted hashrate is unmeasured. Add `--no-status` to hide live status lines.

## Testnet2 solo

将本地 Pearl V4 网关连到已同步的 testnet2 节点，在网关中设置自己的测试网收款地址。设置 `PEARL_GATEWAY_SOCKET` 为网关的 Unix 套接字路径，`PEARL_NATIVE_DIR` 为官方 `pearl_mining` 扩展所在目录。使用下方固定版本构建该扩展。

Connect your local Pearl V4 gateway to a synced testnet2 node and configure your own testnet payout address on the gateway. Set `PEARL_GATEWAY_SOCKET` to its Unix socket path and `PEARL_NATIVE_DIR` to the directory containing the official `pearl_mining` extension built from the pinned revision. [Pearl source](https://github.com/pearl-research-labs/pearl/commit/11956cb9c42a1b22ba6301a724b6dace9a1384fd)

```sh
python3 -B fp8miner.py mine --preset testnet2 \
  --socket "$PEARL_GATEWAY_SOCKET" --native-dir "$PEARL_NATIVE_DIR" \
  --work-dir runs --attempts 1
```

省略 `--attempts 1` 即可持续挖矿。`gateway_queued` 表示网关已排队，出块以节点主链为准。两次 testnet2 solo 测试共完成 481 个任务，**4 个块经节点确认进入主链**。合计扫描平均 95.53 TH/s。

Omit `--attempts 1` to keep mining. `gateway_queued` acknowledges queuing. Check the node for block acceptance. Across two testnet2 solo sessions, 481 completed jobs produced **4 blocks confirmed on the active chain**. The combined scan rate was 95.53 TH/s. [Testnet2 FP8 激活块 / activation block 150,283](https://blockbook.testnet2.pearlnetwork.net/block/150283)

## 自愿打赏 / Optional donations

如果 Cyberminer 对你有帮助，欢迎使用 **Pearl 主网 PRL** 自愿打赏。qwq

If Cyberminer has been useful to you, you can send an optional tip in **PRL on the Pearl mainnet**.

`prl1pw0sxd2j7s7hjfsyrh4pq4hmwsd8w00c6wqcqyjggxj4dd6vykmsqve30mm`
