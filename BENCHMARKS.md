# Pearl testnet2 FP8 实测 / Measurements

每次稠密乘加计一次。完整扫描速率等于各作业的 `M × N × K` 之和除以扫描总秒数，再除以 10¹²。计时覆盖输入生成、承诺、矩阵乘法、折叠与筛票。开叶、网关排队和节点提交另计。商业矿池长期接受算力尚未实测。

One dense multiply-accumulate counts once. The full-scan rate is the sum of `M × N × K` over completed jobs, divided by total scan seconds and 10¹². Timing covers input generation, commitments, matrix multiplication, folding and ticket filtering. Proof opening, gateway queueing and node submission are timed separately. Long-run commercial pool accepted hashrate is unmeasured. [数据 / Data](benchmarks.json)

## v0.2.0 版本对应 / Measured builds

下表对应本次发包的程序及取得实测记录时的版本。每版扫描器、开叶程序和数值库均与实测文件逐字节一致。后面的算力和区块表沿用实测时的版本号。

The scanner, proof opener and native math library in each build are byte-identical to the tested binaries. The rate and block tables below use the versions that produced the measurements.

| 版本 / Edition | `--version` | 实测版本 / Tested as |
| --- | --- | --- |
| desktop | `v0.2.0` | `v0.1.2-dev.3` |
| 5060Ti-r10.3 | `v0.2.0+r10.3` | `v0.1.1-dev.2+r10.3` |
| Spark-gb10.2 | `v0.2.0+gb10.2` | `v0.1.2-dev.3-gb10.2` |

## RTX 5060 Ti 8 GB

`r10.3` 指 `v0.1.1-dev.2+r10.3`。同卡 180 W，核心偏移 +250 MHz，负载显存 7,001 MHz。每行两个版本使用相同的新种子。

`r10.3` is `v0.1.1-dev.2+r10.3`. Each row compares the same GPU and fresh seeds at a 180 W limit, +250 MHz core offset and a reported 7,001 MHz memory clock.

| 预设 / Preset | M × N × K | 配对种子 / Seeds | v0.1.1-dev.2 | r10.3 | 提升 / Gain |
| --- | --- | ---: | ---: | ---: | ---: |
| `throughput` | 1,044,480 × 540,672 × 13,312 | 2 | 100.348 TH/s | **100.839 TH/s** | +0.491 TH/s |
| `testnet2` | 131,072 × 131,072 × 16,384 | 24 | 98.525 TH/s | **98.985 TH/s** | +0.460 TH/s |

大矩阵两次为 100.862 和 100.816 TH/s，合计 149.100 秒。历史记录中的“约 101”是这两份完整作业合计值的整数舍入。后来用两个新种子同卡交叉复测，r10.3 为 100.245，桌面通用版为 100.229 TH/s，两版的票和开叶逐字节相同。

The two historical large jobs measured 100.862 and 100.816 TH/s in 149.100 s total. Their combined rate rounds to 101. A later crossover using two fresh seeds measured 100.245 on r10.3 and 100.229 TH/s on the desktop build, with byte-identical tickets and openings.

## 桌面通用版 / Desktop build

标准 testnet2 尺寸。每行比较 `v0.1.2-dev.2` 与 `v0.1.2-dev.3` 采用的改动。7,001 MHz 的历史实验与这组标准尺寸实验分别统计。

Each row compares the previous desktop build (`v0.1.2-dev.2`) with the change selected for `v0.1.2-dev.3` on the standard testnet2 shape. The last column gives the two-sided 95% interval for the paired group differences.

| GPU | 功耗上限 / Limit | 核心偏移 / Offset | 配对种子 / Seeds | 基线 / Before | 新版 / After | 配对差 95% 区间 / 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5060 | 145 W | +0 MHz | 40 | 73.192 TH/s | 73.319 TH/s | +0.067 … +0.188 TH/s |
| RTX 5060 Ti 8 GB | 180 W | +250 MHz | 40 | 98.950 TH/s | 99.064 TH/s | +0.059 … +0.168 TH/s |
| RTX 5070 | 175 W | +490 MHz | 24 | 119.031 TH/s | 119.218 TH/s | +0.129 … +0.246 TH/s |
| RTX 5080 | 250 W | +0 MHz | 24 | 174.709 TH/s | 177.816 TH/s | +2.551 … +3.659 TH/s |

两款 5060 在 13,801 MHz 显存档复用本作业的 A 输入。显式显存分配由约 2.05 增至 4.05 GiB，显存不足时自动回退，也可设置 `CYBERMINER_OPERAND_CACHE=off`。5070 交错调度相邻两列。5080 在性能档复用功耗监控连接，核心上限 2,400 MHz。

Both 5060 models reused the current job’s A input at a reported 13,801 MHz memory clock. Explicit allocations rose from about 2.05 to 4.05 GiB, with automatic fallback under memory pressure. Set `CYBERMINER_OPERAND_CACHE=off` to select the low-memory path. The 5070 interleaves adjacent column pairs. The 5080 reuses its power-monitor connection in the performance profile, with a 2,400 MHz core ceiling.

**RTX 5070 Ti** · 300 W / +0 MHz. 三段各 64 份新作业分别测得 139.02, 155.58, 150.52 TH/s，按总工作量合计 **148.04 TH/s**。24 个连续八作业批次中，达到 160 TH/s 的为 0 个。采用原主内核。

At 300 W and zero offset, the retained 5070 Ti kernel measured **148.04 TH/s** across three 64-job windows. None of the 24 consecutive eight-job batches reached 160 TH/s.

## DGX Spark GB10

同一低功耗设置，24 个配对新种子，核心请求 1,430 MHz。能耗来自设备累计能量计数器。

Same low-power settings and 24 matched seeds, with a 1,430 MHz core-clock request. Energy comes from the device’s cumulative counter.

| 版本 / Build | 每批行数 / Rows per strip | 扫描 / Full scan | 平均设备功率 / Mean device power | J / 10¹² MAC |
| --- | ---: | ---: | ---: | ---: |
| gb10.1 | 1,536 | 56.399 TH/s | 34.97 W | 0.62009 |
| gb10.2 | 1,024 | 64.808 TH/s | 39.12 W | 0.60357 |

每批 A 输入由 24 MiB 缩至 16 MiB。吞吐提高 14.91%，每万亿次乘加耗电下降 2.66%。另一次短测在实测 1787 MHz 下完成 2 份作业，测得 **82.70 TH/s / 58.74 W**。

The A input per strip fell from 24 to 16 MiB. Throughput rose 14.91% and energy per trillion MACs fell 2.66%. In a separate two-job run at 1787 MHz, the same build measured **82.70 TH/s at 58.74 W**.

## 真实 testnet2 模板 / Live testnet2 templates

按实际扫描时间统计。5060 Ti 两行均去掉首份预热。5070 Ti 统计两段运行中完成的 75 和 371 次扫描。

Rates use time spent scanning live templates. Both 5060 Ti rows exclude their first warm-up job. The 5070 Ti row combines 75 and 371 completed scans from two sessions.

| GPU | 版本 / Build | 完整扫描数 / Measured scans | 本地扫描 / Full scan |
| --- | --- | ---: | ---: |
| RTX 5060 Ti 8 GB | `v0.1.1-dev.2+r10.3` | 777 | 98.420 TH/s |
| RTX 5060 Ti 8 GB | `v0.1.2-dev.3` | 463 | 98.736 TH/s |
| RTX 5070 Ti | `v0.1.2-dev.3` | 446 | 156.117 TH/s |
| DGX Spark GB10 | `v0.1.2-dev.3-gb10.2` | 81 | 64.623 TH/s |

所列验证批次共记录 9 个不同的 testnet2 主链块。节点回查与至少两个公共对等节点的 V4 证书核验均通过。

Nine distinct blocks were on the testnet2 active chain when checked. Each record was checked against the node and V4 certificate data from at least two public peers.

| 实际出块版本 / Producing build | 硬件 / Hardware | 已验证块 / Verified blocks |
| --- | --- | ---: |
| `v0.1.2-dev.2` | RTX 5060, RTX 5060 Ti 8 GB, RTX 5070, RTX 5070 Ti, RTX 5080 | 5 |
| `v0.1.2-dev.3` | RTX 5060 Ti 8 GB, RTX 5070 Ti | 2 |
| `v0.1.2-dev.3-gb10.2` | DGX Spark GB10 | 1 |
| `v0.1.1-dev.2+r10.3` | RTX 5060 Ti 8 GB | 1 |
