# 第三方组件与授权

Cyberminer 项目自身代码采用 MIT 许可证。各依赖和随包第三方组件使用各自的授权。[项目许可证](LICENSE)

| 组件 | 随包内容或宿主依赖 |
| --- | --- |
| Pearl | 使用 testnet2 官方验证扩展的源码提交 `11956cb9c42a1b22ba6301a724b6dace9a1384fd`。任务文件沿用内部编码标识 `d797c3070a17e1d0fc1b30346a073a7de2b0eb23`。上游仓库根目录采用 ISC License，各子组件保留各自的告示。验证扩展由操作者提供。[上游告示](licenses/Pearl.txt) |
| NVIDIA CUTLASS 4.4.2 | 编译时使用模板头文件。随附 BSD-3-Clause 版权告示。[CUTLASS 许可](licenses/CUTLASS.txt) |
| NVIDIA CUDA Toolkit 13.0 | 两个 CUDA 可执行文件静态链接 CUDA Runtime。运行包使用系统 NVIDIA 驱动。选择高级 cuBLASLt 内核时，程序按需加载系统的 `libcublasLt.so.13` 并核对其构建标识。CUDA 工具包的许可及其附录随包保留。[NVIDIA CUDA 许可](licenses/NVIDIA-CUDA.txt) |
| NumPy | 操作者提供的 Python 数组运算包，采用 BSD-3-Clause。其发行包内另列第三方组件告示。 |
| blake3 | 操作者提供的 Python BLAKE3 绑定，可按 CC0-1.0 或 Apache-2.0 使用。 |
| OpenSSL 3 | 系统动态库 `libcrypto.so.3` 提供 SHA-256，OpenSSL 3 采用 Apache-2.0。 |

授权原文中的公开归属和供应商联系信息按原样保留。
