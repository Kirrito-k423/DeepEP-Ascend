# EP64 FP8 Dispatch：预热 10 次，采集第 11 次

供 A5 内网测试使用。配置固定为 64 ranks、每 rank **4096 个实际 token**、384 专家、top-k 6、hidden 7168。
每 rank 有 6 个专家，使用 FP8 E4M3、row-major scales、float32 top-k weights、expert alignment 128、32 AI cores / 64 AIV。
采集 fresh dispatch 和接收展开 epilogue；不运行 Combine。路由为 seed=rank 的均匀随机 top-k，保留同 rank 去重；默认不清 L2。

## 准备

使用上游要求的 A5 / CANN 9.2 / PyTorch 与 torch_npu 配套环境，并按上游 README 检查 HDK、固件和 UBMEM/URMA 连通性。
以下分支包含核心打点、运行入口和本说明。复制代码到离线内网时必须带上两个 submodule 的实际文件，不能只复制 gitlink。

```bash
git clone -b codex/ep64-debugclock-guide https://github.com/Kirrito-k423/DeepEP-Ascend.git
cd DeepEP-Ascend
git submodule update --init --recursive third-party/deep_jit third-party/ascend-kernel-lab
source /path/to/cann/set_env.sh
python -m pip install --no-build-isolation .
export PYTHONPATH="$PWD/third-party/ascend-kernel-lab/python:$PYTHONPATH"
```

AKL 固定为 `a697a16953c8d22bc5effc85c0b55b5bf3ba72a1`；DeepJIT 固定为上游 gitlink。
更新 AKL 时也必须更新 `csrc/runtime/jit.hpp` 的 cache signature。JIT 自动编译关闭、空打点、开启三种独立版本。
wheel 包含 AKL 设备头；离线解析用上面的 AKL Python 源目录，不需要在 Mac 安装 DeepEP 或 CANN。

## 启动

以 8 节点 × 每节点 8 个 NPU 为例，在每个节点设置相同的 MASTER_ADDR、独立 NODE_RANK=0..7。
设备数量不同时可以调整节点数和每节点进程数，但乘积必须是 64。各模式使用新进程和新输出目录。

```bash
set -e
export MASTER_ADDR=<node0-ip>
export NODE_RANK=<0-to-7>
for mode in off empty on; do
  torchrun --nnodes=8 --nproc-per-node=8 --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port=29500 \
    scripts/capture_ep64_dispatch.py --clock-mode="$mode" --output="results/$mode"
done
```

所有节点需执行同一 mode 顺序。输出可以位于共享目录；若使用节点本地目录，采集完成后汇总所有 rank 目录。
脚本会执行恰好 11 次相同配置的 fresh Dispatch：前 10 次预热，第 11 次读取 NPU event 时间。
`on` 和 `empty` 都只在两个 kernel 各自的第 11 次分配 trace buffer；内核完成后再统一导出。
`empty` 保留末尾同步和写回但跳过时钟读取；`off` 不分配 trace 内存，也不执行设备打点。
采集后以 HCCL reference 按位检查 FP8 payload、scales、weights、source metadata、展开 slot 和 padding；错误使进程失败。

## 输出与解析

每个模式必须有 64 份 `rank*/manifest.json`，全部 `status=passed` 才能认定整次运行成功。
manifest 保存实际配置、版本、AKL revision、NPU event 的 `sample_us` 和精度结果。
`on/raw/<kernel>/rank*-pid*-launch*/{capture.json,trace.bin}` 按 kernel 汇总所有 rank，合计 **128 份 capture**。
目录名的 launch 是 AKL 导出编号，采集轮次以 manifest 的 `captured_iteration=11` 为准。
每个 kernel、每个 AIV 独享 32 个槽，即 `64 + 32×16 = 576 B`；每 rank 两个 GM buffer 合计 72 KiB。
`empty` 的原始行只有有效头部，没有事件，因此不送入语义时间线解析器。

```bash
python -m akl.semantic results/on --keep-intermediates \
  --source deep_ep/include/deep_ep/impls/ep/dispatch.hpp \
           deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.hpp \
           deep_ep/include/deep_ep/common/dispatch_trace.hpp
python -m zipfile -c ep64-debugclock.zip results
```

AKL 输出 HTML、SVG、事件 JSONL、summary 和 Trace JSON。先保留 raw 与 manifest，再生成报告。
默认用原始 tick；确认本机 `GetSystemCycle` 频率后才给解析命令加 `--clock-mhz=<verified MHz>`。
不要把 AIV/跨 rank 的时钟当成已校准的统一时间轴。用每个 AIV 内相邻点的整数差分析阶段耗时。

## 打点含义

| Kernel | 重点区间 | 解读 |
| --- | --- | --- |
| dispatch | entry → soc-ready | 可观测入口到 SoC 初始化结束 |
| dispatch | barrier-begin → barrier-end | 原有跨 rank 入口同步 |
| dispatch | simt-issued → simt-done | 覆盖 SIMT 路由、计数、WQE，与 Scalar metadata / 本地 copy 重叠；不是可相加的串行阶段 |
| dispatch | metadata-begin → metadata-ready | scales、weights、源 metadata 整理与写回 |
| dispatch | metadata-ready → local-slots-ready | 等待 SIMT 发布本地 slot |
| dispatch | local-slots-ready → local-copy-done | 本 rank payload copy 完成 |
| dispatch | simt-done → doorbell-issued | Jetty head 更新与 doorbell 区域；不代表远端数据已可用 |
| epilogue | remote-wait-begin → remote-wait-end | 原有通信完成屏障，含 rank/AIV 等待，不是纯网络延迟 |
| epilogue | histogram-begin → layout-ready | 专家 histogram、前缀和、输出 slot 准备 |
| epilogue | cumsum-wait-begin → cumsum-wait-end | AIV 间前缀和完成等待 |
| epilogue | expand-begin → expand-issued | payload、scales、weights、metadata 的展开写发射 |
| epilogue | expand-issued → padding-issued → pipes-drained | padding 发射及诊断末尾流水 drain；最后一点在 trace 写回之前 |

主 kernel 第 11 次末尾的 trace 写回会推迟 epilogue 起点。`sample_us` 在 trace 模式还包含诊断分配、同步和写回扰动。
性能基线使用 `off`；`empty/on` 用于估计扰动，不能拿 `on` 的等待区间直接宣称生产网络延迟。
只有一个第 11 次样本，不据此宣称统计稳定的吞吐或优化收益；带宽也不能用 `top-k×payload` 直接当作真实网络字节。

## 内网回传

请返回完整 ZIP、64 rank 日志、CANN/驱动/固件/HDK 版本与拓扑；保留失败日志和 incomplete manifest。
报告分列 off 的操作耗时、empty/on 的扰动、每 AIV 的阶段 tick、最慢 rank、精度和 dropped。
任一 capture 丢事件、时钟回退、rank 缺失或精度失败都需要先说明，再分析瓶颈。
本 PR 尚无 A5 64-rank 实测结果；现有验证覆盖主机采集逻辑与 A5 目标 DebugClock 探针编译/链接。
完整上游与修改版均无法在现有 CANN 9.1 编译，缺少上游要求的 CANN 9.2 HCOMM 类型；内网仍需完整构建及多 rank 验证。
