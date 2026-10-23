# Qwen3-TTS：帧内 KV cache＋整帧 AR 图

从 `v0.30.0`（a8576ccb725c4e21cd13c3eb5f9a546b21149d2b）开发，分支 `codex/qwen3-tts-kv-frame-graph`。
只使用原始 AR，不训练、不投机。CPU 通过不代表 NPU 通过，未收到服务器结果前性能待验证。

## 执行模式

| 模式 | 每帧主干 | Code Predictor 图 replay | 作用 |
|---|---:|---:|---|
| stock | 15 次 | 独立帧工具中 15 次；服务外层图可能合并 | tag 原始实现 |
| kv_step | 15 次 | 15 次 | 首次两位置 prefill，然后 14 次单位置 decode |
| kv_frame | 15 次 | 1 次 | projection、主干、Heads、采样、embedding、KV 更新全部展开捕获 |

`code_predictor_ar_execution` 放在部署 YAML 的共享内存 connector `extra` 中，默认 stock。
KV 模式显式关闭外层 talker_mtp 包装，保留正常 Talker 图；捕获失败不会回退。
按 batch 1/2/4/8/16 和 greedy/sampling 分别捕获。单 worker 内只支持同一执行流串行调度，不支持并行写同一缓存。
完整图和小图共享冻结权重，每个 bucket 有独立缓存。KV 只在一帧内有效，下一帧从位置 0 覆盖。

采样默认 temperature=0.9、top-k=50、top-p=1；其他随机参数明确拒绝。每帧 uniforms 在图外更新。
生产 generator 按码本再按有效请求行消费随机数；padding 行不消耗真实请求的 generator。
验证 uniforms 由请求/帧身份固定，避免 batch 排列改变采样。

## 服务器准备

先将此分支文件部署到下面的仓库目录，并确认 Python 导入的是该目录；不要在旧实验进程运行时切换其代码。
本文命令不使用环境变量。vLLM 与 vLLM Ascend 应已安装到当前 NPU Python 环境。

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts

git branch --show-current
python -c "import vllm_omni; print(vllm_omni.__file__)"
```

本地实现位于独立 worktree；不会自动更新服务器仓库。
仅 service_bench.py 支持同目录重新运行：先停止上一轮服务测试，再执行原命令，脚本会清空指定的服务 results 目录并重新开始，不保留旧结果。其他工具的 results 子目录仍必须尚不存在，保持原有行为。
首次模型校验会读取模型权重；首次图捕获可能耗时较长，不计入正式计时。

## 1. 只读复用已有采集

```bash
python tools/qwen3_tts_ar_graph/experiment.py collect \
  --import-from /softwarePlatform/y00899301/qwen3_tts_zero_train_spec_run01 \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/capture \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0
```

导入只创建引用，不改原文件。验证数据包含固定 30 条文本和 20 条测试请求的帧；检查模型配置和 predictor 权重。
旧的独立 Q/K/V、gate/up 权重仅按行拼接为 tag 的融合存储，不迁入投机逻辑。
若检查提示不兼容或缺帧，使用本文末尾的重新采集流程。

## 2. 固定三帧检查

```bash
python tools/qwen3_tts_ar_graph/experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/probe \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --probe-only \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

固定 smoke011 的 0、2 帧及 smoke030 最后一帧，两种采样模式，含非整桶 batch。
检查 cached eager 逐位置 logits、三种路径 codes、重复运行、随机数更新、请求换序、batch 与单请求对照。
这张 probe 证书不能授权 bench。

## 3. 60 帧抽样检查

```bash
python tools/qwen3_tts_ar_graph/experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/check \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

每个测试请求取首/中/末帧，共 60 帧；覆盖 batch 1/2/3/4/5/8/9/15/16。
首个失败保存 failure.pt、correctness.json，停止，不放宽 logits 容差（atol/rtol 均为 0.02）或忽略 codes 差异。
若 stock 的 batch 与单请求自身不一致，同样停止并诊断数值布局。
查看结果：

```bash
cat /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/check/correctness.json
```

## 4. 帧级性能

```bash
python tools/qwen3_tts_ar_graph/experiment.py bench \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/capture \
  --correctness /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/check/correctness.json \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/bench \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --warmup 20 \
  --executions-per-round 60 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend

cat /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/bench/report.md
```

五个 batch bucket、greedy/sampling、stock→候选→候选→stock 四轮配对。
模型权重、源代码、软件环境、数据或帧集合变化会令旧证书失效。
均为完整帧调用计时：包含身份绑定 uniforms 的 CPU 生成与 H2D、输入暂存及输出克隆；上游帧输入预先在设备上。
该随机数准备是验证工具的固定口径，不等同于生产服务的随机数生成成本，需结合服务结果判断。
报告 batch 延迟、有效帧/s、设备事件 p50/p95、配对 batch bootstrap 区间。
设备事件可能含主机提交空隙，不能称为纯 kernel 时间；循环回放不增加独立样本数。
内存峰值含常驻 stock 图，不是候选独占内存。服务图范围不能由独立帧工具推断。

## 5. 服务多并发

服务测试不再要求或校验帧级正确性证书；旧命令的 `--correctness` 可省略，保留时不会读取该文件。模型与源码版本仍写入运行元数据，不作为服务启动拦截条件。其他入口的证书要求不变。

先停止占用同一张 NPU 的其他测试。脚本逐个启动/停止自己创建的服务，不操作已有服务；8091 端口必须空闲。
每组先执行带 seed 的取消/替换检查，然后并发 1/2/4/8/16，每档 16 次预热、120 请求、三轮。
30 条文本明确展开四次，不依赖随机 oversampling。服务采样沿用默认参数；预热/取消检查失败会停止性能实验。

```bash
python tools/qwen3_tts_ar_graph/service_bench.py \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/capture \
  --correctness /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/check/correctness.json \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/service \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --port 8091 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

三套 deploy 保持 Codec 分块策略相同，仅配置执行模式和 max_num_seqs=16。
后台每秒落盘一次 host batch 计数，不同步设备。每轮 batch-before/after 快照包含 benchmark 的预热请求。
报告请求吞吐、音频吞吐、TTFP、E2E、RTF、underrun、错误率和真实 batch 分布；完整上游结果保留在 result.json。
不同 API 请求的输出长度可能不同，吞吐同时报告请求/s 与音频秒/s。不将客户端并发数视为实际 batch。

## 6. 独立 profiling

```bash
python tools/qwen3_tts_ar_graph/experiment.py profile \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/capture \
  --correctness /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/check/correctness.json \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_run01/profile \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --warmup 20 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

固定六帧构造六个诊断 batch，batch 1/4/16，三组执行模式和两种采样。
用于检查图间空隙、Head/采样算子和 memcpy；不把 profiler 延迟混入 bench。

## 需要重新采集时

用新目录 prepare，再单独终端启动 stock 服务，最后发送顺序请求。不要与 benchmark 同时运行。

```bash
python tools/qwen3_tts_ar_graph/experiment.py collect \
  --prepare \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_capture02 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0

vllm-omni serve /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_capture02/deploy.yaml \
  --host 127.0.0.1 --port 8091 --trust-remote-code --omni
```

服务就绪后，在另一终端：

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts
python tools/qwen3_tts_ar_graph/experiment.py collect \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_graph_capture02 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --url http://127.0.0.1:8091
```

后续将 `--input` 改为该目录。采集会同步写帧文件，不用于性能评估。

## 本地 CPU 测试

```bash
python -m pytest tests/model_executor/models/qwen3_tts/test_cached_ar.py tests/model_executor/models/qwen3_tts/test_ar_graph_tools.py tests/model_executor/models/qwen3_tts/test_code_predictor_dtype.py -q -o addopts= --confcutdir=tests/model_executor/models/qwen3_tts
```

CPU 不模拟 NPU kernel 数值，也不能确认 graph capture 成功。NPU 上必须先执行 probe 和 check。
