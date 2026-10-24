# Qwen3-TTS：商业版高效 KV 路径与整帧 AR 图

基于 v0.30.0，分支 `codex/qwen3-tts-kv-frame-graph`。只做原始 AR，不训练、不投机。
本地 CPU 检查不代表 NPU 算子或图已验证；实际收益由服务器结果决定。

## 执行路径与图数量

| 配置 | KV/attention | 每帧主干 | 每帧 Code Predictor replay | 全部五个 bucket 的图数 |
|---|---|---:|---:|---:|
| stock | v0.30 原始重算路径 | 15 | 帧级工具 15，服务以实际外层图为准 | 沿用原实现 |
| kv_step | FIA 原生 GQA，静态 KV，短序列投影 | 15 | 15 | 75，采样模式共用 |
| kv_frame | 与 kv_step 相同 | 15 | 1 | 10，greedy/sampling 各 5 |

KV 预热捕获 batch buckets `[1,2,4,8,16]`，输出捕获进度和耗时；首次初始化发生在服务预热调用中。
图按实际推理 batch 选择，不按客户端并发选择；batch=3 使用 bucket=4。
KV 模式关闭外层 MTP 包装，保留正常 Talker 图；不嵌套 replay。
FIA 直接读取完整 KV-head 缓存，有效长度排除旧数据，不扩展 GQA heads 或复制缓存前缀。
`o_proj`、`down_proj` 只处理 prefill 两位置或 decode 单位置，保留当前融合 QKV、gate/up、RoPE、RMSNorm。
两种 KV 模式共用缓存主干；kv_frame 另外启用下文的整帧 Gumbel 和投影表预计算。每个 bucket 独立 KV 缓存，同一执行流串行调度。
输出在缓冲区复用前克隆；新帧从位置 0 覆盖，禁止读取本帧有效长度之外的缓存。
FIA 不可用、非 BF16、捕获失败或不支持的采样参数会明确报错，不静默回退。

采样参数为 temperature=0.9、top-k=50、top-p=1，也支持 greedy。
整帧 uniforms 在图外准备；按码本、有效请求行消费 generator，填充行不消耗真实请求随机流。

## 各测试命令独立运行

check、bench、profile、服务测试均可独立运行，没有先通过哪个测试才能运行下一个的要求。
bench/profile 不读取正确性证书；旧命令中的 `--correctness` 可省略，保留时也忽略。
各命令仍需实际使用的数据和模型；report 读取已有计时文件进行汇总，不判断其他测试是否通过。

## 数值检查口径

`--stock-comparison strict|record` 默认 strict；本次命令显式使用 record。
record 单独记录 stock 与缓存 eager 的 codes 差异，允许测性能；不表示与 stock 完全一致。
缓存 eager、kv_step、kv_frame 必须 codes 一致，逐位置 logits 使用 atol=rtol=0.02。
重复运行、随机数更新、行换序、bucket 切换、请求替换和输出生命周期仍严格检查。
检查命令发现非有限值、非法 token、状态污染或 KV 执行差异时保存诊断并结束本次检查，不拦截 bench/profile/服务测试。
性能命令仍会报告自身的运行错误，例如图捕获失败。
`--stock-batch-policy record-stable` 单独处理 stock 自身稳定的跨 batch 差异，与上述选项互不替代。

检查输出：
- `correctness.json`：KV 一致性、stock 差异统计及代码/数据/策略绑定。
- `stock_comparison.json`：每组自由生成 codes 差异，以及强制使用 stock 历史前缀后的逐位置 logits 差异。
- `stock_batch_variations.json`：stock 自身的跨 batch 差异（存在时写入）。
- `failure.pt`：首个失败输入、位置、logits、KV 和 codes。失败时停止；stock record 差异不算执行失败。

统计中的帧次包括不同 batch/采样模式下的重复检查，不等于独立帧数。

## 服务器准备

部署本分支代码到指定服务器仓库，确认 Python 导入路径正确。vLLM、vLLM Ascend 应已安装到当前 Python 环境。
复用现有 capture，不需要重新采集。以下命令使用新的统一结果目录，不覆盖旧 capture。
非服务工具的结果子目录须尚不存在；服务工具支持清理旧结果后重新运行。

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts
git branch --show-current
python -c "import vllm_omni; print(vllm_omni.__file__)"
```

## 1. 固定三帧预检

smoke011 第 0、2 帧和 smoke030 最后一帧。覆盖两种采样模式及 batch 1/2/3/4/5/8/9/15/16。

```bash
python tools/qwen3_tts_ar_graph/experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_ar_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_ar_graph_commercial_kv_run01/probe \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --stock-comparison record \
  --stock-batch-policy record-stable \
  --probe-only \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

## 2. 60 帧抽样检查

每个测试请求取首/中/末三帧，不做全量检查。本检查可选，结果不作为其他命令的前置条件。

```bash
python tools/qwen3_tts_ar_graph/experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_ar_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_ar_graph_commercial_kv_run01/check \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --stock-comparison record \
  --stock-batch-policy record-stable \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

## 3. 三组帧级性能

每组预热 20 batch，每轮 60 batch，stock→候选→候选→stock 四轮，两种采样和五个 bucket。
计时包含身份绑定随机数的 CPU 生成/H2D、输入暂存、执行和输出交接；与生产随机数成本不同。
不设置性能门槛。报告 stock、kv_step、kv_frame 延迟、吞吐、配对区间和 step/frame 均值比。
step/frame 来自各自的 ABBA 测量，不是直接配对置信区间。设备事件包含潜在主机空隙，不是纯 kernel 时间。

```bash
python tools/qwen3_tts_ar_graph/experiment.py bench \
  --input /softwarePlatform/y00899301/qwen3_tts_ar_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_ar_graph_commercial_kv_run01/bench \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --stock-comparison record \
  --stock-batch-policy record-stable \
  --warmup 20 \
  --executions-per-round 60 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

```bash
cat /softwarePlatform/y00899301/qwen3_tts_ar_graph_commercial_kv_run01/bench/report.md
```

## 4. 独立 profiling

六个固定诊断 batch，三组模式、两种采样、batch 1/4/16。核对 FIA 算子、短 GEMM 输入形状、图启动数量及设备空隙。
profiling 不混入正式计时；缓存后端和图元数据保存在 manifest.json。

```bash
python tools/qwen3_tts_ar_graph/experiment.py profile \
  --input /softwarePlatform/y00899301/qwen3_tts_ar_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_ar_graph_commercial_kv_run01/profile \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --stock-comparison record \
  --stock-batch-policy record-stable \
  --warmup 20 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

## 5. 三组服务自动比较

先停止手动服务，释放 NPU 与端口。脚本依次启动/停止 stock、kv_step、kv_frame，部署配置只改变 Code Predictor 模式。
每组先做取消/替换检查，再按并发 1/4/16 测试，每档预热 3 请求、正式 120 请求、1 轮。
本地请求绕过代理；健康检查每 20 秒，客户端每 20 秒输出进度。服务测试不读取或校验正确性证书。
服务报告同时保留音频时长、吞吐、TTFP、E2E、RTF、underrun、错误率、实际 batch 分布。输出可能与 stock 不同，结合音频时长分析吞吐。

```bash
python tools/qwen3_tts_ar_graph/service_bench.py \
  --input /softwarePlatform/y00899301/qwen3_tts_ar_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_ar_graph_commercial_kv_run01/service \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --stock-batch-policy record-stable \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend \
  --port 8091
```

各组日志位于 service/stock/server.log、service/kv_step/server.log、service/kv_frame/server.log。
各并发轮的 client.log、result.json 与 batch-before/after.json 保留在对应 c1/c4/c16/round0 目录。

```bash
cat /softwarePlatform/y00899301/qwen3_tts_ar_graph_commercial_kv_run01/service/report.md
```

## 6. 分别启动模型进行手动测试

三条命令分别运行，停止前一服务后再启动下一服务，避免占用同一 NPU/端口。
KV 服务预热捕获全部五个 batch bucket；最大活动请求数为 16。

### stock

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts

vllm-omni serve \
  /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_ar_stock.yaml \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --omni
```

### kv_step

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts

vllm-omni serve \
  /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_ar_kv_step.yaml \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --omni
```

### kv_frame

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts

vllm-omni serve \
  /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_ar_kv_frame.yaml \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --omni
```

## CPU 回归

```bash
python -m pytest -o addopts= --confcutdir=tests/model_executor/models/qwen3_tts tests/model_executor/models/qwen3_tts/test_cached_ar.py tests/model_executor/models/qwen3_tts/test_ar_graph_tools.py tests/model_executor/models/qwen3_tts/test_code_predictor_dtype.py -q
```

CPU 验证缓存和状态逻辑，模拟 FIA 参数与图管理；不能替代 NPU 算子数值和真实捕获检查。


## kv_frame 专用预计算优化

`kv_frame` 默认启用 `batched_gumbel_projected_embeddings_v1`，不改变 `stock` 和 `kv_step`。

- uniforms 的生成顺序、请求随机流和 padding 策略保持不变；整帧图内对 `[batch, 15, vocab]` 一次计算 `log(-log(u))`，各步沿用原 FP32 减法、temperature、Top-k 和 argmax。Greedy 不计算 Gumbel。
- 权重加载后、捕图前分块预计算 c1 至 c14 的 embedding＋projection（包含 bias），帧内改为查表。Talker hidden 和 c0 embedding 仍动态投影。c15 不再作为输入，无需投影表。
- 表由模型的所有 bucket、greedy/sampling 图和执行器共享，不写入 checkpoint；原 embedding 保留供 Talker 使用。仅用于冻结权重推理，修改权重后须重建执行器和图。
- 正式 BF16 CustomVoice 模型额外表内存为 `14 × 2048 × 1024 × 2 = 56 MiB`。五个 bucket 的 FP32 Gumbel 缓冲合计约 `3.63 MiB`（batch16 为 `1.875 MiB`），不包含图池及临时工作区。
- 预计算采用不同的 GEMM 形状，可能产生 BF16 舍入差异；现有逐位置/codes 检查仍会报告差异，不放宽标准。CPU 测试不能代替 NPU 数值和捕图验证。
- `bench` 的 manifest 和中文报告记录优化版本及实际额外缓冲字节数。所有测试仍可独立运行，无需正确性证书。
