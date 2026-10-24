# kv_frame 基线上的 Code2Wav 缓存＋NPU graph

两组始终使用当前 kv_frame（包括批量 Gumbel、预投影 embedding）。
只比较 Code2Wav：baseline 保留 v0.30 缓存与原部署的外层图配置；npu_graph 保留缓存算法，
关闭 Stage 1 外层捕获，由 decoder 管理内层图。Stage 0 配置相同。

- Code2Wav 最大 batch=4，捕获 [1,2,4]；Code Predictor 仍捕获 [1,2,4,8,16]。
- 固定首块1帧、后续25帧、默认窗口72；默认捕获15张 Code2Wav 图。
- 只支持无参考前缀 CustomVoice；不支持 ICL、chunk ramp 或自适应 chunk。
- 短块不补齐时间轴：使用有状态 eager 更新真实缓存，后续可达时恢复 graph。
- 捕获失败明确报错，保存阶段信息；运行时 eager 回退计入报告。
- 所有命令独立，不读取正确性证书、不要求前一项通过。
- collect 的 codes 是 Code2Wav 实际输入，不能使用旧 Code Predictor 单帧采集替代。

## 本地 CPU 测试

```bash
python tools/qwen3_tts_code2wav_graph/test_cpu.py
```

使用真实小型 decoder，图 replay 由同布局 eager 模拟，覆盖缓存、分桶、短块续接、
换序、隔离、输出生命周期、采集续跑及报告。不能证明 NPU 捕获成功或加速。

## 服务器命令

先进入仓库。所有新产物集中在同一个根目录，不依赖旧实验目录。

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts
```

### 1. 采集真实音频和 codes

脚本自行启动一次 baseline 服务并关闭。默认30条英语文本，Vivian、seed42、单请求。
可以先加 `--limit 3`；之后执行不带 limit 的同一命令补齐，其余完整请求会跳过。
采集期间单独关闭 Stage 1 外层捕获，以确保 Python 记录钩子执行；正式 baseline 服务配置不变。

```bash
python tools/qwen3_tts_code2wav_graph/experiment.py collect \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend \
  --port 8091
```

输出 `audio/*.wav`、`chunks/*.pt`、`requests.json`、`manifest.json`。
每个 chunk 保存内部/外部请求 ID、顺序、codes、基线波形和结束标记；
scheduler 的结束回调另存 `chunks/finished-*.jsonl`。
续跑失败请求的旧孤立 chunk 不参与分析，清单仅引用本次成功产物。

### 2. NPU 小检查

前三条已完成请求，另加固定随机 codes 检查跨窗口、短块和续接。覆盖 batch 1/2/3/4。

```bash
python tools/qwen3_tts_code2wav_graph/experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/probe \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --probe-only \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

### 3. 实际采集请求检查

```bash
python tools/qwen3_tts_code2wav_graph/experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/check \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

检查同形状 eager 与 graph 的逐块波形、拼接、边界和缓存。
BF16 默认 atol=1e-5、rtol=0.016，结果中记录确切值。
原生 batch 对照和 batch/single 差异单独列出；不能因同形状检查通过就宣称跨 batch 完全一致。
首个失败保存 `failure.pt` 和 `correctness.json`；捕获失败保存 `capture_failure.json`。

### 4. 单独性能与 profiling

每组预热20次，baseline→graph→graph→baseline四轮，每轮60次。
`--batches 1 2 3 4` 中3用于填充路径。
从完整采集分块中等间距选择固定作业；历史状态准备在计时外，输入暂存、计算、状态交接在计时内。
直接 decoder baseline 是现有有状态 eager 路径；原服务外层图的收益由下一项服务测试衡量。

```bash
python tools/qwen3_tts_code2wav_graph/experiment.py bench \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/bench \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --batches 1 2 3 4 \
  --warmup 20 \
  --executions-per-round 60 \
  --profile \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

`--profile` 在正式计时结束后独立采六个固定批次，不污染计时汇总。
trace 和算子计数写入根目录的 `profile/`，缓存暂存与输出交接有单独标记。
设备事件不是纯 kernel 时间。内存峰值包括模型、全部驻留图和实验状态。

仅重新生成报告：

```bash
python tools/qwen3_tts_code2wav_graph/experiment.py report \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/bench
```

### 5. 两组完整服务性能

无需 capture 或 correctness 输入。自动按 baseline→code2wav_graph 启停服务。
每组并发1/4/16，每档预热3请求、正式120请求、1轮。健康检查20秒一次，请求绕过代理。
取消请求检查只记录诊断，不拦截性能测试。

```bash
python tools/qwen3_tts_code2wav_graph/service_bench.py \
  --results /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/service \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --requests 120 \
  --concurrencies 1 4 16 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend \
  --port 8091
```

中断后可重复执行同一命令，覆盖本轮管理的结果；不递归删除 NFS 目录。
服务日志：`service/baseline/server.log`、`service/code2wav_graph/server.log`。
客户端日志：对应配置的 `c1/round0/client.log`、`c4/round0/client.log`、`c16/round0/client.log`。
`telemetry/` 保存 Code2Wav 分组和图路由；并发16不代表 Code2Wav batch16。
端口已有服务时不会杀死外部进程，请先停止手动启动的服务。

查看结果：

```bash
cat /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/bench/report.md
cat /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/service/report.md
```

## 手动启动

两条命令分别运行，同一端口上的上一套服务停止后再运行下一套。

baseline：

```bash
vllm-omni serve /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_kv_frame_code2wav_baseline.yaml \
  --host 0.0.0.0 --port 8091 --trust-remote-code --omni
```

Code2Wav 有状态 NPU graph：

```bash
vllm-omni serve /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_kv_frame_code2wav_npu_graph.yaml \
  --host 0.0.0.0 --port 8091 --trust-remote-code --omni
```

本地验证不代表 NPU 捕获、数值一致性或性能通过。没有性能门槛，也不将不同实验的加速比相乘。
