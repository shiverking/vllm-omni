# Code2Wav Transformer 增量 KV 实验

实验分支 `codex/qwen3-tts-code2wav-incremental-kv`，从 `78a08520c` 建立。
Code Predictor 固定 kv_frame；Code2Wav 不叠加 constants/pooled/fused。
默认部署行为不变。所有命令独立，不读取正确性证书，也不自动迁回原分支。

|模式|增长阶段|滚动阶段|正确性对照|
|---|---|---|---|
|current|现有窗口重算|现有窗口重算|原有 NPU graph|
|growth_kv|仅计算新增位置|丢弃 Transformer KV，恢复原窗口重算|current|
|stream_kv|仅计算新增位置|连续卷积、绝对位置、滑窗 KV|连续全历史重算；与 current 的差异另外记录|

窗口72包含自身，每层保留最多71个历史 K/V，另保留12帧 Transformer 输出供波形网络使用。
KV、量化、卷积和输出上下文按请求保存；无参考音频，仅支持每块1..25帧、最多16活动请求。
batch 3填充到4，同阶段超过4拆组。取消、结束和异常清理请求状态。
`growth_kv` 增量阶段为首块及 1→26、26→51、51→76；之后复用原滚动算法。
`stream_kv` 不重算窗口左边界，因此可能与原窗口重算不同，不宣称音质等价。

## 执行和图

新增内部配置 `code2wav_transformer_execution: current|growth_kv|stream_kv`，默认 current；
`code2wav_incremental_execution: eager|graph` 默认 graph。
保持原 attention 后端、投影权重、RoPE/RMSNorm/SnakeBeta 和分块策略。
无 Speech API 改动。实验要求 Stage 1 外层图关闭，由内部 Code2Wav 管理图。

正常配置下 growth_kv 捕获12张增量图，另外保留 current 的15张图供滚动路由；
stream_kv 捕获15张增量图。实际图数、后端、捕获耗时写入 graph 元数据。
绝对位置是每轮更新的图输入，不固定为捕获时的位置。
短块及非标准增长历史走显式 eager，达到成熟窗口后可重新命中成熟图。
图捕获失败报错，不静默切换执行模式。图缓冲仅允许单个有序 NPU stream 提交。

## 更新代码

在服务器仓库没有未提交冲突的情况下切换实验分支；不要强制覆盖工作区：

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts
git fetch myfork codex/qwen3-tts-code2wav-incremental-kv
git switch codex/qwen3-tts-code2wav-incremental-kv
git pull --ff-only myfork codex/qwen3-tts-code2wav-incremental-kv
```

首次本地没有该分支时，将 `git switch` 替换为：

```bash
git switch --track -c codex/qwen3-tts-code2wav-incremental-kv myfork/codex/qwen3-tts-code2wav-incremental-kv
```

## 1. NPU 小检查

默认比较 eager 和 graph，前三条采集请求加确定性跨窗口长序列，覆盖batch1/2/3/4。
history/codes关联来自原capture，不重新生成音频。

```bash
python tools/qwen3_tts_code2wav_incremental_kv/experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01/probe \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --variants current growth_kv stream_kv \
  --executions eager graph \
  --batches 1 2 3 4 \
  --probe-only \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

## 2. 全部30条分块检查及音质产物

```bash
python tools/qwen3_tts_code2wav_incremental_kv/experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01/check \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --variants current growth_kv stream_kv \
  --executions eager graph \
  --batches 1 2 3 4 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

检查逐块波形、开头边界、拼接波形、历史状态、graph/eager和连续参考各层K/V。
stock批处理相对独立执行差异单列；不自动放宽阈值。
FP32 atol=1e-5/rtol=1e-4；BF16 atol=1e-5/rtol=0.016（torch默认），产物记录实际参数。
首个失败保存 `failure-模式-执行.pt`：输入、前后状态、波形、模块诊断与图信息。
数值差异仅记录，不停止当前配置、后续块或音频导出，也不在结束时抛出校验异常。
每配置只保存首份详细差异，所有差异统计保存在 correctness.json。
`complete=true` 表示全部执行完成，`passed=false` 仍表示存在差异；二者独立。
真正的算子／捕获等运行异常记录在 errors 中，无法执行的配置跳过，继续其他配置。

`audio/` 保存batch1、最后所选执行方式的成对PCM16 WAV。
`listening.json` 是盲听清单；`listening-key.json` 为解盲键、波形差异、边界跳变、RMS、峰值和时长。
波形数学指标不能自动证明音质等价。stream_kv 迁回须用户听感确认。

## 3. 独立 ABBA 计时及六批 profiling

默认命令只计时 graph；如需 eager 消融，将 executions 改为 `eager graph`。
current 始终作为 graph 对照。每候选分别做current→候选→候选→current四轮。

```bash
python tools/qwen3_tts_code2wav_incremental_kv/experiment.py bench \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01/bench \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --variants current growth_kv stream_kv \
  --executions graph \
  --batches 1 2 3 4 \
  --warmup 20 \
  --executions-per-round 60 \
  --profile \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

每份请求历史先由对应执行器生成快照，快照恢复在计时外；输入暂存、状态搬入、replay、提交和输出复制在计时内。
按完整请求的阶段调用次数加权估算总体耗时，包含滚动与结束短块，不只展示增长阶段。
该加权结果是抽样估算，非完整服务端到端实测；p50/p95是实际抽样调用分位数。
`timings.jsonl` 保存KV搬运字节、实际有效位置数和padding位置数；图统计另存。
设备事件含提交间隙，不是纯kernel时间。内存峰值包含同时驻留的图与历史快照。
`bench/profile/` 的六批trace单独采集，历史准备不进入trace或正式时间。
bench不读取任何正确性结果，性能结果不代表功能通过。

## 4. 服务对比

脚本自行依次启动三组服务；先停止同NPU或8091端口上的其他服务。
Code Predictor一直为kv_frame；并发1/4/16，每档预热3、正式120、1轮。
每20秒health检查，本机HTTP绕过代理；服务及客户端日志分开，显示进度和ETA。

```bash
python tools/qwen3_tts_code2wav_incremental_kv/experiment.py service \
  --results /softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01/service \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --variants current growth_kv stream_kv \
  --concurrencies 1 4 16 \
  --requests 120 \
  --port 8091 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

运行前后的遥测快照可能滞后5秒，用于路由诊断，不作精确请求计数。
取消后再生成的隔离检查独立记录。服务重复运行覆盖自身管理结果，不删除capture。
同codes独立计时是局部收益依据，不能用音频生成时长不同直接推断局部收益。

## 5. 查看结果及单独启动

```bash
cat /softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01/check/report.md
cat /softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01/bench/report.md
cat /softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01/service/report.md

python tools/qwen3_tts_code2wav_incremental_kv/experiment.py report \
  --results /softwarePlatform/y00899301/qwen3_tts_code2wav_incremental_kv_run01/bench

vllm-omni serve \
  /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_kv_frame_code2wav_growth_kv.yaml \
  --host 0.0.0.0 --port 8091 --trust-remote-code --omni
```

单独测试stream_kv时先停止前一个服务，将deploy文件换为同目录
`qwen3_tts_kv_frame_code2wav_stream_kv.yaml`；current沿用`qwen3_tts_kv_frame_code2wav_current.yaml`。

## 本地与迁回边界

CPU：`python tools/qwen3_tts_code2wav_incremental_kv/test_cpu.py`。
使用真实小型decoder和非零码表；CPU图替代路径与模拟计时只验证工具，不证明NPU捕获或速度。
原Code2Wav回归：`python tools/qwen3_tts_code2wav_graph/test_cpu.py`。

growth_kv数值检查、隔离和服务无新增错误后，根据真实收益选择迁回；stream_kv另需音质确认。
仅cherry-pick选择的实现与测试提交，不自动合并整分支，不改变默认current。
