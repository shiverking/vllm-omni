# keep10＋Code2Wav graph：英语端到端对照

分支基于78a08520c，仅移植keep10必要实现。四组依次运行：

|配置|码本|Code2Wav|Code Predictor主干/帧|
|---|---:|---|---:|
|baseline|16|原v0.30缓存路径|15|
|graph16|16|current有状态NPU graph|15|
|keep10|10|原v0.30缓存路径|9|
|keep10_graph|10|current有状态NPU graph|9|

所有组保持kv_frame、高效KV、预投影embedding、批量Gumbel。
不启用growth/stream KV、Triton SnakeBeta、constants/pooled/fused。
keep10保留c0..c9，传输仍16列，后6列是占位；Talker反馈和Codec解码均不计入占位贡献。
它会改变后续生成上下文、随机流和音频时长，不是与keep16数值等价的优化。

## 服务测试（可直接独立执行）

无需capture或正确性证书，端口8091须空闲，不用手动启动模型。
固定Vivian、英语、BF16、种子42；30条smoke文本循环120请求。
四组各保存30条试听WAV，然后并发1/4/16，每档预热3、正式120、1轮。
健康检查20秒，本机客户端绕过代理。Code2Wav最大batch4，Code Predictor buckets为1/2/4/8/16。
默认baseline两组保留原外层图，graph两组由current wrapper管理内层图。

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts

git fetch myfork codex/qwen3-tts-keep10-e2e
git switch codex/qwen3-tts-keep10-e2e
git pull --ff-only myfork codex/qwen3-tts-keep10-e2e

python tools/qwen3_tts_keep10_e2e/service_bench.py \
  --results /softwarePlatform/y00899301/qwen3_tts_keep10_e2e_run01/service \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 --requests 120 --concurrencies 1 4 16 --port 8091 \
  --vllm-repo /vllm-workspace/vllm --ascend-repo /vllm-workspace/vllm-ascend

cat /softwarePlatform/y00899301/qwen3_tts_keep10_e2e_run01/service/report.md
cat /softwarePlatform/y00899301/qwen3_tts_keep10_e2e_run01/service/status.json
```

每组`server.log`位于`service/<配置>/server.log`；客户端日志为`service/<配置>/c<并发>/round0/client.log`。
试听在各组`audio/`，index.json含文本、种子、时长、采样率和哈希。
report.md提供五组贡献对照，summary.json保留全部原始指标和遥测，deploy.yaml记录实际配置。
遥测包含实际batch和主干调用/图路由；周期快照可能滞后，不能当精确请求计数。
不设性能门槛；生成时长不同须结合音频吞吐与试听解释。重跑覆盖已管理文件，不递归删除目录。

## 独立同codes波形检查

只读复用已有Code2Wav采集；不要求keep10与keep16音频相同。
分别检查keep16/keep10的同形状eager与graph、缓存、块边界及拼接波形。
另记录graph与原生eager分组差异。覆盖batch1/2/3/4，并添加跨窗口及短块后继续序列。
BF16 atol=1e-5、rtol=0.016；首次数值差异保存后继续，执行错误记录后转下一码本配置。

```bash
python tools/qwen3_tts_keep10_e2e/check.py \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_keep10_e2e_run01/check \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --vllm-repo /vllm-workspace/vllm --ascend-repo /vllm-workspace/vllm-ascend
```

追加`--probe-only`只用前三条真实请求加合成边界序列。没有采集也可直接运行服务测试。

## 独立部署

四套配置为`vllm_omni/deploy/qwen3_tts_keep10_e2e_{baseline,graph16,keep10,keep10_graph}.yaml`。
例如单独运行最终组合（不要与自动服务脚本同时占用端口）：

```bash
vllm-omni serve /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_keep10_e2e_keep10_graph.yaml \
  --host 0.0.0.0 --port 8091 --trust-remote-code --omni
```

本地CPU测试不验证NPU捕获或音质。实验分支不自动迁回、不替换默认keep16。
