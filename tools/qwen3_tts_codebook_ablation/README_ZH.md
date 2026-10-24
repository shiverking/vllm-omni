# 尾部码本删减试听

复用码本统计实验已经生成的真实 codes，不重新运行 Talker/Code Predictor。
同一请求导出保留 16、14、12、8、4、1 个码本的 WAV。16 为重解码对照，
另复制原服务音频。仅在量化向量求和前截断码本，绝不将删掉的码填成 token 0。
每个版本重新初始化缓存，使用首块 1 帧、后续 25 帧的 eager 解码；不启用 Code2Wav graph。

默认取种子 42、按请求 ID 排序的前 20 条独立文本。结果可重复覆盖，原采集文件只读。
各删减版本使用同一增益；仅在超范围时对该请求所有版本一起衰减，增益记入 metadata.json。
原服务音频原样保存，可能与重解码版本因执行路径不同而有差异；听感比较优先用 keep16 作对照。
不执行逐文件响度归一化，以免掩盖删减造成的幅度变化。WAV 为单声道 PCM16。

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts
python tools/qwen3_tts_codebook_ablation/test_cpu.py

python tools/qwen3_tts_codebook_ablation/experiment.py \
  --input /softwarePlatform/y00899301/qwen3_tts_codebook_stats_run01 \
  --results /softwarePlatform/y00899301/qwen3_tts_codebook_ablation_run01 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --seed 42 --limit 20 --keep 16 14 12 8 4 1
```

`--limit 0` 使用该种子全部文本；`--request-ids smoke001-s42 smoke002-s42` 精确选择请求。
`--keep 16 15 14 13 12` 可以进一步细查尾部删减。始终自动附带 keep16 对照。
输入需包含原统计工具的 completed、codes、audio 三个目录，不需要正确性证书。

下载整个输出目录后打开 listen.html，可直接播放各版本；目录结构：

```text
listen.html
manifest.json
smoke001-s42/original.wav
smoke001-s42/keep16.wav
smoke001-s42/keep14.wav
smoke001-s42/keep12.wav
smoke001-s42/keep08.wav
smoke001-s42/keep04.wav
smoke001-s42/keep01.wav
smoke001-s42/metadata.json
```

也可在服务器启动静态文件服务（测试完 Ctrl+C 停止），在浏览器访问服务器 IP 的 8092 端口：

```bash
python -m http.server 8092 --bind 0.0.0.0 \
  --directory /softwarePlatform/y00899301/qwen3_tts_codebook_ablation_run01
```

建议同时听短句、数字、齿音、气声、句尾和长句的分块边界。
先关注 keep16 重解码是否正常，再比较其他版本是否有发闷、失真、音色变化或辅音丢失。
模型没有因本实验重新训练；该实验只评估听感，不测实际提前结束 AR 的加速，
也不据此认定商业可接受性或人耳无差异。本地 CPU 测试不替代真实 NPU 解码验证。

## 中文试听

新增 12 条中文自编文本，覆盖数字、齿音、绕口令、问句、情绪与长句。
先生成真实中文音频并采集实际 Code2Wav 输入，再执行相同删减解码。
生成脚本自动启动及关闭 kv_frame 基线服务，Vivian、Chinese、种子 42；
服务占用 8091，运行前停止自己手动启动的同端口服务。健康检查每 20 秒，HTTP 请求绕过代理。
已完整生成的请求重跑时跳过；--limit 3 可先生成三条，之后改为 12 补齐。

```bash
python tools/qwen3_tts_codebook_ablation/generate_zh.py \
  --results /softwarePlatform/y00899301/qwen3_tts_codebook_ablation_zh_capture01 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 --limit 12 --seed 42 --port 8091

python tools/qwen3_tts_codebook_ablation/experiment.py \
  --input /softwarePlatform/y00899301/qwen3_tts_codebook_ablation_zh_capture01 \
  --results /softwarePlatform/y00899301/qwen3_tts_codebook_ablation_zh_listen01 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 --seed 42 --limit 12 --keep 16 14 12 8 4 1

python -m http.server 8092 --bind 0.0.0.0 \
  --directory /softwarePlatform/y00899301/qwen3_tts_codebook_ablation_zh_listen01
```

访问服务器 IP 的 8092 端口 /listen.html。生成日志保存在采集目录 server/server.log。


## 真正提前结束生成：keep10 vs full16

内部部署字段 `code_predictor_keep_codebooks: 10` 仅支持 kv_frame。
full16 每帧执行 15 次主干；keep10 生成 c0～c9，每帧执行 9 次主干。
两组均按 batch 1/2/4/8/16 捕获整帧图。传输保持16列，末6列只是零占位；
Talker 的下一帧 embedding 求和与 Codec 的量化向量求和均排除这些占位。
这会改变后续 Talker 上下文及随机流消耗，因此真实音频可能与离线删减不同，必须重新试听。

### 中文完整服务性能（无需采集或证书）

```bash
python tools/qwen3_tts_codebook_ablation/service_bench.py \
  --results /softwarePlatform/y00899301/qwen3_tts_keep10_perf_run01/service \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 --requests 120 --concurrencies 1 4 16 --port 8091 \
  --vllm-repo /vllm-workspace/vllm --ascend-repo /vllm-workspace/vllm-ascend
```

依次启动 full16、keep10，Code2Wav 与 Talker 设置相同。每档预热3请求、正式120请求、1轮。
正式计时前分别保存12条真实中文试听音频到 full16/audio 与 keep10/audio。
两组 server.log 分别保留；report.md、summary.json 包含 TTFP、E2E、RTF、吞吐、音频总时长和错误率。
若时长明显改变，应结合音频秒吞吐与听感判断，不能将少说话算作加速。
重新运行覆盖本工具产物，不删除原采集数据。

### 固定帧性能（可选，使用已有 AR capture）

```bash
python tools/qwen3_tts_codebook_ablation/frame_bench.py \
  --input /softwarePlatform/y00899301/qwen3_tts_ar_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_keep10_perf_run01/frame \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 --warmup 20 --executions-per-round 60 --batches 1 2 4 8 16
```

采用原60帧、greedy及共同uniforms随机模式、ABBA四轮。
同帧前10码差异记录于 prefix_differences.json；不阻止独立性能测量。
计时包括输入复制和相同的整帧uniforms构造；该口径与服务实际只生成所需随机数不同。
无AR采集产物可直接做服务测试，不需要为服务测试补做帧级测试。

### 单独启动服务（两套分别运行）

```bash
vllm-omni serve /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_ar_kv_frame_keep16.yaml \
  --host 0.0.0.0 --port 8091 --trust-remote-code --omni
```

```bash
vllm-omni serve /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_ar_kv_frame_keep10.yaml \
  --host 0.0.0.0 --port 8091 --trust-remote-code --omni
```
