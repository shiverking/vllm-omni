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
