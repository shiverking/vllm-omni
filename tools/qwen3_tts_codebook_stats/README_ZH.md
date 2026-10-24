# 真实生成音频：码本分布与候选覆盖实验

分支 `codex/qwen3-tts-codebook-statistics` 从 `13350ff06` 独立建立。只做 stock、BF16、单请求 CustomVoice 1.7B 的真实音频生成与离线分析，不修改 KV 实现、不训练、不开展投机验证或性能对照。

## 固定数据与分析口径

- Vivian、英语；100 条唯一文本（原有 30 条 smoke + 新编写 70 条）× 种子 42/2026/3407 = 300 段 WAV。文本来源见 `texts.jsonl`。这是真实模型生成音频，不是录制的人类语料。
- SHA256 固定顺序划分 70 条统计文本、30 条测试文本，同一文本的全部种子始终属于同一集合。清单由 prepare 固定后保留。生成按种子优先、文本顺序执行，短跑也会覆盖多条文本。
- stock 正常 AR 生成，残差 temperature=0.9、top-k=50、top-p=1.0。manifest 记录完整 Talker/残差采样配置，码本记录保存实际残差参数。
- 采集 hook 在 Code Predictor 输出处，仅在研究开关启用时记录原始 16 个整数；采集开关关闭外层 MTP 图。原有 stock 内部优化保留。过滤规则与 Talker/Codec 一致：越界/负码、全零帧不参与统计；原始记录保留。
- 不保存模型权重、hidden 或 logits。同步轻量文件写入仅用于研究采集，不能把生成耗时当作生产性能。
- 帧与 WAV 按请求和生成顺序对应，不代表逐音素或采样点对齐；不同种子的帧数可以不同，不按帧号跨种子对齐。

## 服务器命令

在安装好当前分支及 vLLM、vLLM Ascend 的 Linux NPU 环境运行。所有输出集中在一个目录，重新执行 generate 会自动续跑完整清单，不删除已完成的 WAV。

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts

python -m pip install -r tools/qwen3_tts_codebook_stats/requirements.txt

python tools/qwen3_tts_codebook_stats/experiment.py prepare \
  --results /softwarePlatform/y00899301/qwen3_tts_codebook_stats_run01 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend
```

可选：先生成 6 段，检查音频文件和码本记录。此命令自动启动并关闭服务，不需要另开终端手工启动模型。

```bash
python tools/qwen3_tts_codebook_stats/experiment.py generate \
  --results /softwarePlatform/y00899301/qwen3_tts_codebook_stats_run01 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend \
  --port 8091 \
  --limit 6
```

生成全部剩余音频（也可直接运行此命令，无需先短跑）：

```bash
python tools/qwen3_tts_codebook_stats/experiment.py generate \
  --results /softwarePlatform/y00899301/qwen3_tts_codebook_stats_run01 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend \
  --port 8091
```

- 本地 health 和音频请求均通过私有 opener 绕过代理，不修改全局代理。health 每 20 秒检查一次；默认启动超时 1800 秒、单请求超时 600 秒，可用 `--startup-timeout` / `--request-timeout` 修改。
- 一次只管理一个本工具启动的服务；端口被占用时应停止已有服务或改 `--port`。同一输出目录的生成任务使用进程锁防止串扰。
- `server.log` 追加保存各次启动日志，`launch-*.json` 保存启动命令与环境，`progress.log` 输出每个请求和 ETA。
- 失败后保存 `failures/<request>.json` 并停止本次生成/关闭自有服务，避免超时请求继续运行造成归属错误。重新运行相同命令会替换该请求的未完成记录并继续。成功项以 WAV、码本文件和完成标记一致为准；这属于数据完整性，不是正确性证书或性能门槛。
- generate 使用 prepare 中冻结的模型与设备部署；修改实验配置时使用新的结果目录，不混合不同模型/采样的数据。

CPU 离线分析，支持只完成部分音频时运行：

```bash
python tools/qwen3_tts_codebook_stats/experiment.py analyze \
  --results /softwarePlatform/y00899301/qwen3_tts_codebook_stats_run01

cat /softwarePlatform/y00899301/qwen3_tts_codebook_stats_run01/report.md
```

只从已有分析产物重新渲染报告：

```bash
python tools/qwen3_tts_codebook_stats/experiment.py report \
  --results /softwarePlatform/y00899301/qwen3_tts_codebook_stats_run01
```

各命令不读取正确性证书。prepare 提供数据清单，generate 提供真实采集产物，analyze 消费完整产物，report 只消费已有分析汇总；不存在通过某项测试才能运行下一项的拦截。没有完整统计集或测试集音频时，仍输出已有分布并标注预测覆盖不可用。

## 统计与候选器

经验频率、有效词条数、熵（bits）、Top-1/4/8/16 集中度、码本对互信息分别按统计集和测试集汇总。同时提供请求平均；相邻帧统计只使用原始帧号连续的有效帧，不跨请求/被过滤帧间隙。

五种候选器：global（目标位置全局频率）、main（条件 c0）、previous（前一码本）、mixed（两条件分布等权）、temporal（上一有效帧同位置码）。只用统计集拟合：全局加一平滑，条件分布加 32 个全局先验伪样本，未见条件回退全局；参数固定，不使用测试集调整。

Top-1/4/8 覆盖率和 NLL（自然对数 nats）只评价生成的测试 token，不能视为目标模型的概率质量。时间先验增益使用同一批可用相邻帧的 global 对照。支持度分为 unseen=0、sparse=1..31、frequent>=32；mixed 取两个条件样本量的最小值，unseen 意味至少一个条件未见。

前四种候选器构建深度最多 3、各位置 Top-4、预算 8/16/32 的树。用分支联合概率优先扩展，父节点先加入；子分支基于假设前驱码，不能访问真实后续码。树预算包含所有候选节点，不包含已知前缀；帧尾裁剪。命中>=2占比分母包含帧尾全部起点，完整路径覆盖按该起点剩余深度定义。

1000 次 bootstrap 按文本分组，保留所有种子和帧。候选器逐位置筛选有多重比较，报告只提出后续假设，不将区间当作普遍显著性证明。候选覆盖不是实际接受率、功能正确性或加速结论。

## 产物

- `audio/`、`codes/`、`completed/`：WAV、逐帧 JSONL、关联清单和哈希。
- `manifest.json`、`deploy.yaml`、`launch-*.json`、`server.log`、`progress.log`：清单、配置、环境和日志。
- `analysis/summary.json`：分布、逐位置/总体覆盖、区间、按请求均值及样本支持度。
- `analysis/frequencies_*.csv`、`conditional_counts.csv`、`condition_support.csv`：频率和稀疏条件表（未观察到的转移不展开为巨大稠密表）。
- `analysis/coverage*.csv`、`tree*.csv`：总体及逐请求覆盖结果。
- `analysis/examples.json`、`analysis/*.png`、`report.md`：具体命中/失败序列、静态图和中文报告。

本地 CPU 测试不代表真实 NPU 生成已经完成。本实验只覆盖 Vivian、英语、当前采样参数；尤其应结合条件支持度判断规律是否只是小样本现象。
