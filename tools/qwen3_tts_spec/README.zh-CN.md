# Qwen3-TTS CustomVoice 零训练树投机：中文运行说明

[English](README.md)

实验基线为 `v0.30.0`（`a8576ccb725c4e21cd13c3eb5f9a546b21149d2b`），分支为 `codex/qwen3-tts-zero-train-spec`，已替代此前的 rc1 基线。

当前状态：**实现完成，真实 NPU 性能待验证**。CPU 正确性测试不能证明 Ascend 上有加速。

本实验复用已有预测头构造稀疏候选树，采用“目标模型采样后验证树覆盖”的算法，并非论文残差重采样算法的逐项复现。不训练参数，不引入 KV cache，不调整播放缓冲策略。

## 1. 准备环境和目录

下面的命令在 **Linux Ascend NPU 服务器**执行，使用该版本对应的 vLLM-Omni、vLLM-Ascend、PyTorch、torch-npu 和 CANN 环境。先进入本仓库根目录。

固定实验条件：

- 模型：**Qwen3-TTS-12Hz-1.7B-CustomVoice**，不要使用 Base 或 VoiceDesign。
- 音色：Vivian；语言：英语；单请求。
- 默认采样：temperature 0.9、top-k 50、top-p 1.0。
- 数据：仓库自带的 30 条英语 `seed_tts_smoke` 文本，按原顺序前 10 条校准、后 20 条测试。

使用固定版本的本地模型快照，记录其不可变 revision。输出目录必须能被客户端和本机服务 worker 共同访问；当前采集流程不支持跨机器或 Ray 部署。

**在服务端和客户端两个终端分别设置以下变量**，替换为实际绝对路径：

```bash
MODEL=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
OUT=/data/tts-spec/run01
```

先初始化实验目录。将 `CHECKPOINT_COMMIT` 替换为模型快照的实际 commit 或不可变本地版本标识，将两个仓库路径替换为服务器上的位置：

```bash
python tools/qwen3_tts_spec/experiment.py collect --prepare \
  --model "$MODEL" --model-revision CHECKPOINT_COMMIT \
  --output "$OUT" --device npu:0 \
  --vllm-repo /workspace/vllm \
  --ascend-repo /workspace/vllm-ascend
```

这一步生成 `manifest.json`、`collect_config.json` 和 `deploy.yaml`。部署配置从当前 tag 的 YAML 派生，将序列容量设为 1、stage 0 的图捕获 batch sizes 设为 `[1]`，保留原有 Codec 分块设置。所有服务端对照实验使用同一份部署配置。

如果无法自动识别 CANN 版本，在 `collect --prepare` 和后续 `replay` 命令中加上 `--cann-version 实际版本号`。不要根据 torch-npu 版本推测 CANN 版本。

## 2. 运行原始 AR，采集帧输入

在**服务端终端**执行，保持服务运行：

```bash
bash tools/qwen3_tts_spec/serve_npu.sh "$MODEL" "$OUT" collect
```

等待服务就绪后，在**客户端终端**执行：

```bash
python tools/qwen3_tts_spec/experiment.py collect \
  --model "$MODEL" --output "$OUT" --device npu:0
```

默认连接 `http://127.0.0.1:8000`。如果更换端口，服务脚本的第四个参数和客户端 `--url` 必须对应，例如：

```bash
# 服务端终端
bash tools/qwen3_tts_spec/serve_npu.sh "$MODEL" "$OUT" collect 8091
# 客户端终端
python tools/qwen3_tts_spec/experiment.py collect \
  --model "$MODEL" --output "$OUT" --device npu:0 \
  --url http://127.0.0.1:8091
```

客户端按顺序发送 30 个请求。本地控制文件标识当前请求；worker 在**完成整帧 AR 推理后**保存帧输入和 codes，并仅导出一次 Code Predictor 权重及配置。启动预热不计入采集。

这一轮包含 CPU 拷贝和帧级文件写入，**采集请求的耗时不能作为性能证据**。热循环中没有逐 token 写文件。采集期间不要向这个服务发送其他请求；如果采集失败或中断，应换一个新输出目录重新开始，不能把多次采集混在一起。

采集结束后，在服务端终端停止服务，释放模型和设备内存，再进行离线回放。

## 3. 校准、冻结树、回放和生成报告

可以运行封装脚本：

```bash
bash tools/qwen3_tts_spec/replay_npu.sh "$OUT" npu:0
```

也可以逐步执行以下命令。两种方式选一种即可；已有冻结树时不要重复执行 `analyze`：

```bash
# 仅用前 10 条文本的帧进行校准并冻结树
python tools/qwen3_tts_spec/experiment.py analyze \
  --output "$OUT" --device npu:0

# 在后 20 条文本的帧上检查正确性，再执行性能测量
python tools/qwen3_tts_spec/experiment.py replay \
  --output "$OUT" --device npu:0

# 汇总已有产物
python tools/qwen3_tts_spec/experiment.py report \
  --output "$OUT" --device npu:0
```

`analyze` 根据采集的 AR codes 重算校准前缀的隐藏状态，保存校准状态，生成候选链以及 8、16、32 节点树。树配置冻结后带有摘要校验，不能使用测试集结果调树；工具会拒绝覆盖已有冻结配置。

`replay` 先执行两类正确性检查：

- 在全部测试帧上，分别使用 greedy 和共同 uniforms，对比原版 AR 与实验路径的完整 codes。
- 将每个访问到的树节点的下一预测头 logits，与同一前缀逐条串行计算的结果对照。

logits 对照容差为：FP32 下 `atol=rtol=1e-5`，低精度下为 `2e-2`。**不论 logits 容差如何，两种采样模式的 codes 都必须完全一致**。发现差异会保存诊断信息并停止性能测试，不能通过放宽 codes 要求来忽略数值问题。

正确性通过后，性能测量按以下规则执行：

1. 每种配置预热 20 帧。
2. 每组对照按 AR → 实验 → 实验 → AR（ABBA）执行四轮，每轮至少 200 帧。
3. 测试帧不足时循环回放，同时记录独立帧数和总执行次数。
4. 每种树分别对照**原版优化 AR**和**使用相同显式 attention 后端的参考 AR**。
5. 输入搬运和 uniforms 构造在帧计时之外完成；帧级墙钟计时等待设备完成，另用设备事件记录时间。
6. 详细事件和分阶段计时在单独诊断轮运行，正式性能轮关闭这些详细采集。

可以额外使用 `replay --greedy` 测量 greedy 性能，它仍会执行两种模式的正确性检查。该命令会替换回放计时产物；如需保留随机采样结果，先单独备份其产物。

## 4. 正确性通过后，测量完整流式请求

为每个配置重新启动干净的服务，先停掉上一个服务。**完整请求性能轮必须关闭详细采集**。

先运行原版 AR。在服务端终端执行：

```bash
bash tools/qwen3_tts_spec/serve_npu.sh "$MODEL" "$OUT" stock
```

服务就绪后，在客户端终端执行：

```bash
python tools/qwen3_tts_spec/experiment.py collect --e2e-label stock \
  --model "$MODEL" --output "$OUT" --device npu:0
```

完成后停止原版服务。以冻结的 `tree16` 配置为例，在服务端终端启动：

```bash
bash tools/qwen3_tts_spec/serve_npu.sh "$MODEL" "$OUT" tree16
```

然后在客户端终端执行：

```bash
python tools/qwen3_tts_spec/experiment.py collect --e2e-label tree16 \
  --model "$MODEL" --output "$OUT" --device npu:0

python tools/qwen3_tts_spec/experiment.py report \
  --output "$OUT" --device npu:0
```

也可以将两处 `tree16` 同时替换为 `chain`、`tree8` 或 `tree32`。**客户端标签必须与服务端实际模式一致**。

这一步执行后 20 条文本的完整请求，不是回放固定的 AR 历史。指标含义如下：

| 指标 | 当前测量方式 |
|---|---|
| TTFP | 客户端收到第一段非空 PCM 字节的时间 |
| E2E | 从请求开始到响应完成的时间 |
| RTF | E2E 除以音频时长；按单声道、24 kHz、16-bit PCM 计算时长 |
| 错误 | 请求异常或空 PCM 响应 |

这些指标衡量音频送达客户端的情况，不代表扬声器实际播放时间。两组测量都包含连接和启动影响；当前阶段不修改播放或缓冲策略。

## 5. 查看和返回哪些产物

| 文件 | 内容 |
|---|---|
| `manifest.json` | 基线和三个仓库版本、模型 revision、设备与软件版本、数据划分、种子及源码摘要 |
| `capture_metadata.json` | 采集 worker 导出的 dtype、采样参数及预测器配置 |
| `predictor.pt` | 导出的 Code Predictor 权重、配置和 dtype |
| `frames/*.pt` | 每帧输入、原版 AR codes、采样参数、请求／帧／数据划分标识 |
| `calibration_states/*.pt` | 校准前缀的隐藏状态 |
| `trees.json`、`coverage.json` | 冻结树配置与仅基于校准数据计算的覆盖潜力 |
| `events.jsonl` | 每轮起始位置、命中数、提交数、节点数、主干调用次数和退出原因 |
| `diagnostics.jsonl` | 单独诊断轮的分阶段设备／主机时间，以及节点 logits 误差 |
| `timings.jsonl` | 整帧配对墙钟／设备时间、对照配置和实际 attention 后端 |
| `correctness.json` | 正确性检查结果及差异信息 |
| `replay.json` | 实际回放环境、dtype、图捕获 bucket、内存峰值和执行次数 |
| `requests-*.jsonl` | 各配置完整请求的流式指标；`requests-capture.jsonl` 不用于性能验收 |
| `summary.json`、`report.md` | 汇总指标和阶段结论 |

`processed_positions` 和 `attention_pairs` 是主干计算量的代理指标。`estimated_matmul_flops` 是主干、attention、预测头和投影层矩阵乘法的理论运算量之和，不包括逐元素算子和硬件补齐开销，**不是实测硬件 FLOPs**。真实耗时包含预测头和 embedding 等开销。

优先返回：manifest、冻结树、正确性结果、events、diagnostics、timings、replay 元数据、完整请求指标，以及最终报告。tensor 分片可先保留在服务器供定位问题使用，预测器权重文件可能较大。

## 6. 如何判断是否可以继续研究

Bootstrap 以**请求**为重采样单位，将重复执行的帧和配对测量保留在一起；重采样 5,000 次，计算 95% 区间，避免把重复回放当成新的独立样本。

一个固定配置通过本阶段工程门槛，需要同时满足：

- 正确性检查通过。
- 相对原版 AR 的帧级墙钟加速至少为 **1.10×**。
- 配对 bootstrap 的 95% 区间下界大于 1。
- 完整请求无错误；平均 E2E、平均 RTF 和 p95 TTFP 的增长均不超过 **10%**。这是当前对“无明显性能回退”的具体判定。

缺少完整请求证据不能判为通过。只减少主干调用次数、但没有真实提速时，报告为机制有潜力、执行尚未加速；覆盖收益不足时，保留证据供后续训练研究使用。参考 attention AR 对照用于区分算法收益和后端差异。

30 条文本只支持初步可行性判断，不作为论文级泛化或生产性能证据。

## 7. 实现细节与实验边界

- 不设置 `VLLM_OMNI_TTS_SPEC_CONFIG` 即使用原有 AR。只有 Qwen3-TTS wrapper 读取此开关，Qwen3-Omni 不受影响。
- 实验模式明确拒绝大于 1 的 batch，包括预热；关闭外层 MTP 图捕获。原版 AR 回放保留 tag 的预测器内部图机制。
- draft 前向重算已提交前缀，把最后的隐藏状态送入未来最多三个码本的原有 Head，每个位置取 top-4。
- 树节点 embedding、输出 Head 和 RoPE 均按真实码本／序列位置选择，不使用打包索引代替真实位置。
- 树验证采用显式 FP32 attention 矩阵乘法、带 mask 的 softmax 和 value 矩阵乘法。NPU norm／RoPE 仍沿用原有算子。候选节点只能看到已提交前缀、祖先和自身，不能看到兄弟分支。
- 不使用压缩三角因果 mask 验证树。只缓存固定位置／mask 元数据，不缓存隐藏状态或 KV。当前树路径为 eager，不宣称已完成固定形状树图捕获。
- 候选树不读取本轮目标随机数。按原目标分布进行 Gumbel 采样；命中则沿树继续，未命中则提交该目标 token 并结束本轮，叶节点可以额外提交一个目标 token。
- 每个树轮次有一次提交长度从设备到主机的传输，该同步计入耗时；draft 和 verify 的主干调用都计入统计。
- 回放 uniforms 由 CPU 按“种子、请求 ID、帧、码本位置”生成，两条路径使用同一份数据。完整请求沿用 runner 的请求级 uniforms，或从其请求级 generator 一次抽取固定大小的整帧随机数。
- 校准按“完整排名前缀的覆盖增益”贪心增加节点，保证祖先闭合；并列时按深度和排名字典序选择。负排名表示目标 token 不在 top-4 中。候选链固定为 `(0)`、`(0,0)`、`(0,0,0)`。
- 暂不支持 top-p 非 1 的实验采样，不训练偏移向量、不引入 KV 提交／回退、不调整流式播放调度。

## 8. 本地 CPU 正确性检查

在装有 PyTorch、Transformers、pytest、pytest-mock、NumPy 和 PyYAML 的环境中，从仓库根目录执行：

```bash
python -m pytest --noconftest -o addopts='' \
  tests/model_executor/models/qwen3_tts/test_code_predictor_dtype.py \
  tests/model_executor/models/qwen3_tts/test_code_predictor_spec.py -q
```

测试通过直接加载模块并替代 vLLM 运行时依赖，无需加速卡。合成模型集成测试覆盖实际导出／重载、校准、ABBA 回放、报告生成，以及正确性失败后废止旧性能结果。其耗时不能当作真实模型性能结果。
