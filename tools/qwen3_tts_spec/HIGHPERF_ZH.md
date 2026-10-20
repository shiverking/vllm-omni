# 树投机高性能实验运行说明

该入口复用已有采集数据，不需要重新合成音频，也不执行全量 replay。
固定使用原来的 10 条校准请求与 20 条测试请求；默认每条测试请求取首、中、末三帧。
训练、残差重采样和播放调度不在本次实验内。

## 数据流与对照

首次 prefill 产生 c₁；下一轮把已输出但尚未处理的尾 token 作为确定根节点，
与未来最多三个位置的候选一起执行一次主干前向。只提交已处理路径的 KV；
失配或叶后目标 token 等待下一轮处理。帧末不额外处理最后一个已输出 token。
所有权重冻结，目标采样沿用 temperature=0.9、top-k=50、top-p=1.0，另外检查 greedy。

候选位置与旧版本不同，必须重新校准。原始捕获目录和旧 trees.json 保持只读。
新树使用已有校准 hidden；缺失时只补算校准帧。新版树标记为 known_tail_plus_next3_v1，
不能用于旧 current 执行器。

性能默认比较六种配置：stock graph AR、stock eager AR、旧 current tree16、
新 pipeline_nocache、新 pipeline_eager、新 pipeline_graph。
三种 pipeline 使用同一新版候选树，可分别观察缓存与图执行收益。
CPU 运行跳过真实 NPU graph；CPU 结果不能证明 NPU 正确性或加速。

## 服务器命令

从已更新的实验分支运行。下面三个结果目录必须不存在，脚本拒绝覆盖旧结果。

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts

python tools/qwen3_tts_spec/highperf_experiment.py calibrate \
  --input /softwarePlatform/y00899301/qwen3_tts_zero_train_spec_run01 \
  --results /softwarePlatform/y00899301/qwen3_tts_spec_highperf_calibration01 \
  --device npu:0 \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend

python tools/qwen3_tts_spec/highperf_experiment.py check \
  --input /softwarePlatform/y00899301/qwen3_tts_zero_train_spec_run01 \
  --trees /softwarePlatform/y00899301/qwen3_tts_spec_highperf_calibration01/trees.json \
  --results /softwarePlatform/y00899301/qwen3_tts_spec_highperf_check01 \
  --device npu:0 \
  --frames-per-request 3 \
  --sampling both \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend

python tools/qwen3_tts_spec/highperf_experiment.py bench \
  --input /softwarePlatform/y00899301/qwen3_tts_zero_train_spec_run01 \
  --trees /softwarePlatform/y00899301/qwen3_tts_spec_highperf_calibration01/trees.json \
  --correctness /softwarePlatform/y00899301/qwen3_tts_spec_highperf_check01/correctness.json \
  --results /softwarePlatform/y00899301/qwen3_tts_spec_highperf_bench01 \
  --device npu:0 \
  --configs tree16 \
  --frames-per-request 3 \
  --warmup 20 \
  --executions-per-round 60 \
  --sampling both \
  --vllm-repo /vllm-workspace/vllm \
  --ascend-repo /vllm-workspace/vllm-ascend

cat /softwarePlatform/y00899301/qwen3_tts_spec_highperf_bench01/report.md
```

三条命令顺序执行，任何一步失败都先处理失败，不继续计时。
check 先检查 smoke011 的第 0、2 帧和 smoke030 的最后一帧，再检查抽样集合。
包含缓存串行 AR、eager 无缓存／有缓存树、NPU graph，以及旧 current 对照。
固定小样本逐节点审计；60 帧中每个请求第一帧的 tree16 也执行节点审计。
两种采样模式都要求 codec codes 与 stock 完全一致，且 sampling 必须复现原始捕获 codes。
不放宽节点 logits 容差：FP32 为 1e-5，低精度为 0.02。

bench 核验代码、模型文件、数据、树、环境、设备、采样模式和检查帧集合。
更换这些内容后重新运行 check。graph 捕获失败会停止并保存错误，不静默降级。
可将 --configs tree16 改为 --configs tree8 tree16 tree32 测试全部预算；不能据此重新调整树形状。

## 产物与解释

- 校准：trees.json、manifest.json、两种内部研究 engine 的配置模板。
- 检查：correctness.json、events.jsonl、manifest.json、中文 report.md。
- 失败：status.json、未通过的 correctness.json、failure.pt；其中包含输入、KV、节点信息和图状态。
- 计时：timings.jsonl、summary.json、graphs.json、memory.json、profiles.jsonl、中文 report.md。
- 进度：replay-progress.log，包含阶段进度、耗时与阶段 ETA；捕获图时逐位置输出状态。

每种模式先预热，再与 stock 按 AR→候选→候选→AR 四轮配对计时。
每轮默认覆盖 60 帧；只有完整帧调用在计时区间内，逐节点审计和文件写入在区间外。
除 5 组相对 stock 的对照，另做缓存相对无缓存、graph 相对 eager 的两组直接对照。
默认每种模式 7 组 × 4 轮 × 60 帧 = 1680 次帧执行，两种模式共 3360 次。
计时前的预热与检查不计入这 3360 次。

summary.json 分开记录总主干调用、prefill／draft／AR／验证／补算调用及图 replay 次数。
图 replay 与主干调用是不同口径：一次图包含多个算子，但 pipeline 每轮只有一次主干前向。
首次 prefill 产生一个 token；后续每轮提交长度不重复计入已经输出的确定根 token。
接受／提交分布来自同批检查帧；阶段分析额外取前两个测试请求各三帧，单独执行。
attention_gather_nested 包含在 backbone_inclusive 内，不能将两项重复相加。
设备事件区间可能包含主机提交空隙，不能当作纯 kernel 时间。
内存峰值包含同时驻留的执行器和图，不是配置独占内存。

不设置性能通过门槛。调用次数减少、耗时减少分别报告；负收益也正常保留。
完整请求／流式播放未被这些帧级结果覆盖。

只重新生成已有性能报告：

```bash
python tools/qwen3_tts_spec/highperf_experiment.py report \
  --results /softwarePlatform/y00899301/qwen3_tts_spec_highperf_bench01
```

## 内部完整请求开关

校准生成的 pipeline_eager／pipeline_graph 模板默认 correctness=null，不能直接用于完整请求。
帧级 check 通过后，将模板中的 correctness 设置为对应 correctness.json 的绝对路径，
再沿用已有内部研究配置入口。加载时验证树、engine、代码及检查过的采样参数／模型维度。
完整请求必须使用与捕获阶段相同的模型权重；首次请求还会包含图准备成本。
该入口不增加 Speech API 字段，也不代表已经完成完整请求或播放验收。
