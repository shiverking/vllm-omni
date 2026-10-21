# RVQ 帧完成时间与播放期限实验

这是研究入口，默认服务路径不启用。代码从
`2c8d928306e117a2d935d164becb8041a53fb5bc` 分出，分支为
`codex/qwen3-tts-playback-deadline`。

当前本地能验证 CPU 算法、状态隔离、数据产物与调度协议；NPU graph、真实服务与
GStreamer/NPU 联调必须在服务器验证。没有服务器结果前，不宣称加速、连续性或容量改善。

## 实现与归因

- 共享帧状态保存前缀 KV、已输出尾 token、父 hidden、codes 与 uniforms。
  每轮树切换只选择另一张图；仅在请求位置分化时搬运子批次状态。
- 动作是缓存串行一步及 D1N4、D2N4/8/16、D3N4/8/16/32。
  `p` 表示最后已输出的码本；prefill 产生 c1，p=15 结束。
  失配或 bonus 只保留为未处理尾 token，不写入虚构 KV。
- 图键包含实际 batch、位置和裁剪形状。帧尾物理 query 变小，相同形状去重。
  图失败直接停止；不会在 graph 测量中回退 eager。
- 校准保留 advancement、wall、device event 的联合样本。策略包括均值 DP、
  lambda=0/0.5/1/2 和 1 ms 网格、最大 2 秒的完成概率 DP。
  期限过期或概率全为零时，选择最短预期剩余时间。
- 每个实际子批次统一动作。batch=3/5/6/7 分解成 1/2/4/8 的真实子批次；
  stock 保留自己的原生批处理。请求重组、主机同步与查表均计入完整帧耗时。
- LiveServe 的 `audio_interaction.py` 与 GStreamer 播放器来自 `liveserve-test`。
  调度接口按当前引擎适配，保留当前分支的通信实现，未整分支覆盖旧引擎。
- 受控服务的执行许可覆盖 Talker prefill、生成、Code Predictor 和 Code2Wav 的
  完整 stage step；执行完成并同步后才释放。请求取消由原引擎处理，已经提交的一帧不抢占。
- 外层使用实际就绪 stage batch，以及同一组 32 个经验场景，按逾期会话数、
  逾期时间、资源窗口成本和 FCFS 排序。设备占用估计采用实测受控窗口；
  它包含主机空隙，不是纯 kernel 时间。
- 客户端只反馈真实 PCM 收到量、播放量与 chunk 数。服务端用自己的单调时钟计算
  反馈年龄及 Code2Wav 完成到可播放反馈的确认间隔，不相减两端时钟。

模型是经验近似：位置表忽略隐藏状态条件和整个批次的联合分化；外层只枚举当前
两个 stage 已准备好的批次及其策略，后续工作按 EDF 近似，不枚举所有请求子集。
正式数据中的预测误差必须单独报告，不作全局最优或概率保证。

## 先运行 batch=1

以下命令不会 collect、不会修改旧树，也不会运行全量 replay。

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts

python tools/qwen3_tts_spec/playback_experiment.py prepare \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/config.json \
  --input /softwarePlatform/y00899301/qwen3_tts_zero_train_spec_run01 \
  --device npu:0 --batch-buckets 1

python tools/qwen3_tts_spec/playback_experiment.py calibrate \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/config.json

python tools/qwen3_tts_spec/playback_experiment.py check \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/config.json \
  --probe-only

python tools/qwen3_tts_spec/playback_experiment.py check \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/config.json

python tools/qwen3_tts_spec/playback_experiment.py simulate \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/config.json

python tools/qwen3_tts_spec/playback_experiment.py measure \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/config.json
```

配置中已经写入真实模型、vllm 与 vllm-ascend 目录。两个采样模式独立执行。
校准只使用原 10 条校准请求的首／中／末帧。抽样检查使用 20 条测试请求的 60 帧。
三帧预检使用 smoke011 的 0、2 帧和 smoke030 的末帧；通过前不授权 60 帧检查。
正式计时必须有当前代码、捕获数据、策略、动作和 batch 形状匹配的证书。

`fixed_full` 是相同新候选语义、固定 query 的 tree16；`d3n16` 只增加帧尾缩图。
`mean_b1` 在大 batch 复用 B1 策略；`mean` 使用实际子批次 B。
`greedy_rate` 最大化每轮提交量／时间，`serial` 是缓存串行，`stock` 是原始完整帧 AR。
不得把 stock 的成本当成帧内串行动作成本。

每种配置预热 20 帧，再做 ABBA 四轮配对。同一轮覆盖相同抽样输入。
末组不足 batch 时显式重复最后一帧，原始记录保留完整身份列表；独立帧数不增加。
报告的 batch 延迟是整个输入批次完成的时间；不得直接当成每请求独占设备成本。
配对区间按相同输入批次重采样，是探索性区间。

## 再扩展到 1/2/4/8

保留 batch=1 产物，新结果写入子目录；后续 `calibrate/check/simulate/measure` 都用下面的新配置。

```bash
python tools/qwen3_tts_spec/playback_experiment.py prepare \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --input /softwarePlatform/y00899301/qwen3_tts_zero_train_spec_run01 \
  --device npu:0 --batch-buckets 1 2 4 8

python tools/qwen3_tts_spec/playback_experiment.py calibrate \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json

python tools/qwen3_tts_spec/playback_experiment.py check \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json --probe-only

python tools/qwen3_tts_spec/playback_experiment.py check \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json

python tools/qwen3_tts_spec/playback_experiment.py simulate \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json

python tools/qwen3_tts_spec/playback_experiment.py measure \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json
```

切树、变 batch、缓存串行和固定 query 都需要 codes 对照；首错保存后停止。
图捕获错误只保存主机诊断，避免再次读取已损坏的设备上下文。
校准按 sampling/batch 保存检查点，可重跑同一命令恢复；更改绑定后需要新的目录。
`check` 和 `measure` 失败时重新运行该阶段，不复用部分正确性结果。

## 服务闭环：先校准阶段成本

需要服务器 Python 能导入 `gi.repository.Gst`，以及 appsrc、queue、clocksync、fakesink。
播放器是 GStreamer 时钟驱动的静音 sink；不会实际降低播放速度。当前研究入口仅允许 npu:0 共卡。

第一个终端启动受控 AR 服务：

```bash
python tools/qwen3_tts_spec/playback_experiment.py serve \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --method liveserve_ar --sampling sampling
```

第二个终端运行：

```bash
python tools/qwen3_tts_spec/playback_experiment.py bench \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --method liveserve_ar --sampling sampling --phase warmup --concurrency 1 --requests 2

python tools/qwen3_tts_spec/playback_experiment.py bench \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --method liveserve_ar --sampling sampling --phase calibration --concurrency 8 --requests 40
```

分别在 concurrency 1/2/4/8/16 采集；不同并发点的文件分开保存。同一服务进程可继续采集。
阶段成本必须有实际提交 batch 的覆盖；B3/5/6/7 没采到时不会用 B1 插值。
停止服务以刷完日志后，拟合模型：

```bash
python tools/qwen3_tts_spec/playback_experiment.py calibrate \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --stage-events /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/active_service.json
```

`active_service.json` 是最近一次服务的实际日志地址清单，命令没有环境变量或手工占位路径。
拟合只接收 calibration 标签。缺失桶会打印并保留在模型中；联合调度访问缺失桶时明确失败。
Code2Wav 到可播放确认的间隔包含 50 ms 反馈周期，不等于单向网络传输时间。

## 服务测试与对照

```bash
python tools/qwen3_tts_spec/playback_experiment.py serve \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --method joint --sampling sampling
```

另一个终端依次运行：

```bash
python tools/qwen3_tts_spec/playback_experiment.py bench \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --method joint --sampling sampling --phase warmup --concurrency 1 --requests 2

python tools/qwen3_tts_spec/playback_experiment.py bench \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --method joint --sampling sampling --phase check --concurrency 8 --requests 40

python tools/qwen3_tts_spec/playback_experiment.py bench \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json \
  --method joint --sampling sampling --phase measure --concurrency 8 --requests 40

python tools/qwen3_tts_spec/playback_experiment.py report \
  --config /softwarePlatform/y00899301/qwen3_tts_playback_deadline_run01/batch_all/config.json
```

warmup 在实际输入到来后捕获全部认证图，capture 输出有位置与形状；其请求耗时不进入测量。
check 在真实请求中逐帧与同 uniforms 的 stock 对照，首次差异保存并停止。
measure 不执行该对照，并要求同一服务进程已有不低于当前并发的完整请求检查证书。
每个新服务进程都要重新 warmup/check，不能拿另一个进程的图状态作保证。

方法：`native`、`liveserve_ar`、`liveserve_fixed`、`position`、`joint_ar`、`joint`。
消融：`--ablation buffer_only|no_code2wav|no_batch`，serve 与 bench 必须相同。
`--trace` 只用于另外的服务诊断轮，禁止同时做 measure。

`native` 保留原生异步执行和 64 个 resident slots，不经过设备仲裁；受控窗口对照使用
8 个 resident slots，所有受控 AR/树方法一致。原生服务与受控服务的差值包含执行窗口和
批处理差异；投机收益只能在相同受控设置中归因。当前未实现“动态树且保留原生异步预提交”
这个组合，不能把受控结果称为该组合的性能。

## 数据、产物与结果解释

- `actions.json`：校准树、源数据／权重／代码绑定及哈希。
- `costs.json`、`prefill_costs.json`：位置／实际 B／采样模式的联合样本，prefill 单独计费。
- `policy.json`：均值、成本权衡、完成概率表；`simulation.json` 是离线分布预测。
- `probe.json`、`correctness.json`：固定三帧与抽样证书；节点事件包含实际子批次、接受／提交数量。
- `first_failure.pt`／`service_first_failure.pt`：输入、uniforms、KV、节点 logits、映射和图上下文。
- `measurements.json`、`summary.json`：完整帧计时、调用、查表、重组、设备区间和预测误差。
- `service/*/events.jsonl`：服务端同一时钟的到达、反馈、stage grant、交付确认；
  `measure-c*-seed*.json` 保存客户端真实 PCM 时长、TTFP、E2E、断流和逐 chunk 时间线。
- `report.md`、`service_summary.json`：中文结果及服务汇总。报告不设置加速门槛。

正式独立文本未提供时只写 smoke。`prepare --text-manifest` 可去重并冻结 80/40/200 文本，
但不会把旧 smoke 帧伪装成新文本捕获；必须提供相应新捕获数据后才能做独立确认。
已经包含 80/40/200 唯一文本的捕获 manifest 才允许 `bench --formal`；
正式种子为 17/29/43，先排除 30 秒负载建立期，再保持至少 600 秒、200 个成功请求。
建立期原始请求另存 `ramp_requests`，不混入稳态 QoS；原始结果还必须核对成功完成数。
`--arrivals` 接收固定非负升序秒偏移数组，长度等于请求数；排队延迟纳入首音频／E2E。
1.25×/1.5×在独立配置中设置 `playback_rate`，更改绑定后重新验证。

容量按首个可播放音频≤1秒、最长断流≤100ms的会话占比≥95%计算。
必须同时展示失败请求、原始断流／延迟／吞吐，而不是只展示达到门槛的点。
sampling 无收益也应报告，不用 greedy 替代，不把未训练预设为唯一原因。

本地 CPU 验证：

```powershell
cd D:\vllm-omni
.venv-spec/Scripts/python.exe tools/qwen3_tts_spec/run_playback_cpu_tests.py
```

该测试入口仅绕过 vLLM 平台初始化，执行真实 CPU 小模型及协议测试；不模拟 NPU graph 成功。
