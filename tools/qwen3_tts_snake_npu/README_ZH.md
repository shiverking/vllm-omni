# Triton-Ascend SnakeBeta 第一版

仅显式开启时使用 NPU 融合路径，默认仍为 eager。当前分支为
`codex/qwen3-tts-code2wav-incremental-kv`。需要与 CANN、torch-npu 匹配的
Triton-Ascend；普通 CUDA Triton 包不能替代。部署前请确认实际安装版本。

## 实现

按通道切分连续时间块，以有限数量 programs 循环处理各块。输入只读取一次，
五步中间结果保留在片上，输出只写一次；alpha/inv_beta 使用已有冻结参数缓存。
显式保留乘法、sin、平方、乘法、加法各步 dtype 转换，关闭 FP 融合。
这不保证 tl.sin 与 CANN Sin 逐位一致，最终需实际激活和波形检查。

默认 block=2048、programs=32 为初始配置，不代表已调优。支持128～8192的二次幂分块。
支持时间轴连续的 BCT 输入（外层支持 stride，避免裁剪后额外复制）及与输入同 dtype 的连续通道参数；BF16/FP16/FP32，推理模式。
显式启用后编译或执行失败会报错，不静默混入 eager 计时。
现有 wrapper 的预热在图捕获前执行 JIT；微基准额外验证图输入更新。

## 快速微基准

```bash
cd /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts
python tools/qwen3_tts_snake_npu/benchmark.py \
  --results /softwarePlatform/y00899301/qwen3_tts_snake_npu_run01/quick \
  --device npu:0 --batches 1 4 --blocks 2048 --programs 32 \
  --warmup 10 --iterations 30 --executions eager graph
```

三种默认形状覆盖大小张量，但不代替真实激活。扫描分块：

```bash
python tools/qwen3_tts_snake_npu/benchmark.py \
  --results /softwarePlatform/y00899301/qwen3_tts_snake_npu_run01/sweep \
  --device npu:0 --batches 1 2 4 --blocks 1024 2048 4096 \
  --programs 16 32 40 --warmup 10 --iterations 30 --executions graph
```

可用 `--kernel-csv` 读取 profiling 的 kernel_details.csv 中实际 Sin 形状；
其中也可能包含 RoPE 的 Sin，只用于形状覆盖。可用 `--inputs` 读取可信本地
`.pt` 的 `[{"x": tensor, "alpha": tensor, "inv_beta": tensor}, ...]`，
在同一工具中对比真实激活和实际参数。每个配置独立记录错误并继续。
结果包含版本、形状、误差、更新输入后的误差、ABBA 两侧墙钟/设备事件及加速比。
`outside_tolerance` 使用 torch.isclose 默认 rtol=1e-5、atol=1e-8；它不作为性能入口门槛。
正式波形检查使用项目既有标准。当前只记录最大/平均误差，不宣称音质等价。

## 服务实验

保持 growth_kv 和其他设置相同，只增加 SnakeBeta 开关：

```bash
vllm-omni serve \
  /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config /softwarePlatform/y00899301/vllm_debug_omni/vllm-omni-tts/vllm_omni/deploy/qwen3_tts_kv_frame_code2wav_growth_kv_snake.yaml \
  --host 0.0.0.0 --port 8091 --trust-remote-code --omni
```

对照使用原 `qwen3_tts_kv_frame_code2wav_growth_kv.yaml`，分别启动。
可在内部配置中调整 `code2wav_snake_block`、`code2wav_snake_programs`。
不得在已捕获图上动态切换；调整后重启并重新捕获。
所有测试独立，不要求上一测试通过。本地 CPU 只验证舍入顺序、分块覆盖和接口，
不证明 NPU 编译、graph 捕获、精度或加速成功。


## 第二版：跨通道与固定 auto 策略

保留默认 channel 路径。`--layouts channel flat auto` 可分别测试：
flat 将 BCT 展平，一个 tile 可跨多个通道，按元素获取对应 alpha/inv_beta，
避免 T=37 每通道单独补齐。支持外层 stride；不是所有形状都保证更快。

auto 是显式的实验策略，不是运行时自动调优：T<=128 选 flat/1024/40；
128<T<=4096 选 channel/2048/40（已测 B=1,C=96,T=1920 使用32）；
T>4096 选 channel/4096/40。未测中间形状为外推，报告保存实际选择。
`auto` 忽略手动 blocks/programs，每个形状只测一次；手动布局保留完整扫描。

短轴专项扫描（18组）：

```bash
python tools/qwen3_tts_snake_npu/benchmark.py \
  --results /softwarePlatform/y00899301/qwen3_tts_snake_npu_run01/flat02 \
  --device npu:0 --shapes 1,1024,37 4,1024,37 \
  --layouts flat --blocks 256 512 1024 --programs 16 32 40 \
  --warmup 10 --iterations 50 --executions graph
```

auto 六形状检查：

```bash
python tools/qwen3_tts_snake_npu/benchmark.py \
  --results /softwarePlatform/y00899301/qwen3_tts_snake_npu_run01/auto02 \
  --device npu:0 --batches 1 4 --layouts auto \
  --warmup 10 --iterations 50 --executions graph
```

真实激活采集（复用 codes，growth_kv eager，原 SnakeBeta）：

```bash
python tools/qwen3_tts_snake_npu/capture.py \
  --input /softwarePlatform/y00899301/qwen3_tts_kv_frame_code2wav_graph_run01/capture \
  --results /softwarePlatform/y00899301/qwen3_tts_snake_npu_run01/activations02 \
  --model /softwarePlatform/l00927835/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device npu:0 --request-limit 3 --batches 1 4 --max-samples 24 --max-mib 512

python tools/qwen3_tts_snake_npu/benchmark.py \
  --inputs /softwarePlatform/y00899301/qwen3_tts_snake_npu_run01/activations02 \
  --results /softwarePlatform/y00899301/qwen3_tts_snake_npu_run01/real02 \
  --device npu:0 --layouts auto --warmup 10 --iterations 50 --executions graph
```

每种形状仅保留一个真实样本；batch通过复制真实请求覆盖形状，不代表服务请求分布。
最多24个样本/512MiB，coverage.json 记录观察、保存与遗漏形状；不是完整输入域精度证明。
分片逐个加载以控制 NPU 内存。index.json 是本轮入口，重跑可覆盖受管理文件。

服务 opt-in 使用 `qwen3_tts_kv_frame_code2wav_growth_kv_snake_auto.yaml`；
原 snake.yaml 仍使用第一版固定配置，不修改原实验对照。先做微基准和真实激活实验，
完整波形与服务性能需另测；命令之间没有通过门槛。
