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
