# Qwen3-TTS CustomVoice zero-training tree experiment

[中文运行说明](README.zh-CN.md)

Base: `v0.30.0` (`a8576ccb725c4e21cd13c3eb5f9a546b21149d2b`).
Branch: `codex/qwen3-tts-zero-train-spec`. The earlier rc1 base is superseded.

This is an original-head reuse / sparse-tree mechanism experiment. It uses
**target sampling followed by tree coverage**, not the paper's residual
resampling algorithm. No parameters are trained, and no KV cache is introduced.
CPU tests cannot establish Ascend performance. Until actual NPU artifacts and
complete-request measurements pass the gates, the status is **implementation
complete, performance unverified**.

## Algorithm and boundaries

- Qwen3-TTS-12Hz-1.7B-CustomVoice, Vivian, English, exactly one request.
- Unset `VLLM_OMNI_TTS_SPEC_CONFIG` for unchanged stock AR. Only the TTS wrapper
  reads this switch; Qwen3-Omni is unaffected. Experimental batches greater
  than one are errors, including warmup. Outer MTP graph capture is disabled
  for experiments. Stock replay retains the tag's inner predictor graphs.
- A draft forward recomputes the committed prefix. Its final hidden state is
  passed through the next three original codebook heads, top four each. A chain
  or ancestor-closed rank tree supplies candidate embeddings. Depth determines
  embedding, next head, and RoPE position; packed node index never does.
- Verification recomputes prefix plus the tree with explicit FP32 attention
  score matmul, masked softmax, and value matmul. NPU norm/RoPE still use the
  existing native operators. Prefix rows remain causal; candidate rows see
  prefix, ancestors, and themselves, never siblings. No compressed causal mask
  is used for trees. Fixed metadata masks are cached, but no hidden/KV state is.
- Target Gumbel draws use the original temperature/top-k distribution. The
  tree cannot read those draws. A hit follows the matching child; a miss commits
  the target draw and ends the round; a leaf can produce one bonus target code.
  One count transfer to the host occurs per tree round, included in timings.
  Both draft and verify backbone calls count in the denominator.
- Defaults: temperature 0.9, top-k 50, top-p 1.0. Top-p other than 1 is rejected.
  Captured inputs are replayed with CPU-generated uniforms keyed by seed,
  request ID, frame index, and codebook position. Full requests use the runner's
  per-request uniforms or a fixed-size draw from its per-request generator.
- Calibration uses only the first ten English smoke texts. Trees of 8/16/32
  nodes maximize greedy marginal covered prefix length: adding a node gains
  one position for observations matching the **entire** rank prefix. Ancestors
  must already exist. Ties use depth and lexical rank order. Negative ranks
  represent tokens outside top four. Frozen trees include a digest; the twenty
  held-out texts cannot change them. Chain ranks are `(0), (0,0), (0,0,0)`.

## NPU run

Use the normal installed vLLM-Omni/vLLM-Ascend environment for this checkout,
including its matching PyTorch, torch-npu and CANN. No extra model is needed.
Run from the repository root. Use a pinned local checkpoint snapshot and record
its immutable revision. The output directory must be shared by the client and
local server workers. Multi-host/Ray collection is not supported.

```bash
MODEL=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
OUT=/data/tts-spec/run01
python tools/qwen3_tts_spec/experiment.py collect --prepare \
  --model "$MODEL" --model-revision CHECKPOINT_COMMIT \
  --output "$OUT" --device npu:0 \
  --vllm-repo /workspace/vllm --ascend-repo /workspace/vllm-ascend
```

This writes `manifest.json`, `collect_config.json`, and `deploy.yaml`. The
deployment is generated from this tag's YAML with sequence capacity one and
stage-0 graph capture sizes `[1]`; codec chunk sizes and playback behavior are
unchanged. Use the same deployment for all server comparisons. Verify that the
chosen model snapshot is the **1.7B CustomVoice** checkpoint, not Base/VoiceDesign.

In a server terminal:

```bash
bash tools/qwen3_tts_spec/serve_npu.sh "$MODEL" "$OUT" collect
```

Once ready, in a client terminal:

```bash
python tools/qwen3_tts_spec/experiment.py collect \
  --model "$MODEL" --output "$OUT" --device npu:0
```

The client sends the 30 texts sequentially. A local control file identifies the
active request. The worker saves a frame shard **after a complete AR call**, and
exports only the predictor weights/config once. Startup warmup is not captured.
This detailed collection deliberately performs CPU copies and frame-level file
I/O; **its request times are not performance evidence**. No per-token writes
occur. Do not issue other requests to this server while collecting. A partial
capture must be restarted in a fresh output directory, not merged with a rerun.

Stop the server before replay, to release model/device memory. Then:

```bash
bash tools/qwen3_tts_spec/replay_npu.sh "$OUT" npu:0
# Equivalently:
python tools/qwen3_tts_spec/experiment.py analyze --output "$OUT" --device npu:0
python tools/qwen3_tts_spec/experiment.py replay --output "$OUT" --device npu:0
python tools/qwen3_tts_spec/experiment.py report --output "$OUT" --device npu:0
```

`analyze` recomputes teacher-forced calibration states from captured AR codes,
saves them separately, and freezes trees. It refuses to overwrite frozen trees.
`replay` first compares greedy and common-uniform codes against stock AR on all
held-out frames. It also checks every visited tree node's next-head logits
against serial evaluation of the same prefix. Logit audit tolerance is
`atol=rtol=1e-5` in FP32 and `2e-2` in lower precision; **codes must match exactly**
in both modes regardless of tolerance. Any mismatch stops performance work and
records diagnostics. Do not relax the code gate to hide a numerical difference.

Node audit failures also save `node_audit_failure.pt`. The `node_diagnostics`
field in `correctness.json` compares the original logits, a serial prefix padded
to the same packed length, and an independent CPU float64 tree/serial evaluator.
These diagnostics help distinguish shape-dependent numerics from topology
errors; they never replace the original acceptance gate. A `diagnostic_error`
field preserves secondary diagnostic failures without losing the original error.

Performance then runs without detailed events/phase timers: twenty warmup
frames per configuration, four ABBA rounds of at least 200 frames each. Small
datasets repeat with distinct execution IDs and an explicit unique-frame count.
Every tree has independent comparisons against stock AR and reference-attention
AR. Each frame wall timer synchronizes completion; device events additionally
record elapsed stream time. Inputs and RNG are prepared outside those timers.
Diagnostic phase profiling is a separate pass, never added to benchmark times.
Use `replay --greedy` only for an additional greedy benchmark; it still runs both
correctness modes. Preserve the stochastic run's artifacts in a separate copy
before replacing timings.

## Full requests after correctness

Restart a clean server for each configuration (stop the previous one first).
Capture must be disabled during these performance runs:

```bash
# Server terminal, stock baseline:
bash tools/qwen3_tts_spec/serve_npu.sh "$MODEL" "$OUT" stock
# Client terminal:
python tools/qwen3_tts_spec/experiment.py collect --e2e-label stock \
  --model "$MODEL" --output "$OUT" --device npu:0

# Restart server, for example the frozen tree16 configuration:
bash tools/qwen3_tts_spec/serve_npu.sh "$MODEL" "$OUT" tree16
python tools/qwen3_tts_spec/experiment.py collect --e2e-label tree16 \
  --model "$MODEL" --output "$OUT" --device npu:0
python tools/qwen3_tts_spec/experiment.py report --output "$OUT" --device npu:0
```

Labels must match the server mode. These are twenty complete held-out requests,
not a replay of AR histories. TTFP is arrival of first nonempty PCM bytes; E2E
is completion of the response; RTF uses mono 24 kHz signed 16-bit PCM duration.
They measure delivery to the client, not speaker playback. Connection/startup
effects are present in both runs. No playback or buffering policy is modified.

## Artifacts and interpretation

- `manifest.json`: baseline/repository commits, checkpoint revision, package and
  device versions, CANN when discoverable, dataset/split/seed, graph intent.
  `predictor.pt` contains actual exported dtype, config and predictor weights;
  `replay.json` records actual dtype, graph buckets, peak allocated memory and
  execution environment. Source hashes accompany repository commits to identify
  uncommitted implementation changes. If CANN auto-discovery returns null, use
  `--cann-version` with `collect --prepare` and `replay`; do not infer it from
  torch-npu version.
- `frames/*.pt`: complete frame inputs, AR codes, sampling parameters, request,
  frame and split IDs. No cloned voice or reference audio is used.
- `calibration_states/*.pt`, `trees.json`, `coverage.json`: immutable rank trees,
  calibration-only potential, and necessary frozen-prefix hidden states.
- `events.jsonl`: round hits/commits/nodes/calls and exit reasons.
  `diagnostics.jsonl`: separate per-phase device/host times and node-logit errors.
- `timings.jsonl`: complete-frame paired wall/device timing; comparator and
  actual attention backend. `processed_positions` and `attention_pairs` are
  backbone compute proxies. `estimated_matmul_flops` sums analytical backbone,
  attention, head and projection matmul work; it excludes pointwise operations
  and physical kernel padding, and is **not a hardware FLOP measurement**.
  Heads/embedding overhead is included in measured time. Explicit reference attention is eager; no
  tree-graph capture or silent graph fallback is claimed.
- `correctness.json`, `requests-*.jsonl`, `summary.json`, `report.md`.

The bootstrap resamples **requests**, keeping repeated frames and their paired
measurements together (5,000 draws, 95% interval). A passing fixed configuration
needs at least 1.10x frame wall speedup against stock AR and a lower interval
bound above one, plus correct complete requests. The initial explicit meaning
of “no clear E2E regression” is at most 10% increase in mean E2E/RTF and p95 TTFP,
with no request errors. Missing full-request evidence cannot pass this gate.
Reduced backbone calls alone are reported as mechanism potential; insufficient
coverage is retained to motivate later training. The reference AR comparison
separates backend effects from tree algorithm effects. This 30-text exercise
does not establish paper-level generalization or production speedup.

Return the manifest, frozen trees, correctness, events, diagnostics, timings,
replay metadata, request metrics and report. Retain tensor shards locally for
diagnosis; the exported predictor checkpoint can be large.

## CPU checks

```bash
python -m pytest --noconftest -o addopts='' \
  tests/model_executor/models/qwen3_tts/test_code_predictor_dtype.py \
  tests/model_executor/models/qwen3_tts/test_code_predictor_spec.py -q
```

Requires PyTorch, Transformers, pytest, pytest-mock, NumPy and PyYAML. The direct
module fixtures stub vLLM runtime imports so these checks need no accelerator.
The synthetic integration test exercises actual export/reload, calibration,
all ABBA rounds, report generation and invalidation after a correctness failure;
its timings must not be interpreted as real-model results.
