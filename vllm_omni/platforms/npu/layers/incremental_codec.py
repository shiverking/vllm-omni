# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental Qwen3-TTS codec decode with per-request slots and NPU graphs.

The eager Code2Wav path re-runs the sliding-window transformer and the
upsample stack on overlapping context at every chunk. This path keeps causal
convolution tails and the last ``sliding_window - 1`` keys/values, and runs
only the new frames. Fixed ``(batch, frames)`` shapes are captured once.

Backported from vLLM-Omni #9068.
"""

from __future__ import annotations

import time
from collections import Counter, OrderedDict
from types import SimpleNamespace

import torch
from transformers.cache_utils import Cache
from vllm.logger import init_logger

logger = init_logger(__name__)


class _SlotHistory:
    def __init__(self, arena: NPUIncrementalCodec, indices: torch.Tensor) -> None:
        self.arena = arena
        self.indices = indices

    def get(self, module: torch.nn.Module) -> torch.Tensor:
        return self.arena.histories[module].index_select(0, self.indices)

    def __setitem__(self, module: torch.nn.Module, value: torch.Tensor) -> None:
        self.arena.histories[module].index_copy_(0, self.indices, value)


class _CodecCache(Cache):
    def __init__(self, arena: NPUIncrementalCodec, indices: torch.Tensor) -> None:
        super().__init__(layers=[])
        self.arena = arena
        self.indices = indices

    def update(self, keys, values, layer_idx, cache_kwargs):
        del cache_kwargs
        old_keys = self.arena.keys[layer_idx]
        old_values = self.arena.values[layer_idx]
        keys = torch.cat((old_keys.index_select(0, self.indices), keys), dim=2)
        values = torch.cat((old_values.index_select(0, self.indices), values), dim=2)
        kept = self.arena.history
        old_keys.index_copy_(0, self.indices, keys[:, :, -kept:])
        old_values.index_copy_(0, self.indices, values[:, :, -kept:])
        return keys, values


class _ShapeGraph:
    def __init__(self, arena: NPUIncrementalCodec, batch: int, length: int) -> None:
        self.arena = arena
        weight = next(arena.decoder.parameters())
        self.batch = batch
        self.length = length
        self.indices = torch.arange(batch, dtype=torch.long, device=weight.device)
        self.codes = torch.zeros(
            batch,
            arena.decoder.config.num_quantizers,
            length,
            dtype=torch.long,
            device=weight.device,
        )
        self.offsets = torch.arange(length, device=weight.device)
        self.key_offsets = torch.arange(-arena.history, length, device=weight.device)
        self.state = SimpleNamespace(history=_SlotHistory(arena, self.indices))
        self.cache = _CodecCache(arena, self.indices)
        self.graph: torch.npu.NPUGraph | None = None
        self.output: torch.Tensor | None = None

    def compute(self) -> torch.Tensor:
        decoder = self.arena.decoder
        run = self.arena._run_module
        hidden = decoder.quantizer.decode(self.codes)
        hidden = run(decoder.pre_conv, hidden, self.state).transpose(1, 2)
        transformer = decoder.pre_transformer
        hidden = transformer.input_proj(hidden)
        frames = self.arena.frames.index_select(0, self.indices)
        positions = frames[:, None] + self.offsets[None, :]
        key_positions = frames[:, None] + self.key_offsets[None, :]
        allowed = (
            (key_positions[:, None, :] >= 0)
            & (key_positions[:, None, :] <= positions[:, :, None])
            & (key_positions[:, None, :] > positions[:, :, None] - self.arena.window)
        )
        mask = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)[:, None]
        rotary = transformer.rotary_emb(hidden, positions)
        for layer in transformer.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=positions,
                position_embeddings=rotary,
                force_eager_attention=True,
            )
        hidden = transformer.output_proj(transformer.norm(hidden)).permute(0, 2, 1)
        for blocks in decoder.upsample:
            for block in blocks:
                hidden = run(block, hidden, self.state)
        for block in decoder.decoder:
            hidden = run(block, hidden, self.state)
        self.arena.frames.index_copy_(0, self.indices, frames + self.length)
        return hidden.clamp(min=-1, max=1)

    @torch.inference_mode()
    def capture(self, *, pool=None, stream=None) -> None:
        saved = [buffer.index_select(0, self.indices).clone() for buffer in self.arena.buffers]
        if stream is None:
            stream = torch.npu.Stream(device=self.codes.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                self.compute()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(self.graph, stream=stream, pool=pool):
                self.output = self.compute()
        finally:
            torch.npu.current_stream().wait_stream(stream)
            for buffer, original in zip(self.arena.buffers, saved, strict=True):
                buffer.index_copy_(0, self.indices, original)

    @torch.inference_mode()
    def run(
        self, indices: torch.Tensor, codes: torch.Tensor, *, replay: bool, emit: bool = True
    ) -> torch.Tensor | None:
        self.indices.copy_(indices)
        self.codes.copy_(codes)
        if replay and self.graph is not None and not torch.npu.is_current_stream_capturing():
            self.graph.replay()
            assert self.output is not None
            return self.output.clone() if emit else None
        output = self.compute()
        return output if emit else None


class NPUIncrementalCodec:
    """Slot arena over one decoder. One worker thread owns the buffers."""

    def __init__(
        self,
        decoder,
        slots: int,
        max_batch: int,
        chunk_frames: int,
        *,
        initial_chunk_frames: int | None = None,
        reference_cache_capacity: int = 0,
    ) -> None:
        if slots < 1 or max_batch < 1 or chunk_frames < 1:
            raise ValueError("incremental codec slots, batch and chunk frames must be positive")
        if max_batch > slots:
            raise ValueError("incremental codec batch cannot exceed the slot count")
        if initial_chunk_frames is not None and initial_chunk_frames < 1:
            raise ValueError("incremental codec first chunk must be positive")
        if reference_cache_capacity < 0 or (reference_cache_capacity and initial_chunk_frames is None):
            raise ValueError("reference state caching requires an ICL graph and a nonnegative capacity")
        window = int(getattr(decoder.config, "sliding_window", 0) or 0)
        if window < 2:
            raise ValueError("incremental codec requires a sliding attention window")
        self.decoder = decoder
        self.slots = int(slots)
        self.max_batch = int(max_batch)
        self.chunk_frames = int(chunk_frames)
        self.initial_chunk_frames = initial_chunk_frames
        self.reference_cache_capacity = reference_cache_capacity
        self._reference_states: OrderedDict[str, tuple[torch.Tensor, ...]] = OrderedDict()
        self._reference_cache_hits: Counter[str] = Counter()
        self.window = window
        self.history = window - 1
        self.upsample = int(decoder.total_upsample)
        weight = next(decoder.parameters())
        self.device = weight.device
        self.histories: dict[torch.nn.Module, torch.Tensor] = {}
        for module in decoder.modules():
            kind = type(module).__name__
            if kind == "Qwen3TTSTokenizerV2CausalConvNet" and module.padding:
                if module.stride != 1:
                    raise ValueError("incremental codec requires stride-one causal convolutions")
                self.histories[module] = weight.new_zeros(slots, module.conv.in_channels, module.padding)
            elif kind == "Qwen3TTSTokenizerV2CausalTransConvNet" and module.right_pad:
                self.histories[module] = weight.new_zeros(slots, module.conv.out_channels, module.right_pad)
        config = decoder.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        shape = (slots, config.num_key_value_heads, self.history, head_dim)
        layers = decoder.pre_transformer.layers
        self.keys = [weight.new_zeros(shape) for _ in layers]
        self.values = [weight.new_zeros(shape) for _ in layers]
        self.frames = torch.zeros(slots, device=weight.device, dtype=torch.long)
        self.buffers = list(self.histories.values()) + self.keys + self.values + [self.frames]
        self.free = list(range(slots))
        self.graphs: dict[tuple[int, int], _ShapeGraph] = {}
        # Reuse one capture stream; every captured shape owns a private pool.
        self.capture_stream = torch.npu.Stream(device=self.device)  # type: ignore[attr-defined]
        # Runtime formats are prepared by the NPU Code2Wav loader. Attention
        # selection is local to each incremental call; preserve the shared config.
        self.prewarm()

    def prewarm(self) -> None:
        started = time.perf_counter()
        lengths = {1, self.chunk_frames}
        if self.initial_chunk_frames is not None:
            lengths.add(min(self.initial_chunk_frames, self.chunk_frames))
        for batch in range(1, self.max_batch + 1):
            for length in sorted(lengths):
                self._graph(batch, length, capture=True)
        logger.info(
            "Prepared Qwen3-TTS incremental codec graphs: slots=%d batches=1..%d frames=%s graphs=%d seconds=%.3f",
            self.slots,
            self.max_batch,
            sorted(lengths),
            len(self.graphs),
            time.perf_counter() - started,
        )

    def _graph(self, batch: int, length: int, *, capture: bool = False) -> _ShapeGraph:
        key = (batch, length)
        graph = self.graphs.get(key)
        if graph is None:
            graph = _ShapeGraph(self, batch, length)
            # Only prewarm captures graphs. A final partial chunk must not
            # synchronize the device and run several warmups on a live request.
            if capture:
                try:
                    graph.capture(pool=torch.npu.graph_pool_handle(), stream=self.capture_stream)
                except Exception as exc:
                    raise RuntimeError(
                        f"NPU incremental Codec capture failed for batch={batch}, frames={length}"
                    ) from exc
                self.graphs[key] = graph
        return graph

    def alloc(self) -> int:
        if not self.free:
            raise RuntimeError("incremental codec has no free request slots")
        return self.free.pop()

    def release(self, slot: int) -> None:
        self.reset(slot)
        if slot not in self.free:
            self.free.append(slot)

    @torch.inference_mode()
    def reset(self, slot: int) -> None:
        for buffer in self.buffers:
            buffer[slot].zero_()

    def _prime_cached_reference(self, key: str, slots: list[int], codes: torch.Tensor) -> None:
        state = self._reference_states.get(key)
        voice = key.partition(":")[0]
        if state is None:
            # A single row fixes the priming shape and avoids duplicate work
            # when multiple requests for this reference arrive together cold.
            slot, *slots = slots
            self.reset(slot)
            self._decode_same_length([slot], codes.to(device=self.device, dtype=torch.long), emit=False)
            state = tuple(buffer[slot].clone() for buffer in self.buffers)
            self._reference_states[key] = state
            if len(self._reference_states) > self.reference_cache_capacity:
                self._reference_states.popitem(last=False)
            logger.info(
                "Qwen3-TTS Code2Wav reference state cached: voice=%s frames=%d bytes=%d",
                voice,
                codes.shape[-1],
                sum(t.numel() * t.element_size() for t in state),
            )
        else:
            self._reference_states.move_to_end(key)
        if slots:
            indices = torch.tensor(slots, dtype=torch.long, device=self.device)
            for buffer, saved in zip(self.buffers, state, strict=True):
                # Give Ascend's scatter each row in a regular physical layout.
                rows = saved.expand(len(slots), *saved.shape).contiguous()
                buffer.index_copy_(0, indices, rows)
            if not self._reference_cache_hits[voice]:
                logger.info("Qwen3-TTS Code2Wav reference state reused: voice=%s slots=%d", voice, len(slots))
            self._reference_cache_hits[voice] += len(slots)

    def _run_module(self, module, hidden, state):
        kind = type(module).__name__
        if kind == "Qwen3TTSTokenizerV2CausalConvNet":
            width = module.padding
            if width:
                previous = state.history.get(module)
                hidden = torch.cat((previous, hidden), dim=-1)
                state.history[module] = hidden[..., -width:]
            return module.conv(hidden).contiguous()
        if kind == "Qwen3TTSTokenizerV2CausalTransConvNet":
            output = module.conv(hidden)
            overlap = module.right_pad
            if overlap:
                previous = state.history.get(module)
                output[..., :overlap] = output[..., :overlap] + previous
                tail = output[..., -overlap:]
                if module.conv.bias is not None:
                    tail = tail - module.conv.bias.view(1, -1, 1)
                state.history[module] = tail
                output = output[..., :-overlap]
            return output.contiguous()
        if kind == "Qwen3TTSTokenizerV2ConvNeXtBlock":
            residual = hidden
            hidden = self._run_module(module.dwconv, hidden, state).permute(0, 2, 1)
            hidden = module.pwconv2(module.act(module.pwconv1(module.norm(hidden))))
            return residual + (module.gamma * hidden).permute(0, 2, 1)
        if kind == "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit":
            residual = hidden
            hidden = self._run_module(module.conv1, module.act1(hidden), state)
            hidden = self._run_module(module.conv2, module.act2(hidden), state)
            return residual + hidden
        if kind == "Qwen3TTSTokenizerV2DecoderDecoderBlock":
            for child in module.block:
                hidden = self._run_module(child, hidden, state)
            return hidden
        if kind == "SnakeBeta":
            return module(hidden)
        raise TypeError(f"Unsupported incremental codec module: {kind}")

    @torch.inference_mode()
    def _decode_same_length(self, slots: list[int], codes: torch.Tensor, *, emit: bool = True) -> torch.Tensor | None:
        """Advance all slots and optionally return their owned PCM."""
        if not slots:
            raise ValueError("incremental codec received an empty batch")
        length = int(codes.shape[-1])
        pieces: list[torch.Tensor] = []
        offset = 0
        indices = torch.tensor(slots, dtype=torch.long, device=self.device)
        while offset < length:
            step = min(self.chunk_frames, length - offset)
            batch = len(slots)
            graph = self._graph(batch, step)
            piece = graph.run(indices, codes[:, :, offset : offset + step], replay=True, emit=emit)
            if piece is not None:
                pieces.append(piece)
            offset += step
        if not pieces:
            return None
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)

    @torch.inference_mode()
    def decode_requests(self, codes: torch.Tensor, lengths, caches: list[dict]) -> list[torch.Tensor]:
        if codes.ndim != 3 or codes.shape[0] != len(caches) or len(lengths) != len(caches):
            raise ValueError("incremental codec expects one code row and length per request")
        if codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError("incremental codec codebook count does not match the decoder")
        if any(int(length) < 0 or int(length) > codes.shape[-1] for length in lengths):
            raise ValueError("incremental codec lengths exceed the supplied codes")
        live = [
            cache
            for cache, length in zip(caches, lengths, strict=True)
            if int(length) > 0 and not cache.get("_is_dummy_run")
        ]
        slots = [cache["npu_slot"] for cache in live if "npu_slot" in cache]
        if (
            len({id(cache) for cache in live}) != len(live)
            or len(set(slots)) != len(slots)
            or any(type(slot) is not int or not 0 <= slot < self.slots or slot in self.free for slot in slots)
        ):
            raise ValueError("incremental codec requests must own distinct allocated slots")
        if len(live) - len(slots) > len(self.free):
            raise RuntimeError("incremental codec has no free request slots")
        prefixes = [
            int(cache.get("prefix_frames") or 0)
            if self.initial_chunk_frames is not None and "npu_slot" not in cache and not cache.get("_is_dummy_run")
            else 0
            for cache in caches
        ]
        if any(not 0 <= prefix <= int(length) for prefix, length in zip(prefixes, lengths, strict=True) if length):
            raise ValueError("ICL first chunk is shorter than its reference prefix")

        outputs: list[torch.Tensor | None] = [None] * len(caches)
        priming_groups: dict[int, list[int]] = {}
        cached_priming_groups: dict[str, list[int]] = {}
        groups: dict[int, list[int]] = {}
        for index, cache in enumerate(caches):
            length = int(lengths[index])
            if length < 1:
                outputs[index] = torch.zeros(0, device=self.device, dtype=torch.float32)
                continue
            if cache.get("_is_dummy_run"):
                outputs[index] = self.decoder._forward_exact(codes[index : index + 1, :, :length])[0, 0]
                continue
            prefix = prefixes[index]
            key = cache.get("npu_reference_cache_key") if prefix and self.reference_cache_capacity else None
            slot = cache.get("npu_slot")
            if slot is None:
                slot = self.alloc()
                cache["npu_slot"] = slot
                if not key:
                    self.reset(slot)
                cache["npu_produced"] = 0
            if prefix:
                if key:
                    cached_priming_groups.setdefault(key, []).append(index)
                else:
                    priming_groups.setdefault(prefix, []).append(index)
            generated = length - prefix
            if generated:
                groups.setdefault(generated, []).append(index)
            else:
                outputs[index] = torch.zeros(0, device=self.device, dtype=torch.float32)

        # Prime equal-length references independently, then batch generated frames
        # together even when their requests had different reference lengths.
        for key, indices in cached_priming_groups.items():
            first = indices[0]
            prefix = prefixes[first]
            slots = [int(caches[index]["npu_slot"]) for index in indices]
            self._prime_cached_reference(key, slots, codes[first : first + 1, :, :prefix])
            for index in indices:
                caches[index]["npu_produced"] = prefix
        for prefix, indices in priming_groups.items():
            for start in range(0, len(indices), self.max_batch):
                chosen = indices[start : start + self.max_batch]
                slots = [int(caches[index]["npu_slot"]) for index in chosen]
                reference = codes[chosen, :, :prefix].to(device=self.device, dtype=torch.long)
                self._decode_same_length(slots, reference, emit=False)
                for index in chosen:
                    caches[index]["npu_produced"] = prefix
        for length, indices in groups.items():
            for start in range(0, len(indices), self.max_batch):
                chosen = indices[start : start + self.max_batch]
                slots = [int(caches[index]["npu_slot"]) for index in chosen]
                if any(prefixes[index] for index in chosen):
                    chosen_codes = torch.stack(
                        [codes[index, :, prefixes[index] : prefixes[index] + length] for index in chosen]
                    )
                else:
                    chosen_codes = codes[chosen, :, :length]
                waveform = self._decode_same_length(slots, chosen_codes.to(device=self.device, dtype=torch.long))
                assert waveform is not None
                for row, index in enumerate(chosen):
                    outputs[index] = self._trim(waveform[row, 0], caches[index], length)
        if any(output is None for output in outputs):
            raise RuntimeError("incremental codec did not produce an output for every request")
        return [output for output in outputs if output is not None]

    def _trim(self, waveform: torch.Tensor, cache: dict, length: int) -> torch.Tensor:
        produced = int(cache.get("npu_produced", 0))
        prefix = int(cache.get("prefix_frames") or 0)
        skip_first = bool(cache.get("skip_first_audio")) and prefix == 0
        emit_from = prefix + (1 if skip_first else 0)
        keep_lo = max(produced, emit_from)
        keep_hi = produced + length
        cache["npu_produced"] = keep_hi
        if keep_lo >= keep_hi:
            return waveform.new_zeros(0)
        return waveform[(keep_lo - produced) * self.upsample : (keep_hi - produced) * self.upsample]

    def close(self) -> None:
        if self.graphs:
            torch.npu.synchronize()
            for graph in self.graphs.values():
                if graph.graph is not None:
                    graph.graph.reset()
            self.graphs.clear()
        self._reference_states.clear()
        self._reference_cache_hits.clear()
