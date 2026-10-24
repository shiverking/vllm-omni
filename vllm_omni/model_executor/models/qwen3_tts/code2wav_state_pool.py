# SPDX-License-Identifier: Apache-2.0
"""Fixed request slots and graph-owned batched state transactions."""

from types import SimpleNamespace

import torch

from .code2wav_optimization import fixed_stage


class RequestStatePool:
    capacity = 16
    dummy_rows = 4

    def __init__(self, decoder, device):
        parameter = next(decoder.parameters())
        window = int(decoder.config.sliding_window)
        self.quantized = torch.zeros(40, decoder.config.codebook_dim, window, dtype=parameter.dtype, device=device)
        self.conv = torch.zeros(40, window - 2, decoder.config.latent_dim, dtype=parameter.dtype, device=device)
        self.owners = {}
        self.free = list(reversed(range(self.capacity)))

    def acquire(self, cache):
        identity = id(cache)
        if identity in self.owners:
            return
        if not self.free:
            raise RuntimeError("Code2Wav state pool exhausted (16 live requests)")
        slot = self.free.pop()
        self.owners[identity] = cache
        cache["_c2w_slot"] = slot
        cache["_c2w_bank"] = 0
        if "suffix_quantized" in cache:
            self.store(cache, cache["suffix_quantized"], cache["suffix_conv"], bank=0)

    def publish(self, cache, bank, qlength, clength):
        index = cache["_c2w_slot"] * 2 + bank
        cache["_c2w_bank"] = bank
        cache["suffix_quantized"] = self.quantized[index : index + 1, :, :qlength]
        cache["suffix_conv"] = self.conv[index : index + 1, :clength, :]
        # Remove legacy ping-pong aliases after eager import.
        for name in list(cache):
            if name.startswith("_suffix_"):
                del cache[name]

    def store(self, cache, quantized, conv, bank):
        index = cache["_c2w_slot"] * 2 + bank
        qlength, clength = quantized.shape[-1], conv.shape[1]
        self.quantized[index : index + 1, :, :qlength].copy_(quantized)
        self.conv[index : index + 1, :clength, :].copy_(conv)
        self.publish(cache, bank, qlength, clength)

    def release(self, cache):
        if self.owners.pop(id(cache), None) is not None:
            self.free.append(cache.pop("_c2w_slot"))
            cache.pop("_c2w_bank", None)
            # A released request must never retain a view into a reused slot.
            cache.clear()
            cache["prefix_frames"] = 0

    def clear(self):
        for cache in list(self.owners.values()):
            self.release(cache)

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in (self.quantized, self.conv))


class PooledGraphMixin:
    @property
    def pooled(self):
        return self.optimization in ("pooled", "fused")

    def ensure_pool(self, device):
        if getattr(self, "state_pool", None) is None:
            self.state_pool = RequestStatePool(self.decoder, device)

    def release_requests(self, caches):
        if self.pooled and getattr(self, "state_pool", None) is not None:
            for cache in caches:
                self.state_pool.release(cache)

    def reset_pool(self):
        if self.pooled and getattr(self, "state_pool", None) is not None:
            self.state_pool.clear()

    def import_histories(self, caches, device):
        """Benchmark snapshot restoration, explicitly outside the timed call."""
        if self.pooled:
            self.ensure_pool(device)
            for cache in caches:
                self.state_pool.acquire(cache)

    def _capture_pool(self, batch, target, device, dtype):
        self.ensure_pool(device)
        first = target is None
        previous = 0 if first else self._xvec_previous_frames_by_target[target]
        length = self.initial_chunk_frames if first else target
        new_frames = length - previous
        pool = self.state_pool
        static = dict(
            codes=torch.zeros(batch, self.num_quantizers, new_frames, device=device, dtype=dtype),
            read=torch.arange(32, 32 + 2 * batch, 2, device=device, dtype=torch.long),
            write=torch.arange(33, 33 + 2 * batch, 2, device=device, dtype=torch.long),
        )
        static["indices"] = torch.stack((static["read"], static["write"]))
        static["read"], static["write"] = static["indices"][0], static["indices"][1]
        # One host-to-device copy stages all row indexes per replay.
        host = torch.empty(2, batch, dtype=torch.long)
        caches = self._make_dummy_xvec_cache(batch, device, next(self.decoder.parameters()).dtype)
        state = dict(input=static, cache=caches, host=host, host_array=host.numpy())

        def forward():
            with fixed_stage(self, batch, length):
                if first:
                    wav = self.decoder._decode_xvec_first_chunk(static["codes"], caches)
                    q, c = caches["suffix_quantized"], caches["suffix_conv"]
                else:
                    q = pool.quantized[:, :, : min(previous, self.prefix_length)].index_select(0, static["read"])
                    c = pool.conv[:, : self._cached_conv_frames(previous), :].index_select(0, static["read"])
                    wav, q, c = self._run_combined_suffix("xvec", static["codes"], q, c, caches, target, None)
                pool.quantized[:, :, : q.shape[-1]].index_copy_(0, static["write"], q)
                pool.conv[:, : c.shape[1], :].index_copy_(0, static["write"], c)
                return wav, q, c

        # Populate fixed constants outside capture, then warm up the actual
        # gather/decode/scatter execution. CPU uses this same forward for tests.
        with fixed_stage(self, batch, length):
            state["output"] = forward()
        if device.type == "cpu":

            def replay():
                state["output"] = forward()

            state["graph"] = SimpleNamespace(replay=replay)
        else:
            self._synchronize(device)
            graph = self._new_graph()
            with self._graph_context(graph):
                state["output"] = forward()
            state["graph"] = graph
        if first:
            self.xvec_prefix_states[batch] = state
        else:
            self.combined_states[("xvec", batch, target)] = state

    def _pool_replay(self, state, codes_list, caches, target):
        pool = self.state_pool
        static = state["input"]
        batch = static["codes"].shape[0]
        count = len(codes_list)
        for row in range(batch):
            if row < count:
                cache = caches[row]
                pool.acquire(cache)
                read = cache["_c2w_slot"] * 2 + cache["_c2w_bank"]
                write = cache["_c2w_slot"] * 2 + 1 - cache["_c2w_bank"]
                static["codes"][row].copy_(codes_list[row][0])
            else:
                read, write = 32 + row * 2, 33 + row * 2
            state["host_array"][0, row] = read
            state["host_array"][1, row] = write
        static["codes"][count:].zero_()
        static["indices"].copy_(state["host"])
        state["graph"].replay()
        wav, q, c = state["output"]
        # One ownership transfer for the batch; returned row views own this copy.
        output = wav[:count].clone()
        for row, cache in enumerate(caches):
            if target is None:
                for key in ("ref_hidden", "ref_conv", "prefix_hidden"):
                    cache[key] = state["cache"][key][row : row + 1]
                cache["past_key_values"] = self.decoder._slice_dynamic_cache(state["cache"]["past_key_values"], row)
                cache["decoder_prefix_frames"] = 0
            pool.publish(cache, 1 - cache["_c2w_bank"], q.shape[-1], c.shape[1])
            cache["suffix_frames"] = self.initial_chunk_frames if target is None else target
        self._record_graph_hit("xvec_prefix" if target is None else f"suffix_{target}", batch, count)
        return [output[i : i + 1] for i in range(count)]

    def _pool_suffix(self, target, codes, caches, lengths):
        previous = self._xvec_previous_frames_by_target[target]
        expected = target - previous
        outputs = [None] * len(codes)
        exact = [i for i, n in enumerate(lengths) if n == expected]
        for i, n in enumerate(lengths):
            if n != expected:
                outputs[i] = self._decode_request_fallback(codes[i], caches[i])
        if exact:
            batch = self._select_batch_bucket(len(exact), {1, 2, 4})
            result = self._pool_replay(
                self.combined_states[("xvec", batch, target)],
                [codes[i] for i in exact],
                [caches[i] for i in exact],
                target,
            )
            for i, output in zip(exact, result, strict=True):
                outputs[i] = output
        return outputs
