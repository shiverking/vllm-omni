# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-step device grant integration for the two-stage TTS experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path

from vllm_omni.core.sched.playback_broker import rpc


def worker_begin(worker, grant, request_map, retired):
    model = worker.model_runner.get_model()
    session = getattr(getattr(model, "code_predictor", None), "playback", None)
    if session is None and grant["stage"] != "code2wav":
        raise RuntimeError("playback grant reached a Talker without an initialized PlaybackSession")
    if session is not None:
        session.begin(grant, request_map, retired)


def worker_end(worker):
    import torch

    # A permit must not be released while device work is still queued.
    torch.npu.synchronize()
    model = worker.model_runner.get_model()
    session = getattr(getattr(model, "code_predictor", None), "playback", None)
    result = dict(completed_frames={}, cp_ms=0, events=[])
    if session is not None:
        result.update(completed_frames=session.completed, cp_ms=session.cp_ms, events=session.events)
        session.grant = None
    return result


class PlaybackStage:
    def __init__(self, core, config_path, stage_id):
        self.core = core
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.url, self.stage_id = self.config["broker_url"], stage_id
        if stage_id not in (0, 1):
            raise ValueError("playback experiment supports Talker and Code2Wav only")
        self.previous = set()

    def priorities(self):
        scheduler = self.core.scheduler
        mapping = {rid: r.external_req_id for rid, r in scheduler.requests.items()}
        if not mapping or self.config["service_method"] == "native":
            return mapping
        priorities = rpc(self.url, "priorities", dict(ids=list(set(mapping.values()))))

        def key(r):
            return (*priorities[r.external_req_id], r.arrival_time, r.request_id)

        scheduler.running.sort(key=key)
        # Reorder queued admission without freeing/recomputing any resident KV.
        waiting = list(scheduler.waiting)
        if waiting:
            from vllm.v1.core.sched.request_queue import create_request_queue

            scheduler.waiting = create_request_queue(scheduler.policy)
            for request in sorted(waiting, key=key):
                scheduler.waiting.add_request(request)
        return mapping

    def step(self):
        core = self.core
        if not core.scheduler.has_requests():
            return {}, False
        mapping = self.priorities()
        was_prefill = {rid: r.num_computed_tokens < r.num_prompt_tokens for rid, r in core.scheduler.requests.items()}
        output = core.scheduler.schedule()
        scheduled = output.num_scheduled_tokens
        grant = None
        try:
            if output.total_num_scheduled_tokens:
                # Mixed prefill/decode steps are conservatively classified as
                # prefill; all their NPU work still holds the single permit.
                prefill = any(was_prefill[r] for r in scheduled)
                stage = "code2wav" if self.stage_id == 1 else "prefill" if prefill else "generate"
                ids = [mapping[r] for r in scheduled]
                grant = rpc(self.url, "claim", dict(ids=ids, stage=stage))
                retired = list(self.previous - set(mapping.values()))
                self.previous = set(mapping.values())
                core.model_executor.collective_rpc(worker_begin, args=(grant, mapping, retired))
            start = time.perf_counter()
            future = core.model_executor.execute_model(output, non_block=True)
            grammar = core.scheduler.get_grammar_bitmask(output)
            with core.log_error_detail(output), core.log_iteration_details(output):
                result = future.result()
                if result is None:
                    result = core.model_executor.sample_tokens(grammar)
            if grant is not None:
                worker_rows = core.model_executor.collective_rpc(worker_end)
                if len(worker_rows) != 1:
                    raise ValueError("one shared-card worker per stage is required")
                rpc(
                    self.url,
                    "release",
                    dict(token=grant["token"], wall_ms=(time.perf_counter() - start) * 1000, **worker_rows[0]),
                )
                grant = None
        except Exception as exc:
            if grant is not None:
                # Fail closed. Never pretend failed/captured work completed.
                rpc(self.url, "release", dict(token=grant["token"], error=repr(exc)))
            raise
        core._process_aborts_queue()
        return core.scheduler.update_from_output(output, result), output.total_num_scheduled_tokens > 0
