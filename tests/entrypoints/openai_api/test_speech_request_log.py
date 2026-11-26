# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
import socket
from datetime import datetime

import pytest

from vllm_omni.entrypoints.openai import speech_request_log
from vllm_omni.entrypoints.openai.speech_request_log import initialize_speech_request_log, log_speech_input

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_speech_request_log_preserves_concurrent_inputs(tmp_path):
    directory = tmp_path / "logs"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        prefix = f"request_raw_text_{listener.getsockname()[1]}_"
        path = initialize_speech_request_log(listener, directory)
        assert path is not None and path.parent == directory
        assert path.name.startswith(prefix) and path.suffix == ".log"
        datetime.strptime(path.stem.removeprefix(prefix), "%Y%m%dT%H%M%S.%fZ")
    texts = [f"中文 {index}\n第二行  " for index in range(64)]
    batch = ["另一条", " 多行\n批量文本 "]

    async def write_requests():
        await asyncio.gather(*(log_speech_input(path, text, 1700000000.125) for text in [*texts, batch]))

    asyncio.run(write_requests())
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(texts) + 1
    assert {row["timestamp"] for row in records} == {"2023-11-14T22:13:20.125+00:00"}
    assert sorted(row["text"] for row in records if isinstance(row["text"], str)) == sorted(texts)
    assert [row["text"] for row in records if isinstance(row["text"], list)] == [batch]


def test_restart_creates_a_new_log_and_preserves_previous_logs(tmp_path, monkeypatch):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 26, 14, 23, 15, 123456, tzinfo=tz)

    monkeypatch.setattr(speech_request_log, "datetime", FixedClock)
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        first_log = initialize_speech_request_log(first, tmp_path)
        second_log = initialize_speech_request_log(second, tmp_path)
        assert first_log is not None and second_log is not None and first_log != second_log
        assert first_log.name == f"request_raw_text_{first.getsockname()[1]}_20261126T142315.123456Z.log"
        assert second_log.name == f"request_raw_text_{second.getsockname()[1]}_20261126T142315.123456Z.log"
        asyncio.run(log_speech_input(first_log, "服务一", 1700000000))
        asyncio.run(log_speech_input(second_log, "服务二", 1700000000))

        restarted_log = initialize_speech_request_log(first, tmp_path)
        assert restarted_log is not None and restarted_log not in (first_log, second_log)
        assert restarted_log.name == f"request_raw_text_{first.getsockname()[1]}_20261126T142315.123457Z.log"
        assert restarted_log.read_bytes() == b""
        asyncio.run(log_speech_input(restarted_log, "服务一重启后", 1700000000))
        assert json.loads(restarted_log.read_text(encoding="utf-8"))["text"] == "服务一重启后"
        assert json.loads(first_log.read_text(encoding="utf-8"))["text"] == "服务一"
        assert json.loads(second_log.read_text(encoding="utf-8"))["text"] == "服务二"
