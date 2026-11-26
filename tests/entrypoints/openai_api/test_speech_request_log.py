# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
import socket

import pytest

from vllm_omni.entrypoints.openai.speech_request_log import initialize_speech_request_log, log_speech_input

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_speech_request_log_preserves_concurrent_inputs(tmp_path):
    directory = tmp_path / "logs"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        path = directory / f"request_raw_text_{listener.getsockname()[1]}.log"
        assert initialize_speech_request_log(listener, directory) == path
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


def test_restart_clears_only_the_log_for_that_port(tmp_path):
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        first_log = initialize_speech_request_log(first, tmp_path)
        second_log = initialize_speech_request_log(second, tmp_path)
        assert first_log is not None and second_log is not None and first_log != second_log
        asyncio.run(log_speech_input(first_log, "服务一", 1700000000))
        asyncio.run(log_speech_input(second_log, "服务二", 1700000000))

        assert initialize_speech_request_log(first, tmp_path) == first_log
        assert first_log.read_bytes() == b""
        asyncio.run(log_speech_input(first_log, "服务一重启后", 1700000000))
        assert json.loads(first_log.read_text(encoding="utf-8"))["text"] == "服务一重启后"
        assert json.loads(second_log.read_text(encoding="utf-8"))["text"] == "服务二"
