# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Raw speech inputs recorded once at the API boundary."""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from socket import AF_INET, AF_INET6, socket
from threading import Lock

logger = logging.getLogger(__name__)

_LOG_DIRECTORY = Path("/cache/y00899301/server_log")
_WRITE_LOCK = Lock()


def initialize_speech_request_log(listener: socket, directory: Path = _LOG_DIRECTORY) -> Path | None:
    """Create one log for this TCP listener and server start."""
    if listener.family not in (AF_INET, AF_INET6):
        return None
    port = listener.getsockname()[1]
    started_at = datetime.now(timezone.utc)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        while True:
            path = directory / f"request_raw_text_{port}_{started_at:%Y%m%dT%H%M%S.%fZ}.log"
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                started_at += timedelta(microseconds=1)
                continue
            os.close(fd)
            return path
    except OSError:
        logger.exception("Could not initialize speech request log for port %d in %s", port, directory)
        return None


def _append_record(path: Path, record: bytes) -> None:
    with _WRITE_LOCK:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab", buffering=0) as log:
            # One append write keeps records from different API workers together on Linux.
            if log.write(record) != len(record):
                raise OSError("Incomplete speech request log write")


async def log_speech_input(path: Path | None, text: str | list[str], timestamp: float) -> None:
    if path is None:
        return
    record = {
        "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds"),
        "text": text,
    }
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8", errors="backslashreplace")
    try:
        await asyncio.to_thread(_append_record, path, line)
    except OSError:
        logger.exception("Could not append to speech request log at %s", path)
