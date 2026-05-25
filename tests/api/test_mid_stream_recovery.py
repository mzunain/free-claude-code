"""Tests for post-commit mid-stream drop recovery in api.services."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from api.services import (
    _MID_STREAM_TRANSPORT_TYPES,
    _format_mid_stream_error_event,
    _mid_stream_error_message,
    _stream_with_mid_drop_recovery,
)


# ---- _mid_stream_error_message ------------------------------------------------


def test_error_message_remote_protocol_error_includes_provider():
    msg = _mid_stream_error_message(
        httpx.RemoteProtocolError("peer closed"), "nvidia_nim/qwen3-coder"
    )
    assert "dropped mid-stream" in msg
    assert "nvidia_nim/qwen3-coder" in msg


def test_error_message_remote_protocol_error_without_provider():
    msg = _mid_stream_error_message(httpx.RemoteProtocolError("peer closed"), None)
    assert "dropped mid-stream" in msg
    assert "via" not in msg


def test_error_message_read_timeout():
    msg = _mid_stream_error_message(httpx.ReadTimeout("slow"), "zai/glm-4.6")
    assert "timed out" in msg
    assert "ReadTimeout" in msg
    assert "zai/glm-4.6" in msg


def test_error_message_network_error():
    msg = _mid_stream_error_message(httpx.NetworkError("net"), "open_router/x")
    assert "network error" in msg
    assert "NetworkError" in msg


def test_error_message_plain_connection_error():
    msg = _mid_stream_error_message(ConnectionError("closed"), "nvidia_nim/foo")
    assert "connection error" in msg.lower()
    assert "nvidia_nim/foo" in msg


def test_error_message_generic_exception_truncates_detail():
    big = "x" * 1000
    msg = _mid_stream_error_message(RuntimeError(big), "p/m")
    # Detail clipped to 200 chars (plus prefix and label)
    assert len(msg) < 400
    assert "RuntimeError" in msg


# ---- _format_mid_stream_error_event ------------------------------------------


def _parse_sse_event(payload: str) -> tuple[str, dict]:
    """Parse a single Anthropic SSE frame into (event_type, data_dict)."""
    lines = [line for line in payload.strip().splitlines() if line]
    event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
    data_line = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
    return event, json.loads(data_line)


def test_format_mid_stream_error_event_is_valid_sse():
    exc = httpx.RemoteProtocolError("peer closed connection")
    raw = _format_mid_stream_error_event(exc, "nvidia_nim/qwen")
    event, data = _parse_sse_event(raw)
    assert event == "error"
    assert data["type"] == "error"
    assert data["error"]["type"] == "api_error"
    assert "dropped mid-stream" in data["error"]["message"]
    assert "nvidia_nim/qwen" in data["error"]["message"]


def test_format_mid_stream_error_event_terminates_with_blank_line():
    raw = _format_mid_stream_error_event(httpx.ReadTimeout("t"), "p/m")
    # SSE frames must end with a blank line so clients flush the event
    assert raw.endswith("\n\n")


# ---- _stream_with_mid_drop_recovery ------------------------------------------


async def _collect(gen: AsyncIterator[str]) -> list[str]:
    out = []
    async for chunk in gen:
        out.append(chunk)
    return out


@pytest.mark.asyncio
async def test_recovery_passes_through_clean_stream():
    async def clean() -> AsyncIterator[str]:
        yield "event: message_start\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"

    chunks = await _collect(
        _stream_with_mid_drop_recovery(clean(), "req_test", "nvidia_nim/qwen")
    )
    assert chunks == [
        "event: message_start\ndata: {}\n\n",
        "event: message_stop\ndata: {}\n\n",
    ]


@pytest.mark.asyncio
async def test_recovery_emits_clean_error_on_remote_protocol_error():
    async def drops_after_first() -> AsyncIterator[str]:
        yield "event: message_start\ndata: {}\n\n"
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )

    chunks = await _collect(
        _stream_with_mid_drop_recovery(
            drops_after_first(), "req_test", "nvidia_nim/qwen"
        )
    )
    # First chunk is the real upstream event, second is the synthesized error frame
    assert len(chunks) == 2
    assert chunks[0].startswith("event: message_start")
    event, data = _parse_sse_event(chunks[1])
    assert event == "error"
    assert data["error"]["type"] == "api_error"
    assert "nvidia_nim/qwen" in data["error"]["message"]


@pytest.mark.asyncio
async def test_recovery_emits_clean_error_on_read_timeout():
    async def drops() -> AsyncIterator[str]:
        yield "event: content_block_delta\ndata: {}\n\n"
        raise httpx.ReadTimeout("upstream stalled")

    chunks = await _collect(
        _stream_with_mid_drop_recovery(drops(), "req_test", "zai/glm-4.6")
    )
    assert len(chunks) == 2
    event, data = _parse_sse_event(chunks[1])
    assert event == "error"
    assert "timed out" in data["error"]["message"]


@pytest.mark.asyncio
async def test_recovery_emits_clean_error_on_generic_exception():
    """Non-transport errors after commit should also close cleanly with an SSE error."""

    async def crashes() -> AsyncIterator[str]:
        yield "event: message_start\ndata: {}\n\n"
        raise RuntimeError("provider impl exploded")

    chunks = await _collect(
        _stream_with_mid_drop_recovery(crashes(), "req_test", "p/m")
    )
    assert len(chunks) == 2
    event, data = _parse_sse_event(chunks[1])
    assert event == "error"
    assert "RuntimeError" in data["error"]["message"]


@pytest.mark.asyncio
async def test_recovery_propagates_cancelled_error():
    """CancelledError must propagate — swallowing it breaks task shutdown semantics."""

    async def cancels() -> AsyncIterator[str]:
        yield "event: message_start\ndata: {}\n\n"
        raise asyncio.CancelledError

    gen = _stream_with_mid_drop_recovery(cancels(), "req_test", "p/m")
    # First yield works
    first = await gen.__anext__()
    assert first.startswith("event: message_start")
    # Second iteration must raise (not swallow) CancelledError
    with pytest.raises(asyncio.CancelledError):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_recovery_yields_error_when_drop_happens_before_first_chunk():
    """If the inner stream raises immediately, the wrapper still emits a parseable error."""

    async def drops_immediately() -> AsyncIterator[str]:
        if False:
            yield ""  # make this an async generator
        raise httpx.RemoteProtocolError("immediate")

    chunks = await _collect(
        _stream_with_mid_drop_recovery(
            drops_immediately(), "req_test", "nvidia_nim/qwen"
        )
    )
    assert len(chunks) == 1
    event, _ = _parse_sse_event(chunks[0])
    assert event == "error"


# ---- _MID_STREAM_TRANSPORT_TYPES ---------------------------------------------


def test_transport_types_cover_known_drop_classes():
    """All exception classes we have seen drop NIM connections must be caught."""
    seen_in_logs = [
        httpx.RemoteProtocolError("x"),
        httpx.ReadError("x"),
        httpx.WriteError("x"),
        httpx.ReadTimeout("x"),
        httpx.WriteTimeout("x"),
        httpx.PoolTimeout("x"),
        httpx.NetworkError("x"),
        ConnectionError("x"),
    ]
    for exc in seen_in_logs:
        assert isinstance(exc, _MID_STREAM_TRANSPORT_TYPES), (
            f"{type(exc).__name__} not covered by transport-type tuple"
        )
