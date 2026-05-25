"""Application services for the Claude-compatible API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS, format_sse_event
from core.trace import api_messages_request_snapshot, trace_event, traced_async_stream
from providers.base import BaseProvider
from providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)

from .model_router import ModelRouter, RoutedMessagesRequest
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim", "opencode", "zai"})


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _parse_sse_event(raw: str) -> tuple[str | None, dict | None]:
    """Parse one SSE event chunk into ``(event_name, data_dict)``.

    Returns ``(None, None)`` if the chunk is malformed or has no JSON payload.
    """
    event_name: str | None = None
    data_str: str | None = None
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            piece = line[5:].strip()
            data_str = piece if data_str is None else data_str + piece
    if not event_name or data_str is None:
        return None, None
    try:
        return event_name, json.loads(data_str)
    except json.JSONDecodeError:
        return event_name, None


async def aggregate_sse_to_anthropic_response(
    sse_stream: AsyncIterator[str],
) -> dict[str, Any]:
    """Consume an Anthropic SSE stream and assemble a single Messages JSON response.

    Used to satisfy ``stream: false`` requests when the upstream gateway always
    streams. Builds content blocks (text / tool_use / thinking) from the
    block_start/delta/stop events and merges usage from message_delta.
    """
    message: dict[str, Any] = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": "",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    blocks: dict[int, dict[str, Any]] = {}
    tool_arg_buffers: dict[int, list[str]] = {}
    buffer = ""
    async for chunk in sse_stream:
        buffer += chunk
        while "\n\n" in buffer:
            event_text, buffer = buffer.split("\n\n", 1)
            name, data = _parse_sse_event(event_text)
            if name is None or data is None:
                continue
            if name == "message_start":
                msg = data.get("message", {})
                if isinstance(msg, dict):
                    for k in ("id", "model"):
                        if msg.get(k):
                            message[k] = msg[k]
                    usage = msg.get("usage")
                    if isinstance(usage, dict):
                        message["usage"]["input_tokens"] = usage.get(
                            "input_tokens", message["usage"]["input_tokens"]
                        )
                        message["usage"]["output_tokens"] = usage.get(
                            "output_tokens", message["usage"]["output_tokens"]
                        )
            elif name == "content_block_start":
                idx = data.get("index", 0)
                block = data.get("content_block", {})
                if isinstance(block, dict):
                    blocks[idx] = dict(block)
                    if block.get("type") == "tool_use":
                        tool_arg_buffers[idx] = []
                        blocks[idx].setdefault("input", {})
                    elif block.get("type") == "thinking":
                        blocks[idx].setdefault("thinking", "")
                    else:
                        blocks[idx].setdefault("text", "")
            elif name == "content_block_delta":
                idx = data.get("index", 0)
                delta = data.get("delta", {})
                if not isinstance(delta, dict):
                    continue
                block = blocks.setdefault(idx, {"type": "text", "text": ""})
                dtype = delta.get("type")
                if dtype == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif dtype == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get(
                        "thinking", ""
                    )
                elif dtype == "input_json_delta":
                    tool_arg_buffers.setdefault(idx, []).append(
                        delta.get("partial_json", "")
                    )
            elif name == "content_block_stop":
                idx = data.get("index", 0)
                if idx in tool_arg_buffers:
                    joined = "".join(tool_arg_buffers.pop(idx))
                    if joined:
                        try:
                            blocks[idx]["input"] = json.loads(joined)
                        except json.JSONDecodeError:
                            blocks[idx]["input"] = {"_raw": joined}
            elif name == "message_delta":
                delta = data.get("delta", {})
                if isinstance(delta, dict):
                    if "stop_reason" in delta:
                        message["stop_reason"] = delta["stop_reason"]
                    if "stop_sequence" in delta:
                        message["stop_sequence"] = delta["stop_sequence"]
                usage = data.get("usage", {})
                if isinstance(usage, dict) and "output_tokens" in usage:
                    message["usage"]["output_tokens"] = usage["output_tokens"]
            elif name == "message_stop":
                pass
    message["content"] = [blocks[k] for k in sorted(blocks.keys())]
    if message["stop_reason"] is None:
        message["stop_reason"] = "end_turn"
    return message


_CONVERSION_RETRYABLE_MARKERS = (
    "image blocks are not supported",
    "OpenAI chat conversion does not support",
    "use a native Anthropic transport provider",
    "use a vision-capable native Anthropic provider",
)


def _is_retryable_provider_error(exc: BaseException) -> bool:
    """Return True for errors that warrant trying the next provider in a chain."""
    if isinstance(exc, InvalidRequestError):
        # Provider-specific conversion limitations (e.g. an OpenAI-chat transport
        # rejecting image blocks) should fall through to a different provider that
        # can handle the feature, rather than failing the whole request.
        message = str(exc)
        return any(marker in message for marker in _CONVERSION_RETRYABLE_MARKERS)
    if isinstance(
        exc,
        (
            RateLimitError,
            OverloadedError,
            AuthenticationError,
            ServiceUnavailableError,
        ),
    ):
        return True
    if isinstance(exc, APIError):
        return exc.status_code >= 500 or exc.status_code in (
            401,
            403,
            404,
            408,
            410,
            429,
        )
    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
            httpx.ReadError,
        ),
    )


async def _continue_stream(
    stream: AsyncIterator[str], first_chunk: str
) -> AsyncIterator[str]:
    """Prepend ``first_chunk`` to an in-flight async stream."""
    yield first_chunk
    async for chunk in stream:
        yield chunk


# After ``STREAM_COMMIT`` we have already forwarded SSE bytes to the client and
# cannot transparently swap to another provider candidate without producing a
# malformed Anthropic Messages stream (duplicate ``message_start``, dangling
# content blocks, mismatched indices). Instead we trap the upstream transport
# failure and emit a top-level Anthropic ``event: error`` so Claude Code parses
# a structured error and applies its own retry policy, rather than seeing a
# truncated chunked-transfer body ("incomplete chunked read").
_MID_STREAM_TRANSPORT_TYPES: tuple[type[BaseException], ...] = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.NetworkError,
    ConnectionError,
)


def _mid_stream_error_message(
    exc: BaseException, provider_label: str | None
) -> str:
    """Human-readable summary of a mid-stream drop for the SSE error event."""
    if isinstance(exc, httpx.RemoteProtocolError):
        base = "Upstream connection dropped mid-stream"
    elif isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        base = f"Upstream stream timed out ({type(exc).__name__})"
    elif isinstance(exc, (httpx.ReadError, httpx.WriteError, httpx.NetworkError)):
        base = f"Upstream network error ({type(exc).__name__})"
    elif isinstance(exc, ConnectionError):
        base = "Upstream connection error"
    else:
        detail = str(exc)[:200]
        base = f"Upstream stream error ({type(exc).__name__}): {detail}".rstrip(": ")
    return f"{base} (via {provider_label})" if provider_label else base


def _format_mid_stream_error_event(
    exc: BaseException, provider_label: str | None
) -> str:
    """Build a top-level Anthropic SSE ``event: error`` frame for a mid-stream drop."""
    return format_sse_event(
        "error",
        {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": _mid_stream_error_message(exc, provider_label),
            },
        },
    )


async def _stream_with_mid_drop_recovery(
    inner: AsyncIterator[str],
    request_id: str,
    provider_label: str | None,
) -> AsyncIterator[str]:
    """Yield from ``inner``; on post-commit transport failure emit a clean SSE error.

    This guarantees Claude Code (or any other client) always sees a well-formed
    Anthropic Messages SSE stream: either the upstream completes normally, or we
    finish with a parseable ``event: error`` frame that the client can act on.
    Without this wrapper, an upstream ``RemoteProtocolError`` would propagate out
    of the response generator and Starlette would slam the chunked body shut, so
    the client only ever sees ``peer closed connection without sending complete
    message body``.
    """
    try:
        async for chunk in inner:
            yield chunk
    except asyncio.CancelledError:
        raise
    except _MID_STREAM_TRANSPORT_TYPES as exc:
        logger.warning(
            "MID_STREAM_DROP: request_id={} provider={} exc={} msg={}",
            request_id,
            provider_label or "unknown",
            type(exc).__name__,
            str(exc)[:200],
        )
        yield _format_mid_stream_error_event(exc, provider_label)
    except Exception as exc:
        logger.warning(
            "MID_STREAM_ERROR: request_id={} provider={} exc={} msg={}",
            request_id,
            provider_label or "unknown",
            type(exc).__name__,
            str(exc)[:200],
        )
        yield _format_mid_stream_error_event(exc, provider_label)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    async def create_message_non_streaming(
        self, request_data: MessagesRequest
    ) -> JSONResponse:
        """Aggregate the streaming response and return a single Messages JSON.

        Used to satisfy clients that send ``stream: false`` (or omit ``stream``,
        which Anthropic treats as non-streaming). The gateway always produces an
        SSE stream internally; this method consumes it and returns the assembled
        Anthropic Messages object as a JSON response.
        """
        streaming = self.create_message(request_data)
        if not isinstance(streaming, StreamingResponse):
            return streaming  # already a non-streaming response (e.g. optimization short-circuit)
        body = streaming.body_iterator
        try:
            message = await aggregate_sse_to_anthropic_response(body)
        finally:
            close = getattr(body, "aclose", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()
        return JSONResponse(message)

    def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            chain = self._model_router.resolve_messages_request_chain(request_data)
            primary = chain.primary

            # Single-candidate chains preserve the historical eager-raise contract:
            # if the sole provider is an openai-chat upstream that cannot handle the
            # request's web tool blocks, fail fast here rather than deferring to the
            # streaming generator. Multi-candidate chains let the fallback machinery
            # try the next provider instead.
            if (
                len(chain.candidates) == 1
                and primary.resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS
            ):
                tool_err = openai_chat_upstream_server_tool_error(
                    primary.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                primary.request
            ):
                input_tokens = self._token_counter(
                    primary.request.messages,
                    primary.request.system,
                    primary.request.tools,
                )
                trace_event(
                    stage="routing",
                    event="api.optimization.web_server_tool",
                    source="api",
                    model=primary.request.model,
                )
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        primary.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(primary.request, self._settings)
            if optimized is not None:
                trace_event(
                    stage="routing",
                    event="api.optimization.short_circuit",
                    source="api",
                    model=primary.request.model,
                )
                return optimized
            logger.debug("No optimization matched, routing to provider")

            # Emit request-level trace events eagerly so observers see the same
            # ingress timeline whether or not the body iterator is consumed.
            request_id = f"req_{uuid.uuid4().hex[:12]}"
            trace_event(
                stage="ingress",
                event="api.request.received",
                source="api",
                message_count=len(primary.request.messages),
                snapshot=api_messages_request_snapshot(primary.request),
            )
            if self._settings.log_raw_api_payloads:
                logger.debug(
                    "FULL_PAYLOAD [{}]: {}",
                    request_id,
                    primary.request.model_dump(),
                )

            if len(chain.candidates) == 1:
                # Single-candidate chain: preserve the historical eager-bind behaviour
                # so synchronous provider errors surface as HTTP 500 here.
                return self._stream_single_candidate(primary, request_id)

            return anthropic_sse_streaming_response(
                self._stream_with_fallback(chain.candidates, request_id),
            )

        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    def _stream_single_candidate(
        self, routed: RoutedMessagesRequest, request_id: str
    ) -> StreamingResponse:
        """Eager single-candidate path. Mirrors the pre-chain behaviour."""
        resolved = routed.resolved
        provider = self._provider_getter(resolved.provider_id)
        provider.preflight_stream(
            routed.request, thinking_enabled=resolved.thinking_enabled
        )
        trace_event(
            stage="routing",
            event="api.route.resolved",
            source="api",
            provider_id=resolved.provider_id,
            provider_model=resolved.provider_model,
            provider_model_ref=resolved.provider_model_ref,
            gateway_model=routed.request.model,
            thinking_enabled=resolved.thinking_enabled,
        )
        candidate_tag = f"{resolved.provider_id}/{resolved.provider_model}"
        with logger.contextualize(request_id=request_id):
            logger.info(
                "API_REQUEST: request_id={} model={} messages={}",
                request_id,
                routed.request.model,
                len(routed.request.messages),
            )
            input_tokens = self._token_counter(
                routed.request.messages,
                routed.request.system,
                routed.request.tools,
            )
            streamed = traced_async_stream(
                provider.stream_response(
                    routed.request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    thinking_enabled=resolved.thinking_enabled,
                ),
                stage="egress",
                source="api",
                complete_event="api.response.stream_completed",
                interrupted_event="api.response.stream_interrupted",
                chunk_event=None,
                extra={
                    "request_id": request_id,
                    "provider_id": resolved.provider_id,
                    "gateway_model": routed.request.model,
                },
            )
            return anthropic_sse_streaming_response(
                _stream_with_mid_drop_recovery(streamed, request_id, candidate_tag)
            )

    async def _stream_with_fallback(
        self,
        candidates: tuple[RoutedMessagesRequest, ...],
        request_id: str,
    ) -> AsyncIterator[str]:
        """Iterate candidates in order, falling back on retryable upstream errors.

        Only fires fallback before any SSE bytes commit to the client. Each provider
        is asked to ``raise_pre_stream_errors`` so an upstream 429/5xx/auth/conversion
        failure surfaces as an exception we can catch and route to the next candidate.
        """
        last_error: BaseException | None = None
        total = len(candidates)
        single = total == 1
        for i, routed in enumerate(candidates):
            resolved = routed.resolved
            is_last = i == total - 1
            candidate_tag = f"{resolved.provider_id}/{resolved.provider_model}"

            if resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    last_error = InvalidRequestError(tool_err)
                    if is_last:
                        raise last_error
                    logger.warning(
                        "FALLBACK_SKIP: candidate={} ({}/{}) incompatible with Anthropic web tools, trying next",
                        candidate_tag,
                        i + 1,
                        total,
                    )
                    continue

            try:
                provider = self._provider_getter(resolved.provider_id)
                provider.preflight_stream(
                    routed.request,
                    thinking_enabled=resolved.thinking_enabled,
                )
            except ProviderError as e:
                last_error = e
                if is_last or not _is_retryable_provider_error(e):
                    raise
                logger.warning(
                    "FALLBACK_PREFLIGHT: candidate={} ({}/{}) {} status={}, trying next",
                    candidate_tag,
                    i + 1,
                    total,
                    type(e).__name__,
                    getattr(e, "status_code", "n/a"),
                )
                continue
            except Exception as e:
                last_error = e
                if is_last:
                    raise
                logger.warning(
                    "FALLBACK_PREFLIGHT: candidate={} ({}/{}) raised {}, trying next",
                    candidate_tag,
                    i + 1,
                    total,
                    type(e).__name__,
                )
                continue

            trace_event(
                stage="routing",
                event="api.route.resolved",
                source="api",
                provider_id=resolved.provider_id,
                provider_model=resolved.provider_model,
                provider_model_ref=resolved.provider_model_ref,
                gateway_model=routed.request.model,
                thinking_enabled=resolved.thinking_enabled,
                candidate_index=i + 1,
                candidate_total=total,
            )
            logger.info(
                "API_REQUEST: request_id={} candidate={}/{} provider={} model={} messages={}",
                request_id,
                i + 1,
                total,
                resolved.provider_id,
                routed.request.model,
                len(routed.request.messages),
            )
            with logger.contextualize(request_id=request_id):
                input_tokens = self._token_counter(
                    routed.request.messages,
                    routed.request.system,
                    routed.request.tools,
                )
                stream = provider.stream_response(
                    routed.request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    thinking_enabled=resolved.thinking_enabled,
                    raise_pre_stream_errors=not single,
                )

                try:
                    first_chunk = await stream.__anext__()
                except StopAsyncIteration:
                    return
                except ProviderError as e:
                    last_error = e
                    if is_last or not _is_retryable_provider_error(e):
                        raise
                    logger.warning(
                        "FALLBACK_STREAM: candidate={} ({}/{}) {} status={}, trying next",
                        candidate_tag,
                        i + 1,
                        total,
                        type(e).__name__,
                        getattr(e, "status_code", "n/a"),
                    )
                    continue
                except Exception as e:
                    last_error = e
                    if is_last or not _is_retryable_provider_error(e):
                        raise
                    logger.warning(
                        "FALLBACK_STREAM: candidate={} ({}/{}) raised {}, trying next",
                        candidate_tag,
                        i + 1,
                        total,
                        type(e).__name__,
                    )
                    continue

                logger.info(
                    "STREAM_COMMIT: request_id={} candidate={}/{} provider={} model={}",
                    request_id,
                    i + 1,
                    total,
                    resolved.provider_id,
                    resolved.provider_model,
                )
                streamed = traced_async_stream(
                    _continue_stream(stream, first_chunk),
                    stage="egress",
                    source="api",
                    complete_event="api.response.stream_completed",
                    interrupted_event="api.response.stream_interrupted",
                    chunk_event=None,
                    extra={
                        "request_id": request_id,
                        "provider_id": resolved.provider_id,
                        "gateway_model": routed.request.model,
                    },
                )
                # Post-commit: cannot silently swap candidates. Convert mid-stream
                # transport drops into a clean Anthropic SSE error so the client
                # parses a structured error instead of an aborted chunked body.
                async for chunk in _stream_with_mid_drop_recovery(
                    streamed, request_id, candidate_tag
                ):
                    yield chunk
                return

        if last_error is not None:
            raise last_error

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="api.route.resolved",
                    source="api",
                    kind="count_tokens",
                    provider_id=routed.resolved.provider_id,
                    provider_model=routed.resolved.provider_model,
                    provider_model_ref=routed.resolved.provider_model_ref,
                    gateway_model=routed.request.model,
                )
                trace_event(
                    stage="ingress",
                    event="api.count_tokens.completed",
                    source="api",
                    message_count=len(routed.request.messages),
                    input_tokens=tokens,
                    snapshot=api_messages_request_snapshot(routed.request),
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e
