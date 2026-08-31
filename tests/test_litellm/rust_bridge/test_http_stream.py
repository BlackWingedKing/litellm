from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from pydantic import BaseModel

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.rust_bridge import http_stream as bridge


class _FakeStream:
    def __init__(self, chunks: tuple[bytes, ...], error: Exception | None = None) -> None:
        self.status_code: Final = 200
        self.headers: Final = (("content-type", "text/event-stream"), ("x-upstream", "test"))
        self._chunks: Final = iter(chunks)
        self._error: Final = error
        self._raised = False
        self.closed = False

    def next_bytes(self) -> bytes | None:
        if self.closed:
            return None
        try:
            return next(self._chunks)
        except StopIteration:
            if self._error is not None and not self._raised:
                self._raised = True
                raise self._error
            return None

    async def anext_bytes(self) -> bytes | None:
        return self.next_bytes()

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _FactoryCall:
    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes
    open_timeout_seconds: float | None
    read_idle_timeout_seconds: float | None


class _RecordingFactory:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks: Final = chunks
        self.calls: tuple[_FactoryCall, ...] = ()

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        open_timeout_seconds: float | None,
        read_idle_timeout_seconds: float | None,
    ) -> _FakeStream:
        stream: Final = _FakeStream(self._chunks)
        self.calls += (_FactoryCall(method, url, headers, body, open_timeout_seconds, read_idle_timeout_seconds),)
        return stream


class _RecordingAsyncFactory:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks: Final = chunks
        self.calls: tuple[_FactoryCall, ...] = ()

    async def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        open_timeout_seconds: float | None,
        read_idle_timeout_seconds: float | None,
    ) -> _FakeStream:
        self.calls += (_FactoryCall(method, url, headers, body, open_timeout_seconds, read_idle_timeout_seconds),)
        return _FakeStream(self._chunks)


class _FakeUpstreamError(Exception):
    pass


class _FakeDeclined(Exception):
    pass


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstreamError


@pytest.fixture(autouse=True)
def reset_bridge() -> Iterator[None]:
    bridge.set_rust_http_stream(open_stream=None, aopen_stream=None, capability=None)
    yield
    bridge.set_rust_http_stream(open_stream=None, aopen_stream=None, capability=None)


def _adapter(stream: _FakeStream) -> bridge.HttpResponseStreamAdapter:
    return bridge.HttpResponseStreamAdapter(stream, httpx.Request("POST", "https://example.test/v1/messages"))


async def _collect_bytes(chunks: AsyncIterator[bytes]) -> tuple[bytes, ...]:
    try:
        chunk: Final = await anext(chunks)
    except StopAsyncIteration:
        return ()
    return (chunk, *(await _collect_bytes(chunks)))


def test_sync_line_framing_handles_byte_boundaries_crlf_utf8_and_final_line() -> None:
    stream: Final = _FakeStream((b"first\r", b"\nsecond\nmultibyte: \xe2", b"\x82\xac\rlast"))
    response: Final = _adapter(stream)

    assert tuple(response.iter_lines()) == ("first", "second", "multibyte: €", "last")
    assert response.headers["x-upstream"] == "test"


@pytest.mark.asyncio
async def test_async_byte_iteration_and_close() -> None:
    stream: Final = _FakeStream((b"one", b"two"))
    response: Final = _adapter(stream)

    assert await _collect_bytes(response.aiter_bytes()) == (b"one", b"two")
    await response.aclose()
    assert stream.closed is True


def test_close_before_consumption_stops_iteration() -> None:
    stream: Final = _FakeStream((b"unused",))
    response: Final = _adapter(stream)

    response.close()
    assert tuple(response.iter_bytes()) == ()


@pytest.mark.asyncio
async def test_rejects_mixed_consumption_modes() -> None:
    response: Final = _adapter(_FakeStream((b"one",)))
    assert tuple(response.iter_bytes()) == (b"one",)

    with pytest.raises(RuntimeError, match="cannot mix"):
        await anext(response.aiter_bytes())


def test_maps_native_iteration_errors_without_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge, "get_native_bridge", lambda: _FakeNative())
    response: Final = _adapter(_FakeStream((b"one",), _FakeUpstreamError(0, "connection reset")))

    with pytest.raises(httpx.ReadError, match="connection reset"):
        tuple(response.iter_bytes())


def test_declined_capability_never_calls_native_factory() -> None:
    factory: Final = _RecordingFactory((b"unused",))
    bridge.set_rust_http_stream(open_stream=factory, capability=lambda provider: False)

    assert (
        bridge.open_stream(
            provider="anthropic",
            litellm_params=MappingProxyType({"rust": True}),
            custom_client=False,
            method="POST",
            url="https://example.test/v1/messages",
            headers=MappingProxyType({}),
            body=b"{}",
            timeout=1.0,
        )
        is None
    )
    assert factory.calls == ()


def _anthropic_sse() -> tuple[bytes, ...]:
    events: Final = (
        '{"type":"message_start","message":{"id":"msg_native","type":"message","role":"assistant",'
        '"model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":1,"output_tokens":0}}}',
        '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"native"}}',
        '{"type":"content_block_stop","index":0}',
        '{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}',
        '{"type":"message_stop"}',
    )
    payload: Final = "".join(f"data: {event}\n\n" for event in events).encode()
    return payload[:17], payload[17:63], payload[63:]


class _Delta(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    delta: _Delta


class _Chunk(BaseModel):
    choices: tuple[_Choice, ...]


def _chunk_text(chunk: object) -> str:
    assert isinstance(chunk, BaseModel)
    parsed: Final = _Chunk.model_validate(chunk.model_dump())
    return "".join(choice.delta.content or "" for choice in parsed.choices)


def _completion_text(chunks: Iterator[object]) -> str:
    return "".join(_chunk_text(chunk) for chunk in chunks)


async def _async_completion_text(chunks: AsyncIterator[object]) -> str:
    try:
        chunk: Final = await anext(chunks)
    except StopAsyncIteration:
        return ""
    return _chunk_text(chunk) + await _async_completion_text(chunks)


def test_public_sync_stream_uses_injected_native_transport() -> None:
    factory: Final = _RecordingFactory(_anthropic_sse())
    bridge.set_rust_http_stream(open_stream=factory, capability=lambda provider: provider == "anthropic")
    messages: Final = [{"role": "user", "content": "hi"}]  # mutable-ok: public SDK requires a message list

    response: Final = litellm.completion(  # pyright: ignore[reportUnknownMemberType]  # public SDK has legacy coarse types
        model="anthropic/claude-sonnet-4-6",
        messages=messages,
        api_key="test",
        api_base="https://example.test",
        max_tokens=8,
        stream=True,
        rust=True,
    )

    assert isinstance(response, CustomStreamWrapper)
    assert isinstance(response, Iterator)
    assert _completion_text(response) == "native"
    assert len(factory.calls) == 1
    assert (factory.calls[0].method, factory.calls[0].url) == ("POST", "https://example.test/v1/messages")
    assert json.loads(factory.calls[0].body)["stream"] is True


@pytest.mark.asyncio
async def test_public_async_stream_uses_injected_native_transport() -> None:
    factory: Final = _RecordingAsyncFactory(_anthropic_sse())
    bridge.set_rust_http_stream(aopen_stream=factory, capability=lambda provider: provider == "anthropic")
    messages: Final = [{"role": "user", "content": "hi"}]  # mutable-ok: public SDK requires a message list

    response: Final = await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]  # public SDK has legacy coarse types
        model="anthropic/claude-sonnet-4-6",
        messages=messages,
        api_key="test",
        api_base="https://example.test",
        max_tokens=8,
        stream=True,
        rust=True,
    )
    assert isinstance(response, CustomStreamWrapper)
    assert isinstance(response, AsyncIterator)
    assert await _async_completion_text(response) == "native"
    assert len(factory.calls) == 1
