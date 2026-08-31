from __future__ import annotations

import codecs
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, NoReturn, Protocol

import httpx

from litellm.rust_bridge.loader import get_native_bridge


class RustHttpResponseStream(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Sequence[tuple[str, str]]: ...

    def next_bytes(self) -> bytes | None: ...

    async def anext_bytes(self) -> bytes | None: ...

    def close(self) -> None: ...

    async def aclose(self) -> None: ...


class RustOpenHttpStream(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        open_timeout_seconds: float | None,
        read_idle_timeout_seconds: float | None,
    ) -> RustHttpResponseStream: ...


class RustAopenHttpStream(Protocol):
    async def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        open_timeout_seconds: float | None,
        read_idle_timeout_seconds: float | None,
    ) -> RustHttpResponseStream: ...


class RustStreamCapability(Protocol):
    def __call__(self, provider: str) -> bool: ...


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustHttpStreamState:
    open_stream: RustOpenHttpStream | None = None
    aopen_stream: RustAopenHttpStream | None = None
    capability: RustStreamCapability | None = None


_STATE: Final = _RustHttpStreamState()


def set_rust_http_stream(
    *,
    open_stream: RustOpenHttpStream | None | _Unset = _UNSET,
    aopen_stream: RustAopenHttpStream | None | _Unset = _UNSET,
    capability: RustStreamCapability | None | _Unset = _UNSET,
) -> None:
    if not isinstance(open_stream, _Unset):
        _STATE.open_stream = open_stream
    if not isinstance(aopen_stream, _Unset):
        _STATE.aopen_stream = aopen_stream
    if not isinstance(capability, _Unset):
        _STATE.capability = capability


def _load_open_stream() -> RustOpenHttpStream | None:
    if _STATE.open_stream is not None:
        return _STATE.open_stream
    native: Final = get_native_bridge()
    return None if native is None else getattr(native, "open_http_stream", None)


def _load_aopen_stream() -> RustAopenHttpStream | None:
    if _STATE.aopen_stream is not None:
        return _STATE.aopen_stream
    native: Final = get_native_bridge()
    return None if native is None else getattr(native, "aopen_http_stream", None)


def _enabled(provider: str, litellm_params: Mapping[str, object] | None, custom_client: bool) -> bool:
    if custom_client or litellm_params is None or litellm_params.get("rust") is not True:
        return False
    capability: Final = _STATE.capability
    return capability is not None and capability(provider)


def _timeouts(timeout: float | httpx.Timeout | None) -> tuple[float | None, float | None]:
    if timeout is None:
        return None, None
    if isinstance(timeout, httpx.Timeout):
        open_timeout: Final = timeout.connect if timeout.connect is not None else timeout.read
        return open_timeout, timeout.read
    seconds: Final = float(timeout)
    return seconds, seconds


class _LineFramer:
    def __init__(self) -> None:
        self._decoder: Final = codecs.getincrementaldecoder("utf-8")()
        self._pending = ""

    def feed(self, chunk: bytes) -> tuple[str, ...]:
        self._pending += self._decoder.decode(chunk)
        return self._drain(final=False)

    def finish(self) -> tuple[str, ...]:
        self._pending += self._decoder.decode(b"", final=True)
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> tuple[str, ...]:
        fragments: Final = tuple(self._pending.splitlines(keepends=True))
        complete: Final = fragments if final or self._pending.endswith("\n") else fragments[:-1]
        self._pending = "" if len(complete) == len(fragments) else fragments[-1]
        return tuple(_strip_line_ending(line) for line in complete)


def _strip_line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith(("\r", "\n")):
        return line[:-1]
    return line


def _native_exceptions() -> tuple[type[BaseException], type[BaseException]] | None:
    native: Final = get_native_bridge()
    if native is None:
        return None
    declined: Final = getattr(native, "RustBridgeDeclined", None)
    upstream: Final = getattr(native, "RustUpstreamError", None)
    if not isinstance(declined, type) or not isinstance(upstream, type):
        return None
    return declined, upstream


def _raise_stream_error(error: Exception, request: httpx.Request) -> NoReturn:
    exceptions: Final = _native_exceptions()
    if exceptions is None or not isinstance(error, exceptions[1]):
        raise error
    status_arg: Final = error.args[0] if error.args else 0
    message_arg: Final = error.args[1] if len(error.args) > 1 else error
    status: Final = status_arg if isinstance(status_arg, int) else 0
    message: Final = message_arg if isinstance(message_arg, str) else str(message_arg)
    if status:
        response: Final = httpx.Response(status, request=request, text=message)
        raise httpx.HTTPStatusError(message, request=request, response=response) from error
    raise httpx.ReadError(message, request=request) from error


class HttpResponseStreamAdapter:
    def __init__(self, stream: RustHttpResponseStream, request: httpx.Request) -> None:
        self._stream: Final = stream
        self._request: Final = request
        self.headers: Final = httpx.Headers((*stream.headers, ("x-litellm-rust", "true")))
        self.status_code: Final = stream.status_code
        self._mode: Literal["sync", "async"] | None = None

    def _claim(self, mode: Literal["sync", "async"]) -> None:
        if self._mode is None:
            self._mode = mode
            return
        if self._mode != mode:
            raise RuntimeError("native stream cannot mix synchronous and asynchronous consumption")

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        self._claim("sync")
        yield from iter(self._next_bytes, None)

    def _next_bytes(self) -> bytes | None:
        try:
            return self._stream.next_bytes()
        except Exception as error:  # noqa: BLE001  # native exception classes define public mapping
            _raise_stream_error(error, self._request)

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        self._claim("async")
        async for chunk in _AsyncNativeByteIterator(self._stream, self._request):
            yield chunk

    def iter_lines(self) -> Iterator[str]:
        framer: Final = _LineFramer()
        for chunk in self.iter_bytes():
            yield from framer.feed(chunk)
        yield from framer.finish()

    async def aiter_lines(self) -> AsyncIterator[str]:
        framer: Final = _LineFramer()
        async for chunk in self.aiter_bytes():
            for line in framer.feed(chunk):
                yield line
        for line in framer.finish():
            yield line

    def close(self) -> None:
        self._stream.close()

    async def aclose(self) -> None:
        await self._stream.aclose()


class _AsyncNativeByteIterator:
    def __init__(self, stream: RustHttpResponseStream, request: httpx.Request) -> None:
        self._stream: Final = stream
        self._request: Final = request

    def __aiter__(self) -> _AsyncNativeByteIterator:
        return self

    async def __anext__(self) -> bytes:
        try:
            chunk: Final = await self._stream.anext_bytes()
        except Exception as error:  # noqa: BLE001  # native exception classes define public mapping
            _raise_stream_error(error, self._request)
        if chunk is None:
            raise StopAsyncIteration
        return chunk


def open_stream(
    *,
    provider: str,
    litellm_params: Mapping[str, object] | None,
    custom_client: bool,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float | httpx.Timeout | None,
) -> HttpResponseStreamAdapter | None:
    if not _enabled(provider, litellm_params, custom_client):
        return None
    opener: Final = _load_open_stream()
    if opener is None:
        return None
    open_timeout, read_timeout = _timeouts(timeout)
    request: Final = httpx.Request(method, url, headers=headers, content=body)
    try:
        native_headers: Final = dict(headers)  # mutable-ok: PyO3's mapping boundary requires a concrete dict
        stream: Final = opener(method, url, native_headers, body, open_timeout, read_timeout)
    except Exception as error:  # noqa: BLE001  # native exception classes determine whether fallback is safe
        exceptions: Final = _native_exceptions()
        if exceptions is not None and isinstance(error, exceptions[0]):
            return None
        _raise_stream_error(error, request)
    return HttpResponseStreamAdapter(stream, request)


async def aopen_stream(
    *,
    provider: str,
    litellm_params: Mapping[str, object] | None,
    custom_client: bool,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float | httpx.Timeout | None,
) -> HttpResponseStreamAdapter | None:
    if not _enabled(provider, litellm_params, custom_client):
        return None
    opener: Final = _load_aopen_stream()
    if opener is None:
        return None
    open_timeout, read_timeout = _timeouts(timeout)
    request: Final = httpx.Request(method, url, headers=headers, content=body)
    try:
        native_headers: Final = dict(headers)  # mutable-ok: PyO3's mapping boundary requires a concrete dict
        stream: Final = await opener(method, url, native_headers, body, open_timeout, read_timeout)
    except Exception as error:  # noqa: BLE001  # native exception classes determine whether fallback is safe
        exceptions: Final = _native_exceptions()
        if exceptions is not None and isinstance(error, exceptions[0]):
            return None
        _raise_stream_error(error, request)
    return HttpResponseStreamAdapter(stream, request)
