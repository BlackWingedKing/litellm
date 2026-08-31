from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest

from litellm.exceptions import APIError
from litellm.rust_bridge import messages as messages_bridge

try:
    from litellm.rust_bridge import _native as native
except ImportError:
    if os.getenv("LITELLM_REQUIRE_NATIVE_BRIDGE") == "1":
        raise
    pytest.skip("the native bridge has not been built", allow_module_level=True)

REQUEST_BODY: Final = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}
SUCCESS_BODY: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "hi"}],
    "model": "claude-sonnet-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 2},
}


@contextmanager
def upstream(status: int, body: dict[str, object]) -> Iterator[str]:
    encoded = json.dumps(body).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.path == "/v1/messages"
            length = int(self.headers.get("content-length", "0"))
            self.rfile.read(length)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format_string: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def native_kwargs(api_base: str) -> dict[str, object]:
    return {
        "model": "claude-sonnet-4-5",
        "body": REQUEST_BODY,
        "api_key": "sk-test",
        "api_base": api_base,
        "custom_llm_provider": "anthropic",
        "extra_headers": None,
        "timeout_seconds": 5.0,
    }


@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    [
        ("chat_completions", {"model": "model", "messages": {}}),
        ("achat_completions", {"model": "model", "messages": {}}),
        ("messages", {"model": "model", "body": []}),
        ("amessages", {"model": "model", "body": []}),
        ("ocr", {"model": "model", "document": object()}),
        ("aocr", {"model": "model", "document": object()}),
        ("transcription", {"model": "model", "audio": object()}),
        ("atranscription", {"model": "model", "audio": object()}),
        ("chat_completions_stream", {"request": object(), "provider": "anthropic"}),
        ("achat_completions_stream", {"request": object(), "provider": "anthropic"}),
        ("messages_stream", {"request": object(), "provider": "anthropic"}),
        ("amessages_stream", {"request": object(), "provider": "anthropic"}),
        ("responses_stream", {"request": object(), "provider": "openai"}),
        ("aresponses_stream", {"request": object(), "provider": "openai"}),
    ],
)
def test_every_native_route_preserves_marshalling_declines(
    function_name: str,
    kwargs: dict[str, object],
) -> None:
    function = getattr(native, function_name)
    with pytest.raises(native.RustBridgeDeclined):
        function(**kwargs)


def test_native_websocket_connect_preserves_decline() -> None:
    with pytest.raises(native.RustBridgeDeclined):
        native.ResponsesWebSocketSession.connect(provider="unsupported")


def test_native_upstream_error_preserves_status_and_message() -> None:
    with upstream(429, {"error": "rate limited"}) as api_base:
        with pytest.raises(native.RustUpstreamError) as exc_info:
            native.messages(**native_kwargs(api_base))

    assert exc_info.value.args[0] == 429
    assert "rate limited" in exc_info.value.args[1]


def test_python_runtime_never_falls_back_after_native_upstream_error() -> None:
    with upstream(503, {"error": "unavailable"}) as api_base:
        with pytest.raises(APIError) as exc_info:
            messages_bridge.messages(
                model="claude-sonnet-4-5",
                body=REQUEST_BODY,
                api_key="sk-test",
                api_base=api_base,
                custom_llm_provider="anthropic",
                extra_headers=None,
                timeout=5.0,
            )

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_sync_and_async_calls_cross_the_native_boundary_successfully() -> None:
    with upstream(200, SUCCESS_BODY) as api_base:
        sync_response = messages_bridge.messages(
            model="claude-sonnet-4-5",
            body=REQUEST_BODY,
            api_key="sk-test",
            api_base=api_base,
            custom_llm_provider="anthropic",
            extra_headers=None,
            timeout=5.0,
        )
        async_response = await messages_bridge.amessages(
            model="claude-sonnet-4-5",
            body=REQUEST_BODY,
            api_key="sk-test",
            api_base=api_base,
            custom_llm_provider="anthropic",
            extra_headers=None,
            timeout=5.0,
        )

    assert sync_response is not None
    assert async_response is not None
    assert sync_response["content"][0]["text"] == "hi"
    assert async_response["content"][0]["text"] == "hi"
