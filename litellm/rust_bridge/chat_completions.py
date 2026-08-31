"""Thin Python wrapper for the native Rust chat completions bridge.

The Rust core owns the conversation translation, the provider call, and the
response normalization for the subset of `/chat/completions` requests it
accepts. This module only marshals inputs and hands the normalized result to
LiteLLM's existing `ModelResponse` builder. The caller supplies the Python
fallback, which runs only when the native route guarantees it sent nothing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import APIError
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.bedrock.request_metadata import bedrock_request_metadata_is_owned
from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

# Providers whose `/chat/completions` deployments the Rust core can serve. A
# provider outside this set never reaches the bridge.
RUST_CHAT_COMPLETIONS_PROVIDERS: Final = frozenset({"anthropic", "bedrock"})

# `litellm_params` values are `object`, so validate the one this module reads
# rather than narrowing an unparameterized `Mapping` and typing the result Any.
_LITELLM_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])

RUST_RESPONSE_HEADER: Final = "x-litellm-rust"

_TRUTHY_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})


class RustChatCompletions(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Mapping[str, object]:
        raise NotImplementedError


class RustAchatCompletions(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Awaitable[Mapping[str, object]]:
        raise NotImplementedError


class ResponseObserver(Protocol):
    """Invoked with the payload the core returned, on success only.

    Lets the caller emit its own `post_call` on whichever path served the
    request. Both entry points call it, so the synchronous and asynchronous
    paths cannot drift apart the way the pre_call suppression once did.
    """

    def __call__(self, rust_response: Mapping[str, object], /) -> None:
        raise NotImplementedError


def response_logger(
    *,
    logging_obj: LiteLLMLoggingObj,
    messages: Sequence[object],
    api_key: str,
    additional_args: Mapping[str, object],
) -> ResponseObserver:
    """A `ResponseObserver` that emits the caller's `post_call` for a Rust-served
    request.

    The core owns the provider call, so the Python transform that normally
    raises this event never runs; without it every `post_call` callback goes
    silent on a Rust-served request and `original_response` stays unset. The
    payload is the core's normalized response rather than the provider's wire
    body, which is the closest thing that crosses the bridge.
    """

    def log(rust_response: Mapping[str, object], /) -> None:
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=json.dumps(rust_response),
            additional_args=additional_args,
        )

    return log


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustChatCompletionsState:
    chat_completions: RustChatCompletions | None = None
    achat_completions: RustAchatCompletions | None = None


_STATE: Final[_RustChatCompletionsState] = _RustChatCompletionsState()


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | None | _Unset = _UNSET,
    achat_completions: RustAchatCompletions | None | _Unset = _UNSET,
) -> None:
    """Inject the native callables, so tests can supply a double instead of
    patching module attributes."""
    if not isinstance(chat_completions, _Unset):
        _STATE.chat_completions = chat_completions
    if not isinstance(achat_completions, _Unset):
        _STATE.achat_completions = achat_completions


def load_rust_chat_completions() -> RustChatCompletions | None:
    if _STATE.chat_completions is not None:
        return _STATE.chat_completions
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    loaded: RustChatCompletions | None = getattr(native_bridge, "chat_completions", None)
    return loaded


def load_rust_achat_completions() -> RustAchatCompletions | None:
    if _STATE.achat_completions is not None:
        return _STATE.achat_completions
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    loaded: RustAchatCompletions | None = getattr(native_bridge, "achat_completions", None)
    return loaded


def _env_enables_rust() -> bool:
    return os.getenv("LITELLM_RUST", "").strip().lower() in _TRUTHY_ENV_VALUES


def _anthropic_user_id_reaches_the_body(litellm_params: Mapping[str, object] | None) -> bool:
    metadata: Final = litellm_params.get("metadata") if litellm_params is not None else None
    try:
        entries: Final = _LITELLM_METADATA_ADAPTER.validate_python(metadata)
    except ValidationError:
        return False
    return entries.get("user_id") is not None


def _litellm_metadata_reaches_the_provider(
    custom_llm_provider: str | None, litellm_params: Mapping[str, object] | None
) -> bool:
    """Whether the Python transform would promote proxy-owned attribution into the
    provider request, below this gate and inside the function the Rust route replaces.

    `AnthropicConfig.transform_request` promotes a valid `metadata["user_id"]`
    into the Messages body, so the core never sees the key and would send the
    request to Anthropic with the abuse-detection attribution missing.

    `AmazonConverseConfig` resolves proxy-owned `requestMetadata` onto the
    Converse body whenever the operator armed `bedrock_request_metadata_fields`.
    Owning that field also means evicting a caller-supplied one, which the core
    cannot do either, so ownership alone is the condition rather than whether
    anything resolved.

    Deliberately a superset of Python's condition in both cases: declining a
    request Python would not have attributed anyway costs only the Rust path,
    while missing one loses the attribution silently.
    """
    match custom_llm_provider:
        case "anthropic":
            return _anthropic_user_id_reaches_the_body(litellm_params)
        case "bedrock":
            return bedrock_request_metadata_is_owned()
        case _:
            return False


def should_attempt_rust_chat_completions(
    *,
    custom_llm_provider: str | None,
    litellm_params: Mapping[str, object] | None,
    stream: object,
    asynchronous: bool,
) -> bool:
    if custom_llm_provider not in RUST_CHAT_COMPLETIONS_PROVIDERS:
        return False
    if stream:
        return False
    opted_in: Final = litellm_params is not None and litellm_params.get("rust") is True
    if not opted_in and not _env_enables_rust():
        return False
    if _litellm_metadata_reaches_the_provider(custom_llm_provider, litellm_params):
        verbose_logger.debug("Rust chat completions skipped because Python owns provider metadata")
        return False
    native_call: Final = load_rust_achat_completions() if asynchronous else load_rust_chat_completions()
    return native_call is not None


def _rust_bridge_exceptions() -> tuple[type[BaseException], type[BaseException]] | None:
    """`(declined, upstream_failed)` from the native module, or None when absent."""
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    declined: Final = getattr(native_bridge, "RustBridgeDeclined", None)
    upstream: Final = getattr(native_bridge, "RustUpstreamError", None)
    if declined is None or upstream is None:
        return None
    return declined, upstream


def _raise_unless_declined(
    rust_error: BaseException,
    *,
    model: str,
    custom_llm_provider: str | None,
) -> None:
    exceptions: Final = _rust_bridge_exceptions()
    if exceptions is None:
        raise rust_error
    declined, upstream_failed = exceptions
    if isinstance(rust_error, upstream_failed):
        args: Final = rust_error.args
        status: Final = args[0] if args else 0
        message: Final = args[1] if len(args) > 1 else ""
        raise APIError(
            status_code=int(status) or 500,
            message=f"litellm rust chat completions: {message}",
            llm_provider=custom_llm_provider or "",
            model=model,
        )
    if not isinstance(rust_error, declined):
        raise rust_error
    verbose_logger.debug(
        "Rust chat completions declined before calling the provider (%s); using the Python path",
        rust_error,
    )


def _build_model_response(
    rust_response: Mapping[str, object],
    model_response: ModelResponse,
) -> ModelResponse:
    built: Final = convert_to_model_response_object(
        response_object=dict(rust_response),  # mutable-ok: the converter takes a real dict and rewrites it
        model_response_object=model_response,
        hidden_params={"additional_headers": {RUST_RESPONSE_HEADER: "true"}},  # mutable-ok: rewritten by the converter
    )
    if not isinstance(built, ModelResponse):
        raise TypeError(f"expected a ModelResponse from the rust path, got {type(built).__name__}")
    return built


def chat_completions_or_fallback(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    on_response: ResponseObserver,
    python_fallback: Callable[[], object],
) -> object:
    rust_chat_completions: Final = load_rust_chat_completions()
    if rust_chat_completions is None:
        return python_fallback()
    try:
        rust_response: Final = rust_chat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    except Exception as rust_error:  # noqa: BLE001  # native exception classes define whether fallback is safe
        _raise_unless_declined(rust_error, model=model, custom_llm_provider=custom_llm_provider)
        return python_fallback()
    on_response(rust_response)
    return _build_model_response(rust_response, model_response)


async def achat_completions_or_fallback(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    on_response: ResponseObserver,
    python_fallback: Callable[[], Awaitable[object]],
) -> object:
    rust_achat_completions: Final = load_rust_achat_completions()
    if rust_achat_completions is None:
        return await python_fallback()
    try:
        rust_response: Final = await rust_achat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    except Exception as rust_error:  # noqa: BLE001  # native exception classes define whether fallback is safe
        _raise_unless_declined(rust_error, model=model, custom_llm_provider=custom_llm_provider)
        return await python_fallback()
    on_response(rust_response)
    return _build_model_response(rust_response, model_response)
