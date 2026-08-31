from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, Generic, NoReturn, TypeAlias, TypeVar, cast

from litellm.exceptions import APIError
from litellm.rust_bridge.loader import get_native_bridge

NativeT = TypeVar("NativeT")
ResultT = TypeVar("ResultT")
BindingT = TypeVar("BindingT")

CORE_ENGINE_HIDDEN_PARAM: Final = "core_engine"
CORE_ENGINE_HEADER: Final = "x-litellm-core"
LEGACY_RUST_HEADER: Final = "x-litellm-rust"


class Unset:
    pass


UNSET: Final = Unset()


class FallbackMode(Enum):
    PYTHON = "python"
    RUST_REQUIRED = "rust_required"


class CoreEngine(str, Enum):
    PYTHON = "python"
    RUST = "rust"


@dataclass(frozen=True, slots=True)
class ExecutionResult(Generic[ResultT]):
    value: ResultT
    source: CoreEngine


@dataclass(frozen=True, slots=True)
class RustHandled(Generic[ResultT]):
    value: ResultT


@dataclass(frozen=True, slots=True)
class RustDeclined:
    pass


@dataclass(frozen=True, slots=True)
class RustUnavailable:
    pass


RustAttempt: TypeAlias = RustHandled[ResultT] | RustDeclined | RustUnavailable


def execution_headers(source: CoreEngine) -> dict[str, str]:
    if source is CoreEngine.RUST:
        return {CORE_ENGINE_HEADER: source.value, LEGACY_RUST_HEADER: "true"}
    return {CORE_ENGINE_HEADER: source.value}


def execution_additional_headers(
    additional_headers: Mapping[str, object] | None,
    source: CoreEngine,
) -> dict[str, str]:
    existing: Final = additional_headers or {}
    reserved: Final = frozenset({CORE_ENGINE_HEADER, LEGACY_RUST_HEADER})
    preserved: Final = {str(name): str(value) for name, value in existing.items() if str(name).lower() not in reserved}
    return {**preserved, **execution_headers(source)}


def execution_hidden_params(
    hidden_params: Mapping[str, object] | None,
    source: CoreEngine,
) -> dict[str, object]:
    existing: Final = hidden_params or {}
    raw_headers: Final = existing.get("additional_headers")
    additional_headers: Final[Mapping[str, object]] = (
        cast(Mapping[str, object], raw_headers) if isinstance(raw_headers, Mapping) else {}
    )
    return {
        **existing,
        CORE_ENGINE_HIDDEN_PARAM: source.value,
        "additional_headers": execution_additional_headers(additional_headers, source),
    }


@dataclass(frozen=True, slots=True)
class BridgeContext:
    route: str
    provider: str
    model: str


class NativeBinding(Generic[BindingT]):
    def __init__(self, attribute: str) -> None:
        self._attribute: Final = attribute
        self._override: BindingT | Unset = UNSET

    def load(self) -> BindingT | None:
        if not isinstance(self._override, Unset):
            return self._override
        native: Final = get_native_bridge()
        return (
            None
            if native is None
            else cast(  # cast-ok: PyO3 callables are checked by each binding protocol
                BindingT | None, getattr(native, self._attribute, None)
            )
        )

    def set(self, value: BindingT | None) -> None:
        self._override = UNSET if value is None else value

    def update(self, value: BindingT | None | Unset) -> None:
        if not isinstance(value, Unset):
            self.set(value)


def invoke(
    *,
    native_call: Callable[[], NativeT] | None,
    fallback: Callable[[], ResultT],
    adapt: Callable[[NativeT], ResultT],
    mode: FallbackMode,
    context: BridgeContext,
) -> ExecutionResult[ResultT]:
    attempt_result: Final = attempt(native_call=native_call, adapt=adapt, context=context)
    if isinstance(attempt_result, RustHandled):
        return ExecutionResult(value=attempt_result.value, source=CoreEngine.RUST)
    if mode is FallbackMode.PYTHON:
        return ExecutionResult(value=fallback(), source=CoreEngine.PYTHON)
    _raise_required(attempt_result, context)


async def ainvoke(
    *,
    native_call: Callable[[], Awaitable[NativeT]] | None,
    fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[NativeT], ResultT],
    mode: FallbackMode,
    context: BridgeContext,
) -> ExecutionResult[ResultT]:
    attempt_result: Final = await aattempt(native_call=native_call, adapt=adapt, context=context)
    if isinstance(attempt_result, RustHandled):
        return ExecutionResult(value=attempt_result.value, source=CoreEngine.RUST)
    if mode is FallbackMode.PYTHON:
        return ExecutionResult(value=await fallback(), source=CoreEngine.PYTHON)
    _raise_required(attempt_result, context)


def attempt(
    *,
    native_call: Callable[[], NativeT] | None,
    adapt: Callable[[NativeT], ResultT],
    context: BridgeContext,
) -> RustAttempt[ResultT]:
    if native_call is None:
        return RustUnavailable()
    exceptions: Final = _native_exceptions()
    if exceptions is None:
        return RustHandled(adapt(native_call()))
    declined, upstream = exceptions
    try:
        native_result: Final = native_call()
    except declined:
        return RustDeclined()
    except upstream as error:
        _raise_upstream(error, context)
    return RustHandled(adapt(native_result))


async def aattempt(
    *,
    native_call: Callable[[], Awaitable[NativeT]] | None,
    adapt: Callable[[NativeT], ResultT],
    context: BridgeContext,
) -> RustAttempt[ResultT]:
    if native_call is None:
        return RustUnavailable()
    exceptions: Final = _native_exceptions()
    if exceptions is None:
        return RustHandled(adapt(await native_call()))
    declined, upstream = exceptions
    try:
        native_result: Final = await native_call()
    except declined:
        return RustDeclined()
    except upstream as error:
        _raise_upstream(error, context)
    return RustHandled(adapt(native_result))


def _native_exceptions() -> tuple[type[BaseException], type[BaseException]] | None:
    native: Final = get_native_bridge()
    if native is None:
        return None
    declined: Final = getattr(native, "RustBridgeDeclined", None)
    upstream: Final = getattr(native, "RustUpstreamError", None)
    if not isinstance(declined, type) or not isinstance(upstream, type):
        return None
    return declined, upstream


def _raise_required(
    attempt_result: RustDeclined | RustUnavailable,
    context: BridgeContext,
) -> NoReturn:
    raise RuntimeError(f"Rust {context.route} bridge {_required_reason(attempt_result)}")


def _required_reason(attempt_result: RustDeclined | RustUnavailable) -> str:
    match attempt_result:
        case RustUnavailable():
            return "is unavailable"
        case RustDeclined():
            return "declined the request"


def _raise_upstream(error: BaseException, context: BridgeContext) -> NoReturn:
    args: Final = cast(tuple[object, ...], error.args)
    status_value: Final = args[0] if args else 0
    message_value: Final = args[1] if len(args) > 1 else str(error)
    status: Final = status_value if isinstance(status_value, int) else 0
    message: Final = message_value if isinstance(message_value, str) else str(message_value)
    api_error: Final = APIError(
        status_code=status or 500,
        message=f"litellm rust {context.route}: {message}",
        llm_provider=context.provider,
        model=context.model,
    )
    api_error.headers = execution_headers(  # pyright: ignore[reportAttributeAccessIssue]  # proxy reads exception headers
        CoreEngine.RUST
    )
    raise api_error from error


def call(operation: Callable[[], ResultT], context: BridgeContext) -> ResultT:
    exceptions: Final = _native_exceptions()
    if exceptions is None:
        return operation()
    upstream: Final = exceptions[1]
    try:
        return operation()
    except upstream as error:
        _raise_upstream(error, context)


async def acall(operation: Callable[[], Awaitable[ResultT]], context: BridgeContext) -> ResultT:
    exceptions: Final = _native_exceptions()
    if exceptions is None:
        return await operation()
    upstream: Final = exceptions[1]
    try:
        return await operation()
    except upstream as error:
        _raise_upstream(error, context)


def identity(value: ResultT) -> ResultT:
    return value


async def async_none() -> None:
    return None
