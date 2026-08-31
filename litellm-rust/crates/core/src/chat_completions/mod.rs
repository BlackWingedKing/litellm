//! The `/chat/completions` call, the Rust equivalent of Python's
//! `litellm.completion()`.
//!
//! [`chat_completions`] is the top-level entrypoint: give it a model, the
//! OpenAI-shaped message list, the provider-mapped optional params, and
//! credentials, and it resolves the provider, translates the conversation,
//! calls the provider, and returns a typed OpenAI-shaped response.

mod client;
mod common_utils;
pub mod conversation;
pub(crate) mod handler;
mod prepare;
pub mod response_utils;
pub mod transformation;
pub mod types;

use crate::error::Error;

use crate::streaming::{
    OpenedStream, StreamApi, StreamCapability, StreamTransport, supports_streaming,
};
use handler::execute_chat_completions_provider_call;
use prepare::prepare_chat_completions_call;
use types::{
    ChatCompletionsRequest, ChatCompletionsResponse, ChatCompletionsStreamRequest, ChatStreamEvent,
};

pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
    let prepared = prepare_chat_completions_call(request).map_err(Error::not_sent)?;
    execute_chat_completions_provider_call(prepared).await
}

pub async fn chat_completions_stream(
    request: ChatCompletionsStreamRequest,
) -> Result<OpenedStream<ChatStreamEvent>, Error> {
    let capability = StreamCapability {
        api: StreamApi::ChatCompletions,
        provider: request.context.provider,
        transport: StreamTransport::Http,
    };
    if !supports_streaming(capability) {
        return Err(Error::not_sent(crate::Error::Unsupported(
            "chat completions streaming",
        )));
    }
    Err(Error::not_sent(crate::Error::Unsupported(
        "chat completions streaming provider registration",
    )))
}

#[cfg(test)]
mod tests;
