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

use crate::error::{ProviderCallError, ProviderCallResult};

use handler::execute_chat_completions_provider_call;
use prepare::prepare_chat_completions_call;
use types::{ChatCompletionsRequest, ChatCompletionsResponse};

pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
) -> ProviderCallResult<ChatCompletionsResponse> {
    let prepared = prepare_chat_completions_call(request).map_err(ProviderCallError::NotSent)?;
    execute_chat_completions_provider_call(prepared).await
}

#[cfg(test)]
mod tests;
