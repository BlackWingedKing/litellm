use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::chat_completions::chat_completions as run_chat_completions;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::{BridgeResult, Error};
use crate::marshal::{RouteOptions, object_or_empty, required_value};
use crate::routes::BridgeRoute;

struct ChatCompletionsCall {
    options: RouteOptions,
    messages: Value,
    optional_params: Map<String, Value>,
}

impl BridgeRoute<ChatCompletionsInputs> for ChatCompletionsCall {
    type Output = ChatCompletionsResponse;

    fn from_python(py: Python<'_>, inputs: ChatCompletionsInputs) -> BridgeResult<Self> {
        (|| -> PyResult<Self> {
            Ok(Self {
                options: RouteOptions::from_python(
                    py,
                    inputs.model,
                    inputs.api_key,
                    inputs.api_base,
                    inputs.custom_llm_provider,
                    inputs.extra_headers,
                    inputs.timeout_seconds,
                )?,
                messages: required_value(
                    py,
                    "messages",
                    inputs.messages,
                    Value::is_array,
                    "list",
                )?,
                optional_params: object_or_empty(py, "optional_params", inputs.optional_params)?,
            })
        })()
        .map_err(Error::declined)
    }

    async fn run(self) -> BridgeResult<ChatCompletionsResponse> {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = self.options;
        run_chat_completions(ChatCompletionsRequest {
            model: &model,
            messages: self.messages,
            optional_params: self.optional_params,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        })
        .await
        .map_err(Into::into)
    }
}

bridge_route! {
    sync = chat_completions,
    asynchronous = achat_completions,
    inputs = ChatCompletionsInputs,
    required = {
        model: String,
        messages: Py<PyAny>,
    },
    optional = {
        optional_params: Option<Py<PyAny>>,
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        extra_headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    },
    call = ChatCompletionsCall,
}
