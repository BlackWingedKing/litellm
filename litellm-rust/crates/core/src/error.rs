use thiserror::Error as ThisError;

#[derive(Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error("provider request was not sent: {0}")]
    NotSent(#[source] Box<Error>),
    #[error("provider request may have been sent: {0}")]
    PossiblySent(#[source] Box<Error>),
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Auth(String),
    #[error("upstream request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("upstream network error: {0}")]
    Network(String),
    /// The provider was never reached: DNS, TCP, TLS or proxy setup failed
    /// before any byte of the request went out. Nothing was billed, so a host
    /// that keeps a reference implementation can serve the request itself.
    /// A timeout is deliberately not this, since the provider may have received
    /// and answered the request already.
    #[error("could not reach the provider: {0}")]
    Connect(String),
    #[error("routing error: {0}")]
    Routing(String),
    /// The request is outside the surface this route covers in Rust. Hosts that
    /// keep a reference implementation treat this as "fall back", not "fail".
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
}

impl Error {
    pub fn not_sent(error: Error) -> Self {
        Self::NotSent(Box::new(error))
    }

    pub fn possibly_sent(error: Error) -> Self {
        Self::PossiblySent(Box::new(error))
    }

    pub fn from_send_error(error: reqwest::Error) -> Self {
        if error.is_connect() {
            return Self::not_sent(Self::Connect(error.to_string()));
        }
        Self::possibly_sent(Self::Network(error.to_string()))
    }

    pub fn root(&self) -> &Error {
        match self {
            Self::NotSent(error) | Self::PossiblySent(error) => error.root(),
            error => error,
        }
    }
}

pub fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}
