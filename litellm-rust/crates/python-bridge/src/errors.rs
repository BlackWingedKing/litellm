use litellm_core::error::{CoreError, ProviderCallError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

pyo3::create_exception!(
    _native,
    RustBridgeDeclined,
    pyo3::exceptions::PyException,
    "The route declined before calling the provider, so the host may retry on its own path."
);

pyo3::create_exception!(
    _native,
    RustUpstreamError,
    pyo3::exceptions::PyException,
    "The request may have reached the provider, so retrying is unsafe. Args are (status, message); status is 0 when there was no HTTP response."
);

pub(crate) type BridgeResult<T> = Result<T, BridgeError>;

#[derive(Debug, Error)]
pub(crate) enum BridgeError {
    #[error(transparent)]
    Python(#[from] PyErr),
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    ProviderCall(#[from] ProviderCallError),
    #[error("bridge declined before entering core: {0}")]
    Declined(String),
}

impl BridgeError {
    pub(crate) fn declined(error: impl std::fmt::Display) -> Self {
        Self::Declined(error.to_string())
    }
}

fn core_error_to_pyerr(error: CoreError) -> PyErr {
    match error {
        CoreError::Auth(message) => PyValueError::new_err(message),
        CoreError::InvalidProvider(_)
        | CoreError::InvalidRequest(_)
        | CoreError::InvalidType { .. }
        | CoreError::MissingField(_) => PyValueError::new_err(error.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn upstream_error_to_pyerr(error: CoreError) -> PyErr {
    match error {
        CoreError::Http { status, body } => {
            RustUpstreamError::new_err((status, format!("{status}: {body}")))
        }
        error => RustUpstreamError::new_err((0u16, error.to_string())),
    }
}

impl From<BridgeError> for PyErr {
    fn from(error: BridgeError) -> Self {
        match error {
            BridgeError::Python(error) => error,
            BridgeError::Core(error) => core_error_to_pyerr(error),
            BridgeError::ProviderCall(ProviderCallError::NotSent(error)) => {
                RustBridgeDeclined::new_err(error.to_string())
            }
            BridgeError::ProviderCall(ProviderCallError::PossiblySent(error)) => {
                upstream_error_to_pyerr(error)
            }
            BridgeError::Declined(reason) => RustBridgeDeclined::new_err(reason),
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_lifecycle_errors_remain_typed_until_the_python_boundary() {
        let error = BridgeError::from(ProviderCallError::NotSent(CoreError::Unsupported(
            "streaming",
        )));

        assert!(matches!(
            error,
            BridgeError::ProviderCall(ProviderCallError::NotSent(CoreError::Unsupported(
                "streaming"
            )))
        ));
    }

    #[test]
    fn provider_lifecycle_errors_map_to_distinct_python_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let declined = PyErr::from(BridgeError::from(ProviderCallError::NotSent(
                CoreError::Unsupported("streaming"),
            )));
            let possibly_sent = PyErr::from(BridgeError::from(ProviderCallError::PossiblySent(
                CoreError::InvalidResponse("bad body".to_string()),
            )));

            assert!(declined.is_instance_of::<RustBridgeDeclined>(py));
            assert!(possibly_sent.is_instance_of::<RustUpstreamError>(py));
            assert_eq!(
                declined
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(String,)>()
                    .unwrap(),
                ("unsupported by the rust path: streaming".to_string(),)
            );
            assert_eq!(
                possibly_sent
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(u16, String)>()
                    .unwrap(),
                (0, "invalid response: bad body".to_string())
            );
        });
    }

    #[test]
    fn upstream_http_error_preserves_status_and_message() {
        Python::initialize();
        Python::attach(|py| {
            let error = PyErr::from(BridgeError::from(ProviderCallError::PossiblySent(
                CoreError::Http {
                    status: 429,
                    body: "rate limited".to_string(),
                },
            )));

            assert_eq!(
                error
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(u16, String)>()
                    .unwrap(),
                (429, "429: rate limited".to_string())
            );
        });
    }
}
