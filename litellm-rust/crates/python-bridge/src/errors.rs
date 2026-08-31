use litellm_core::error::{CoreError, ProviderCallError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

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
    "The provider call was already issued and failed. Args are (status, message); status is 0 when there was no HTTP response."
);

pub(crate) type BridgeResult<T> = Result<T, BridgeError>;

#[derive(Debug)]
pub(crate) enum BridgeError {
    Python(PyErr),
    Core(CoreError),
    Declined(String),
    PossiblySent(CoreError),
}

impl BridgeError {
    pub(crate) fn declined(error: impl std::fmt::Display) -> Self {
        Self::Declined(error.to_string())
    }

    pub(crate) fn into_pyerr(self) -> PyErr {
        match self {
            Self::Python(error) => error,
            Self::Core(error) => core_error_to_pyerr(error),
            Self::Declined(message) => RustBridgeDeclined::new_err(message),
            Self::PossiblySent(CoreError::Http { status, body }) => {
                RustUpstreamError::new_err((status, format!("{status}: {body}")))
            }
            Self::PossiblySent(error) => RustUpstreamError::new_err((0u16, error.to_string())),
        }
    }
}

impl From<PyErr> for BridgeError {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

impl From<CoreError> for BridgeError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl From<ProviderCallError> for BridgeError {
    fn from(error: ProviderCallError) -> Self {
        match error {
            ProviderCallError::NotSent(error) => Self::declined(error),
            ProviderCallError::PossiblySent(error) => Self::PossiblySent(error),
        }
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_lifecycle_errors_map_to_distinct_python_types() {
        Python::initialize();
        Python::attach(|py| {
            let declined = BridgeError::from(ProviderCallError::NotSent(CoreError::Unsupported(
                "streaming",
            )))
            .into_pyerr();
            let possibly_sent = BridgeError::from(ProviderCallError::PossiblySent(
                CoreError::InvalidResponse("bad body".to_string()),
            ))
            .into_pyerr();

            assert!(declined.is_instance_of::<RustBridgeDeclined>(py));
            assert!(possibly_sent.is_instance_of::<RustUpstreamError>(py));
        });
    }
}
