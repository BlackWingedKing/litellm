use litellm_core::Error;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const MAX_BOUNDARY_MESSAGE_CHARS: usize = 512;

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

pub(crate) fn declined(error: impl std::fmt::Display) -> PyErr {
    RustBridgeDeclined::new_err(error.to_string())
}

fn unclassified_to_pyerr(error: Error) -> PyErr {
    match error {
        Error::Auth(message) => PyValueError::new_err(message),
        Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_) => PyValueError::new_err(error.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn upstream_to_pyerr(error: Error) -> PyErr {
    let (status, message) = match error.root() {
        Error::Http { status, body } => (*status, format!("{status}: {body}")),
        root => (0, root.to_string()),
    };
    RustUpstreamError::new_err((status, boundary_message(&message)))
}

fn boundary_message(message: &str) -> String {
    let sanitized: String = message
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .take(MAX_BOUNDARY_MESSAGE_CHARS + 1)
        .collect();
    if sanitized.chars().count() <= MAX_BOUNDARY_MESSAGE_CHARS {
        return sanitized;
    }
    let truncated: String = sanitized.chars().take(MAX_BOUNDARY_MESSAGE_CHARS).collect();
    format!("{truncated}... (truncated)")
}

pub(crate) fn to_pyerr(error: Error) -> PyErr {
    match error {
        Error::NotSent(error) => RustBridgeDeclined::new_err(error.to_string()),
        Error::PossiblySent(error) => upstream_to_pyerr(*error),
        error => unclassified_to_pyerr(error),
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
    fn lifecycle_errors_map_to_distinct_python_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let declined = to_pyerr(Error::not_sent(Error::Unsupported("streaming")));
            let possibly_sent = to_pyerr(Error::possibly_sent(Error::InvalidResponse(
                "bad body".to_string(),
            )));

            assert!(declined.is_instance_of::<RustBridgeDeclined>(py));
            assert!(possibly_sent.is_instance_of::<RustUpstreamError>(py));
        });
    }

    #[test]
    fn upstream_http_error_preserves_status_and_message() {
        Python::initialize();
        Python::attach(|py| {
            let error = to_pyerr(Error::possibly_sent(Error::Http {
                status: 429,
                body: "rate limited".to_string(),
            }));

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

    #[test]
    fn upstream_messages_are_sanitized_and_bounded() {
        let message = format!("bad\nresponse\0{}", "x".repeat(600));
        let sanitized = boundary_message(&message);

        assert!(!sanitized.contains('\n'));
        assert!(!sanitized.contains('\0'));
        assert!(sanitized.ends_with("... (truncated)"));
        assert_eq!(
            sanitized
                .strip_suffix("... (truncated)")
                .unwrap()
                .chars()
                .count(),
            MAX_BOUNDARY_MESSAGE_CHARS
        );
    }
}
