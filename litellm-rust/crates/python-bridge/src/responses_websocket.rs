use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection as RustConnection;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{optional_timeout, string_headers};

#[pyclass]
struct ResponsesWebSocketConnection {
    inner: RustConnection,
}

#[pymethods]
impl ResponsesWebSocketConnection {
    #[classmethod]
    #[pyo3(signature = (url, headers=None, timeout_seconds=None))]
    fn connect<'py>(
        _cls: &Bound<'py, PyType>,
        py: Python<'py>,
        url: String,
        headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = string_headers(py, headers)?;
        let timeout = optional_timeout(timeout_seconds);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let inner = RustConnection::connect_url(&url, &headers, timeout)
                .await
                .map_err(core_error_to_pyerr)?;
            Python::attach(|py| Py::new(py, ResponsesWebSocketConnection { inner }))
        })
    }

    fn send_text<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.send_text(text).await.map_err(core_error_to_pyerr)
        })
    }

    fn recv_text<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.recv_text().await.map_err(core_error_to_pyerr)
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.close().await.map_err(core_error_to_pyerr)
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ResponsesWebSocketConnection>()
}
