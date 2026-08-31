use futures_util::stream;
use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection as RustConnection;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};

use crate::errors::BridgeResult;
use crate::marshal::{optional_timeout, string_headers};
use crate::routes::receiver::BridgeReceiver;
use crate::routes::runtime::{run_async, run_async_with};

struct OpenedConnection {
    connection: RustConnection,
    inbound: BridgeReceiver<String>,
}

impl OpenedConnection {
    async fn connect(
        url: String,
        headers: std::collections::HashMap<String, String>,
        timeout: Option<std::time::Duration>,
    ) -> BridgeResult<Self> {
        let connection = RustConnection::connect_url(&url, &headers, timeout)
            .await
            .map_err(crate::errors::BridgeError::from)?;
        let inbound_connection = connection.clone();
        let inbound = BridgeReceiver::from_stream(stream::unfold(
            inbound_connection,
            |connection| async move {
                match connection.recv_text().await {
                    Ok(Some(text)) => Some((Ok(text), connection)),
                    Ok(None) => None,
                    Err(error) => Some((Err(error.into()), connection)),
                }
            },
        ));
        Ok(Self {
            connection,
            inbound,
        })
    }
}

#[pyclass]
struct ResponsesWebSocketConnection {
    connection: RustConnection,
    inbound: BridgeReceiver<String>,
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
        run_async_with(
            py,
            OpenedConnection::connect(url, headers, timeout),
            |py, opened| {
                Ok(Py::new(
                    py,
                    ResponsesWebSocketConnection {
                        connection: opened.connection,
                        inbound: opened.inbound,
                    },
                )?
                .into_any())
            },
        )
    }

    fn send_text<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let connection = self.connection.clone();
        run_async(py, async move {
            connection.send_text(text).await.map_err(Into::into)
        })
    }

    fn recv_text<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inbound = self.inbound.clone();
        run_async(py, async move { inbound.next().await })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let connection = self.connection.clone();
        let inbound = self.inbound.clone();
        run_async(py, async move {
            inbound.close();
            connection.close().await.map_err(Into::into)
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ResponsesWebSocketConnection>()
}
