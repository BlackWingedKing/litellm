use std::time::Duration;

use futures_util::{StreamExt, stream};
use litellm_core::error::{CoreError, ProviderCallError};
use litellm_core::http_utils::truncate_error_body;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use reqwest::{Client, Method, Response};

use crate::errors::{BridgeResult, Error};
use crate::marshal::{optional_timeout, string_headers};
use crate::routes::receiver::BridgeReceiver;
use crate::routes::runtime::{run_async, run_async_with, run_sync_with};

const ERROR_BODY_LIMIT_BYTES: usize = 4096;

struct HttpStreamCall {
    method: Method,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    open_timeout: Option<Duration>,
    read_idle_timeout: Option<Duration>,
}

struct OpenedHttpStream {
    status_code: u16,
    headers: Vec<(String, String)>,
    receiver: BridgeReceiver<Vec<u8>>,
}

impl HttpStreamCall {
    fn from_python(
        py: Python<'_>,
        method: String,
        url: String,
        headers: Option<Py<PyAny>>,
        body: Option<Py<PyBytes>>,
        open_timeout_seconds: Option<f64>,
        read_idle_timeout_seconds: Option<f64>,
    ) -> BridgeResult<Self> {
        (|| -> PyResult<Self> {
            let method = Method::from_bytes(method.as_bytes())
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
            Ok(Self {
                method,
                url,
                headers: string_headers(py, headers)?.into_iter().collect(),
                body: body
                    .map(|body| body.bind(py).as_bytes().to_vec())
                    .unwrap_or_default(),
                open_timeout: optional_timeout(open_timeout_seconds),
                read_idle_timeout: optional_timeout(read_idle_timeout_seconds),
            })
        })()
        .map_err(Error::declined)
    }

    async fn run(self) -> BridgeResult<OpenedHttpStream> {
        let client = Client::builder()
            .build()
            .map_err(|error| Error::declined(CoreError::Connect(error.to_string())))?;
        let mut request = client.request(self.method, &self.url).body(self.body);
        for (name, value) in self.headers {
            request = request.header(name, value);
        }
        let response = match self.open_timeout {
            Some(timeout) => tokio::time::timeout(timeout, request.send())
                .await
                .map_err(|_| {
                    Error::from(ProviderCallError::PossiblySent(CoreError::Network(
                        "stream response open timed out".to_string(),
                    )))
                })?,
            None => request.send().await,
        }
        .map_err(|error| {
            if error.is_connect() || error.is_builder() {
                Error::declined(CoreError::Connect(error.to_string()))
            } else {
                Error::from(ProviderCallError::PossiblySent(CoreError::Network(
                    error.to_string(),
                )))
            }
        })?;

        if !response.status().is_success() {
            return Err(Error::from(ProviderCallError::PossiblySent(
                response_error(response, self.read_idle_timeout).await,
            )));
        }

        let status_code = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .map(|(name, value)| {
                value
                    .to_str()
                    .map(|value| (name.as_str().to_string(), value.to_string()))
                    .map_err(|error| CoreError::InvalidResponse(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| Error::from(ProviderCallError::PossiblySent(error)))?;
        let timeout = self.read_idle_timeout;
        let byte_stream = stream::unfold(response.bytes_stream(), move |mut body| async move {
            let next = match timeout {
                Some(timeout) => match tokio::time::timeout(timeout, body.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        return Some((
                            Err(Error::from(ProviderCallError::PossiblySent(CoreError::Network(
                                "stream response read timed out".to_string(),
                            )))),
                            body,
                        ));
                    }
                },
                None => body.next().await,
            };
            next.map(|item| {
                (
                    item.map(|bytes| bytes.to_vec()).map_err(|error| {
                        Error::from(ProviderCallError::PossiblySent(CoreError::Network(
                            error.to_string(),
                        )))
                    }),
                    body,
                )
            })
        });
        Ok(OpenedHttpStream {
            status_code,
            headers,
            receiver: BridgeReceiver::from_stream(byte_stream),
        })
    }
}

async fn response_error(response: Response, read_idle_timeout: Option<Duration>) -> CoreError {
    let status = response.status().as_u16();
    let mut body = response.bytes_stream();
    let mut captured = Vec::new();
    while captured.len() < ERROR_BODY_LIMIT_BYTES {
        let next = match read_idle_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, body.next()).await {
                Ok(next) => next,
                Err(_) => break,
            },
            None => body.next().await,
        };
        let Some(chunk) = next else {
            break;
        };
        let Ok(chunk) = chunk else {
            break;
        };
        let remaining = ERROR_BODY_LIMIT_BYTES - captured.len();
        captured.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    CoreError::Http {
        status,
        body: truncate_error_body(&String::from_utf8_lossy(&captured)),
    }
}

#[pyclass]
struct HttpResponseStream {
    status_code: u16,
    headers: Vec<(String, String)>,
    receiver: BridgeReceiver<Vec<u8>>,
}

impl From<OpenedHttpStream> for HttpResponseStream {
    fn from(opened: OpenedHttpStream) -> Self {
        Self {
            status_code: opened.status_code,
            headers: opened.headers,
            receiver: opened.receiver,
        }
    }
}

#[pymethods]
impl HttpResponseStream {
    #[getter]
    fn status_code(&self) -> u16 {
        self.status_code
    }

    #[getter]
    fn headers(&self) -> Vec<(String, String)> {
        self.headers.clone()
    }

    fn next_bytes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let receiver = self.receiver.clone();
        run_sync_with(
            py,
            async move { receiver.next().await },
            |py, item| match item {
                Some(bytes) => Ok(PyBytes::new(py, &bytes).unbind().into_any()),
                None => Ok(py.None()),
            },
        )
    }

    fn anext_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = self.receiver.clone();
        run_async_with(
            py,
            async move { receiver.next().await },
            |py, item| match item {
                Some(bytes) => Ok(PyBytes::new(py, &bytes).unbind().into_any()),
                None => Ok(py.None()),
            },
        )
    }

    fn close(&self) {
        self.receiver.close();
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = self.receiver.clone();
        run_async(py, async move {
            receiver.close();
            Ok(())
        })
    }
}

#[pyfunction]
#[pyo3(signature = (method, url, headers=None, body=None, open_timeout_seconds=None, read_idle_timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn open_http_stream(
    py: Python<'_>,
    method: String,
    url: String,
    headers: Option<Py<PyAny>>,
    body: Option<Py<PyBytes>>,
    open_timeout_seconds: Option<f64>,
    read_idle_timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let call = HttpStreamCall::from_python(
        py,
        method,
        url,
        headers,
        body,
        open_timeout_seconds,
        read_idle_timeout_seconds,
    )?;
    run_sync_with(py, call.run(), |py, opened| {
        Ok(Py::new(py, HttpResponseStream::from(opened))?.into_any())
    })
}

#[pyfunction]
#[pyo3(signature = (method, url, headers=None, body=None, open_timeout_seconds=None, read_idle_timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn aopen_http_stream(
    py: Python<'_>,
    method: String,
    url: String,
    headers: Option<Py<PyAny>>,
    body: Option<Py<PyBytes>>,
    open_timeout_seconds: Option<f64>,
    read_idle_timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    let call = HttpStreamCall::from_python(
        py,
        method,
        url,
        headers,
        body,
        open_timeout_seconds,
        read_idle_timeout_seconds,
    )?;
    run_async_with(py, call.run(), |py, opened| {
        Ok(Py::new(py, HttpResponseStream::from(opened))?.into_any())
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HttpResponseStream>()?;
    module.add_function(wrap_pyfunction!(open_http_stream, module)?)?;
    module.add_function(wrap_pyfunction!(aopen_http_stream, module)?)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;

    fn serve(
        response: impl FnOnce(TcpStream) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind server");
        let address = listener.local_addr().expect("server address");
        let task = thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("accept request");
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).expect("read request");
            response(socket);
        });
        (format!("http://{address}"), task)
    }

    fn call(
        url: String,
        open_timeout: Option<Duration>,
        read_timeout: Option<Duration>,
    ) -> HttpStreamCall {
        HttpStreamCall {
            method: Method::GET,
            url,
            headers: Vec::new(),
            body: Vec::new(),
            open_timeout,
            read_idle_timeout: read_timeout,
        }
    }

    #[tokio::test]
    async fn returns_after_headers_and_streams_ordered_chunks() {
        let (url, server) = serve(|mut socket| {
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nx-test: ready\r\ntransfer-encoding: chunked\r\n\r\n",
                )
                .expect("write headers");
            socket.flush().expect("flush headers");
            thread::sleep(Duration::from_millis(300));
            socket.write_all(b"3\r\none\r\n").expect("first chunk");
            socket.flush().expect("flush first chunk");
            thread::sleep(Duration::from_millis(20));
            socket
                .write_all(b"3\r\ntwo\r\n0\r\n\r\n")
                .expect("second chunk");
        });
        let started = Instant::now();
        let opened = tokio::time::timeout(
            Duration::from_millis(200),
            call(
                url,
                Some(Duration::from_secs(1)),
                Some(Duration::from_secs(1)),
            )
            .run(),
        )
        .await
        .expect("stream opens before its body")
        .expect("successful response");

        assert!(started.elapsed() < Duration::from_millis(250));
        assert!(
            opened
                .headers
                .contains(&("x-test".to_string(), "ready".to_string()))
        );
        assert_eq!(opened.status_code, 200);
        assert_eq!(
            opened.receiver.next().await.expect("first read"),
            Some(b"one".to_vec())
        );
        assert_eq!(
            opened.receiver.next().await.expect("second read"),
            Some(b"two".to_vec())
        );
        assert_eq!(opened.receiver.next().await.expect("end of stream"), None);
        server.join().expect("server task");
    }

    #[tokio::test]
    async fn enforces_open_and_read_idle_timeouts_separately() {
        let (open_url, open_server) = serve(|_socket| {
            thread::sleep(Duration::from_millis(100));
        });
        assert!(matches!(
            call(open_url, Some(Duration::from_millis(20)), None).run().await,
            Err(Error::ProviderCall(ProviderCallError::PossiblySent(
                CoreError::Network(message)
            )))
                if message == "stream response open timed out"
        ));
        open_server.join().expect("open timeout server");

        let (read_url, read_server) = serve(|mut socket| {
            socket
                .write_all(b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n")
                .expect("write headers");
            socket.flush().expect("flush headers");
            thread::sleep(Duration::from_millis(100));
        });
        let opened = call(read_url, None, Some(Duration::from_millis(20)))
            .run()
            .await
            .expect("stream opens");
        assert!(matches!(
            opened.receiver.next().await,
            Err(Error::ProviderCall(ProviderCallError::PossiblySent(
                CoreError::Network(message)
            )))
                if message == "stream response read timed out"
        ));
        read_server.join().expect("read timeout server");
    }

    #[tokio::test]
    async fn rejects_non_success_before_returning_a_handle() {
        let body = "x".repeat(ERROR_BODY_LIMIT_BYTES + 512);
        let expected_length = body.len();
        let (url, server) = serve(move |mut socket| {
            write!(
                socket,
                "HTTP/1.1 429 Too Many Requests\r\ncontent-length: {expected_length}\r\n\r\n{body}"
            )
            .expect("write response");
        });

        let error = match call(url, None, None).run().await {
            Ok(_) => panic!("non-success response returned a stream"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            Error::ProviderCall(ProviderCallError::PossiblySent(CoreError::Http {
                status: 429,
                body,
            }))
                if body.len() <= ERROR_BODY_LIMIT_BYTES
        ));
        server.join().expect("server task");
    }
}
