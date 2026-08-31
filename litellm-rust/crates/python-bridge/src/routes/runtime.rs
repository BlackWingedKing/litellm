use std::future::Future;
use std::sync::mpsc::sync_channel;

use litellm_python_interop::{release_gil, to_py};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Serialize;

use crate::errors::BridgeResult;

pub(super) fn run_sync_with<T, F, C>(py: Python<'_>, future: F, convert: C) -> PyResult<Py<PyAny>>
where
    T: Send + 'static,
    F: Future<Output = BridgeResult<T>> + Send + 'static,
    C: FnOnce(Python<'_>, T) -> PyResult<Py<PyAny>>,
{
    let (sender, receiver) = sync_channel(1);
    pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
        let _ = sender.send(future.await);
    });
    let task_result = release_gil(py, move || receiver.recv())
        .map_err(|_| PyRuntimeError::new_err("native route task terminated"))?;
    let result = task_result?;
    convert(py, result)
}

pub(super) fn run_async_with<T, F, C>(
    py: Python<'_>,
    future: F,
    convert: C,
) -> PyResult<Bound<'_, PyAny>>
where
    T: Send + 'static,
    F: Future<Output = BridgeResult<T>> + Send + 'static,
    C: FnOnce(Python<'_>, T) -> PyResult<Py<PyAny>> + Send + 'static,
{
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = future.await?;
        Python::attach(|py| convert(py, result))
    })
}

pub(super) fn run_sync<T, F>(py: Python<'_>, future: F) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = BridgeResult<T>> + Send + 'static,
{
    run_sync_with(py, future, |py, value| to_py(py, &value))
}

pub(super) fn run_async<T, F>(py: Python<'_>, future: F) -> PyResult<Bound<'_, PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = BridgeResult<T>> + Send + 'static,
{
    run_async_with(py, future, |py, value| to_py(py, &value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_runner_dispatches_without_holding_the_gil() {
        Python::initialize();
        Python::attach(|py| {
            let result = run_sync(py, async { Ok(42_u8) }).expect("route should complete");

            assert_eq!(
                result
                    .bind(py)
                    .extract::<u8>()
                    .expect("result should convert"),
                42
            );
        });
    }
}
