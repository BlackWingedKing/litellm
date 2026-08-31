use std::future::Future;
use std::sync::mpsc::sync_channel;

use litellm_core::error::{CoreError, CoreResult};
use litellm_python_interop::{release_gil, to_py};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Serialize;

pub(crate) fn run_sync<T, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(CoreError) -> PyErr,
) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = CoreResult<T>> + Send + 'static,
{
    let (sender, receiver) = sync_channel(1);
    pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
        let _ = sender.send(future.await);
    });
    let result = release_gil(py, move || receiver.recv())
        .map_err(|_| PyRuntimeError::new_err("native route task terminated"))?
        .map_err(map_error)?;
    to_py(py, &result)
}

pub(crate) fn run_async<T, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(CoreError) -> PyErr,
) -> PyResult<Bound<'_, PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = CoreResult<T>> + Send + 'static,
{
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = future.await.map_err(map_error)?;
        Python::attach(|py| to_py(py, &result))
    })
}

macro_rules! bridge_route {
    (
        sync = $sync_name:ident,
        asynchronous = $async_name:ident,
        inputs = $inputs:ident,
        required = { $($required_name:ident: $required_type:ty),* $(,)? },
        optional = { $($optional_name:ident: $optional_type:ty),* $(,)? },
        prepare = $prepare:path,
        errors = $map_error:path
        $(, extra = [$($extra:ident),* $(,)?])?
        $(,)?
    ) => {
        struct $inputs {
            $($required_name: $required_type,)*
            $($optional_name: $optional_type),*
        }

        #[pyfunction]
        #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
        #[allow(clippy::too_many_arguments)]
        fn $sync_name(
            py: pyo3::Python<'_>,
            $($required_name: $required_type,)*
            $($optional_name: $optional_type),*
        ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let call = $prepare(py, $inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            crate::route::run_sync(py, call.run(), $map_error)
        }

        #[pyfunction]
        #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
        #[allow(clippy::too_many_arguments)]
        fn $async_name(
            py: pyo3::Python<'_>,
            $($required_name: $required_type,)*
            $($optional_name: $optional_type),*
        ) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
            let call = $prepare(py, $inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            crate::route::run_async(py, call.run(), $map_error)
        }

        pub(super) fn register(
            module: &pyo3::Bound<'_, pyo3::types::PyModule>,
        ) -> pyo3::PyResult<()> {
            module.add_function(pyo3::wrap_pyfunction!($sync_name, module)?)?;
            module.add_function(pyo3::wrap_pyfunction!($async_name, module)?)?;
            $($(module.add_function(pyo3::wrap_pyfunction!($extra, module)?)?;)*)?
            Ok(())
        }
    };
}

pub(crate) use bridge_route;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_runner_dispatches_without_holding_the_gil() {
        Python::initialize();
        Python::attach(|py| {
            let result = run_sync(py, async { Ok(42_u8) }, |error| {
                PyRuntimeError::new_err(error.to_string())
            })
            .expect("route should complete");

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
