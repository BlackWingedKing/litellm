use std::future::Future;

use pyo3::prelude::*;
use serde::Serialize;

use crate::errors::BridgeResult;

mod receiver;
mod runtime;

use runtime::{run_async, run_sync};

trait BridgeRoute<I>: Sized {
    type Output: Serialize + Send + 'static;

    fn from_python(py: Python<'_>, inputs: I) -> BridgeResult<Self>;

    fn run(self) -> impl Future<Output = BridgeResult<Self::Output>> + Send + 'static;
}

macro_rules! bridge_route {
    (
        sync = $sync_name:ident,
        asynchronous = $async_name:ident,
        inputs = $inputs:ident,
        required = { $($required_name:ident: $required_type:ty),* $(,)? },
        optional = { $($optional_name:ident: $optional_type:ty),* $(,)? },
        call = $call:ty
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
            let call = <$call as crate::routes::BridgeRoute<$inputs>>::from_python(py, $inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            crate::routes::run_sync(py, <$call as crate::routes::BridgeRoute<$inputs>>::run(call))
        }

        #[pyfunction]
        #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
        #[allow(clippy::too_many_arguments)]
        fn $async_name(
            py: pyo3::Python<'_>,
            $($required_name: $required_type,)*
            $($optional_name: $optional_type),*
        ) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
            let call = <$call as crate::routes::BridgeRoute<$inputs>>::from_python(py, $inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            crate::routes::run_async(py, <$call as crate::routes::BridgeRoute<$inputs>>::run(call))
        }

        pub(super) fn register(
            module: &pyo3::Bound<'_, pyo3::types::PyModule>,
        ) -> pyo3::PyResult<()> {
            module.add_function(pyo3::wrap_pyfunction!($sync_name, module)?)?;
            module.add_function(pyo3::wrap_pyfunction!($async_name, module)?)?;
            Ok(())
        }
    };
}

macro_rules! routes {
    ($($route:ident),* $(,)?) => {
        $(mod $route;)*

        pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $($route::register(module)?;)*
            Ok(())
        }
    };
}

routes!(
    ocr,
    audio_transcription,
    messages,
    chat_completions,
    streaming,
);
