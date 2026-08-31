mod diagnostics;
mod errors;
mod marshal;
mod responses_websocket;
mod routes;

use pyo3::prelude::*;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(module)?;
    routes::register(module)?;
    responses_websocket::register(module)?;
    diagnostics::register(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_registers_the_complete_python_surface() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_native").expect("module should be created");
            _native(&module).expect("module should initialize");

            for name in [
                "ocr",
                "aocr",
                "transcription",
                "atranscription",
                "messages",
                "amessages",
                "chat_completions_decline",
                "chat_completions",
                "achat_completions",
                "ResponsesWebSocketConnection",
                "RustBridgeDeclined",
                "RustUpstreamError",
                "gil_stats",
            ] {
                assert!(
                    module
                        .hasattr(name)
                        .expect("attribute lookup should succeed")
                );
            }
        });
    }
}
