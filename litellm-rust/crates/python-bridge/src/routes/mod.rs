use pyo3::prelude::*;

macro_rules! routes {
    ($($route:ident),* $(,)?) => {
        $(mod $route;)*

        pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $($route::register(module)?;)*
            Ok(())
        }
    };
}

routes!(ocr, audio_transcription, messages, chat_completions);
