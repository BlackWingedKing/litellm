use litellm_python_bridge::Error;

#[test]
fn error_is_exported_from_the_crate_root() {
    let error: Option<Error> = None;
    assert!(error.is_none());
}
