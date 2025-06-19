use pyo3::prelude::*;

/// A test function that returns a string from Rust.
#[pyfunction]
fn hello_from_rust() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

/// The Python module definition.
#[pymodule]
fn aqft_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    Ok(())
}
