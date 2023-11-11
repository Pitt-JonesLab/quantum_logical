use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use ndarray::Array2;
use num_complex::Complex64;

fn dag(matrix: &Array2<Complex64>) -> Array2<Complex64> {
    matrix.mapv(|elem| elem.conj()).reversed_axes()
}

#[pyfunction]
fn apply_error_channel_rust(
    py: Python,
    state: &PyArray2<Complex64>,
    num_steps: usize,
    kraus_operators: Vec<&PyArray2<Complex64>>,
) -> PyResult<PyObject> {
    let state_array = state.to_owned_array();
    let kraus_ops: Vec<Array2<Complex64>> = kraus_operators
        .iter()
        .map(|&op| op.to_owned_array())
        .collect();

    let mut new_state = state_array;

    for _ in 0..num_steps {
        let mut temp_state = Array2::<Complex64>::zeros(new_state.dim());
        for e in &kraus_ops {
            let e_dag = dag(e);
            temp_state = temp_state + e.dot(&new_state).dot(&e_dag);
        }
        new_state = temp_state;
    }

    Ok(new_state.to_pyarray(py).to_owned().into())
}


#[pymodule]
fn _lib(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_error_channel_rust, m)?)?;
    Ok(())
}
