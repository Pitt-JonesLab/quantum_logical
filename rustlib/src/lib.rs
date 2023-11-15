use pyo3::prelude::*;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::OwnedRepr;
use ndarray::Dim;
use num_complex::Complex64;
use rayon::prelude::*;
use numpy::{PyArray2, ToPyArray};

fn dag(op: &Array2<Complex64>) -> Array2<Complex64> {
    op.t().mapv(|elem| elem.conj())
}

#[pyfunction]
fn apply_operators_in_place(
    py: Python,
    state: &PyArray2<Complex64>,
    num_steps: usize,
    operators: Vec<&PyArray2<Complex64>>
) -> PyResult<PyObject> {
    let mut new_state = state.to_owned_array();
    let ops: Vec<Array2<Complex64>> = operators
        .iter()
        .map(|&op| op.to_owned_array())
        .collect();

    let dags: Vec<Array2<Complex64>> = ops.iter().map(|op| dag(op)).collect();

    for _ in 0..num_steps {
        let temp_states: Vec<Array2<Complex64>> = ops.par_iter().zip(dags.par_iter())
            .map(|(op, op_dag)| op.dot(&new_state).dot(op_dag))
            .collect();

        let sum_state: Array2<Complex64> = temp_states.into_par_iter().reduce_with(|a, b| a + b).unwrap();
        new_state = sum_state;
    }

    Ok(new_state.to_pyarray(py).to_owned().into())
}

#[pymodule]
fn _lib(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_operators_in_place, m)?)?;
    Ok(())
}
