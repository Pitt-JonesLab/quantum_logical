use ndarray::Array2;
use ndarray::OwnedRepr;
use num_complex::Complex64;
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;

fn dag(op: &Array2<Complex64>) -> Array2<Complex64> {
    op.t().mapv(|elem| elem.conj())
}

#[pyfunction]
fn apply_operators_in_place(
    py: Python,
    state: &PyArray2<Complex64>,
    num_steps: usize,
    operator_groups: Vec<Vec<&PyArray2<Complex64>>>,
) -> PyResult<PyObject> {
    let mut state_numpy = state.to_owned_array();

    for _ in 0..num_steps {
        for operator_group in &operator_groups {
            if operator_group.len() > 1 {
                // Assuming this is a group of Kraus operators
                let mut new_state = Array2::<Complex64>::zeros(state_numpy.raw_dim());
                for op in operator_group {
                    let op_numpy = op.to_owned_array();
                    let op_dag = dag(&op_numpy);
                    new_state = new_state + op_numpy.dot(&state_numpy).dot(&op_dag);
                }
                state_numpy = new_state;
            } else {
                // Assuming this is a unitary operation
                let op_numpy = operator_group[0].to_owned_array();
                let op_dag = dag(&op_numpy);
                state_numpy = op_numpy.dot(&state_numpy).dot(&op_dag);
            }
        }
    }

    Ok(state_numpy.to_pyarray(py).to_owned().into())
}

#[pymodule]
fn _lib(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_operators_in_place, m)?)?;
    Ok(())
}
