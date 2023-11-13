use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;

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
    let mut new_state = state.to_owned_array();
    let kraus_ops: Vec<Array2<Complex64>> = kraus_operators
        .iter()
        .map(|&op| op.to_owned_array())
        .collect();

    for _ in 0..num_steps {
        let temp_state: Array2<Complex64> = kraus_ops.par_iter()
            .map(|e| {
                let e_dag = dag(e);
                e.dot(&new_state).dot(&e_dag)
            })
            .reduce_with(|a, b| a + b)
            .unwrap_or_else(|| Array2::<Complex64>::zeros(new_state.dim()));
        
        new_state = temp_state;
    }

    Ok(new_state.to_pyarray(py).to_owned().into())
}


#[pyfunction]
fn apply_error_channel_with_unitary(
    py: Python,
    state: &PyArray2<Complex64>,
    num_steps: usize,
    kraus_operators: Vec<&PyArray2<Complex64>>,
    fractional_unitary: Option<&PyArray2<Complex64>>,
) -> PyResult<PyObject> {
    let mut new_state = state.to_owned_array();
    let kraus_ops: Vec<Array2<Complex64>> = kraus_operators
        .iter()
        .map(|&op| op.to_owned_array())
        .collect();

    for _ in 0..num_steps {
        // Apply the fractional unitary if it's provided
        if let Some(frac_unitary) = fractional_unitary {
            let frac_unitary_array = frac_unitary.to_owned_array();
            new_state = frac_unitary_array.dot(&new_state).dot(&frac_unitary_array.t().mapv(|elem| elem.conj()));
        }

        // Apply the error channel
        let mut temp_state = Array2::<Complex64>::zeros(new_state.dim());
        for e in &kraus_ops {
            let e_dag = dag(e);
            temp_state += &e.dot(&new_state).dot(&e_dag);
        }
        new_state = temp_state;
    }

    Ok(new_state.to_pyarray(py).to_owned().into())
}



#[pymodule]
fn _lib(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_error_channel_rust, m)?)?;
    m.add_function(wrap_pyfunction!(apply_error_channel_with_unitary, m)?)?;
    Ok(())
}
