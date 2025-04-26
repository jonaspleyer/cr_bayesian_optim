use pyo3::prelude::*;

mod sim_branching;

use sim_branching::*;

/// A Python module implemented in Rust.
#[pymodule]
fn cr_bayesian_optim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sim_branching, m)?)?;
    Ok(())
}
