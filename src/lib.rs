use cellular_raza::prelude as cr;
use pyo3::prelude::*;

mod sim_branching;

use sim_branching::*;

/// A Python module implemented in Rust.
#[pymodule]
fn cr_bayesian_optim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sim_branching, m)?)?;
    m.add_class::<cr::NewtonDamped2D>()?;
    m.add_class::<cr::MorsePotential>()?;
    m.add_class::<cr::CellIdentifier>()?;
    Ok(())
}
