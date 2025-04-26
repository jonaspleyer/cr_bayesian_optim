use cellular_raza::prelude as cr;
use pyo3::prelude::*;

mod sim_branching;

use sim_branching::*;

/// A Python module implemented in Rust.
#[pymodule]
fn cr_bayesian_optim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sim_branching, m)?)?;
    m.add_function(wrap_pyfunction!(load_results, m)?)?;
    m.add_function(wrap_pyfunction!(load_results_at_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_iterations, m)?)?;
    m.add_class::<Options>()?;
    m.add_class::<BacterialParameters>()?;
    m.add_class::<DomainParameters>()?;
    m.add_class::<TimeParameters>()?;
    m.add_class::<BacteriaBranching>()?;
    m.add_class::<cr::NewtonDamped2D>()?;
    m.add_class::<cr::MorsePotential>()?;
    m.add_class::<cr::CellIdentifier>()?;
    Ok(())
}
