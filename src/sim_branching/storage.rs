use cellular_raza::prelude::{self as cr, StorageInterfaceLoad};
use pyo3::prelude::*;

use super::{BacteriaBranching, CellOutput, SingleIterCells, SingleIterSubDomains};

fn cell_storage_for_loading(
    path: &std::path::Path,
) -> Result<
    cr::StorageManager<cr::CellIdentifier, (cr::CellBox<BacteriaBranching>, serde_json::Value)>,
    cr::SimulationError,
> {
    let storage_builder = cr::StorageBuilder::new()
        .priority([cr::StorageOption::SerdeJson])
        .location(path)
        .add_date(false)
        .suffix("cells")
        .init();
    let cells = cr::StorageManager::<
        cr::CellIdentifier,
        (cr::CellBox<BacteriaBranching>, serde_json::Value),
    >::open_or_create(storage_builder, 0)?;
    Ok(cells)
}

fn subdomain_storage_for_loading(
    path: &std::path::Path,
) -> Result<cr::StorageManager<cr::SubDomainPlainIndex, serde_json::Value>, cr::SimulationError> {
    let storage_builder = cr::StorageBuilder::new()
        .priority([cr::StorageOption::SerdeJson])
        .location(path)
        .add_date(false)
        .suffix("subdomains")
        .init();
    Ok(cr::StorageManager::open_or_create(storage_builder, 0)?)
}

#[pyfunction]
pub fn load_cells(path: std::path::PathBuf) -> Result<CellOutput, cr::SimulationError> {
    let cells = cell_storage_for_loading(&path)?;
    use cr::StorageInterfaceLoad;
    Ok(cells
        .load_all_elements()?
        .into_iter()
        .map(|(iteration, cells)| {
            (
                iteration,
                cells
                    .into_iter()
                    .map(|(ident, (cbox, _))| (ident, (cbox.cell, cbox.parent)))
                    .collect(),
            )
        })
        .collect())
}

#[pyfunction]
pub fn load_cells_at_iteration(
    path: std::path::PathBuf,
    iteration: u64,
) -> Result<SingleIterCells, cr::SimulationError> {
    let cells = cell_storage_for_loading(&path)?;
    Ok(cells
        .load_all_elements_at_iteration(iteration)?
        .into_iter()
        .map(|(ident, (cbox, _))| (ident, (cbox.cell, cbox.parent)))
        .collect())
}

/// Obtains all iterations for a given result.
///
/// The same comment from :meth:`load_cells` applies.
#[pyfunction]
pub fn get_all_iterations(path: std::path::PathBuf) -> Result<Vec<u64>, cr::SimulationError> {
    let cells = cell_storage_for_loading(&path)?;
    Ok(cells.get_all_iterations()?)
}

#[pyfunction]
pub fn load_subdomains_at_iteration(
    py: Python,
    path: std::path::PathBuf,
    iteration: u64,
) -> PyResult<SingleIterSubDomains> {
    let subdomains = subdomain_storage_for_loading(&path)?;
    Ok(subdomains
        .load_all_elements_at_iteration(iteration)
        .map_err(cr::SimulationError::from)?
        .into_iter()
        .map(|(iteration, element)| Ok((iteration, pythonize::pythonize(py, &element)?)))
        .collect::<PyResult<_>>()?)
}
