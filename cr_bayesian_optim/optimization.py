"""
>>> import cr_bayesian_optim as crb
>>> options = crb.Options()
>>> dim, dim_err = crb.optimization.rhs_fractal_dim(options)
"""

from .cr_bayesian_optim import Options
from cr_bayesian_optim.sim_branching import (
    calculate_fractal_dim_for_pos,
    load_or_compute_last_iter,
)

import numpy as np


def rhs_fractal_dim(options: Options) -> tuple[float, float]:
    cells, _ = load_or_compute_last_iter(options.storage_location)
    pos = np.array([c[0].mechanics.pos for c in cells.values()], dtype=float)

    _, _, popt, pcov = calculate_fractal_dim_for_pos(pos, options)
    return popt[0], pcov[0, 0] ** 0.5
