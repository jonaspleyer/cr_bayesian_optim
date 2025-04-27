"""
>>> import cr_bayesian_optim as crb
>>> a = crb.sum_as_string(1, 2)
>>> assert a == "3"
"""

from .cr_bayesian_optim import (
    run_sim_branching,
    load_results,
    load_results_at_iteration,
    get_all_iterations,
    Options,
    BacterialParameters,
    DomainParameters,
    TimeParameters,
    BacteriaBranching,
)

type CellIdentifier = tuple[int, int]
type CellOutput = dict[
    int, dict[CellIdentifier, tuple[BacteriaBranching, CellIdentifier | None]]
]

from .plotting import *

import cr_bayesian_optim.sim_branching as sim_branching

from .fractal_dim import fractal_dim_main
