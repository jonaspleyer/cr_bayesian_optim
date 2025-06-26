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
from scipy.optimize import differential_evolution
from skopt import gp_minimize, callbacks
import pickle


def rhs_fractal_dim(options: Options) -> tuple[float, float]:
    """
    Parameters
    ----------
    options: Options
        Options to run/load the branching simulation.
        Be aware that the `storage_location` should be set to `None`
        since otherwise, many disk space would be used (and probably
        not reused if running optimization again).

    Returns
    -------
    fractal_dim: float
        The fractal dimension of the last iteration.
    fractal_dim_err: float
        The uncertainty of the fractal dimension.
    """
    cells, _ = load_or_compute_last_iter(options.storage_location)
    pos = np.array([c[0].mechanics.pos for c in cells.values()], dtype=float)

    _, _, popt, pcov = calculate_fractal_dim_for_pos(pos, options)
    return popt[0], pcov[0, 0] ** 0.5


def optimization_diff_evolution(cost, bnds, args=(), workers=-1):
    return differential_evolution(cost,
                                  bounds=bnds,
                                  tol=1e-3,
                                  atol=1e-3,
                                  maxiter=10,
                                  #mutation=(0.3, 1.9),
                                  #recombination=0.7,
                                  popsize=5,
                                  init='latinhypercube',
                                  disp=True,
                                  polish=False,
                                  updating='deferred',
                                  workers=workers,    
                                  strategy='randtobest1bin',
                                  callback=callback_diffevol)#, callback=callback_ll) #init='sobol'


def callback_diffevol(intermediate_result):
    with open("out/Optimization_result_diffevol.pkl", 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(intermediate_result, outp, pickle.HIGHEST_PROTOCOL)


def optimization_bayes(cost, bnds, args=(), workers=-1):
    return gp_minimize(cost,
                      bnds,
                      acq_func="EI",            # the acquisition function: EI, LCB, MES, gp_hedge, PVRS, PI, EIps, PIps
                      n_calls=20,               # the number of evaluations of f
                      n_random_starts=5,        # the number of random initialization points
                      noise=0.,                 # the noise level (optional)
                      random_state=1234,
                      kappa=1.96,
                      xi=0.01,
                      acq_optimizer='lbfgs',    # is needed for parallelization
                      n_restarts_optimizer=5,   # the number of restarts of the optimizer
                      n_jobs=workers,           # the number of parallel evaluations of f
                      callback=[callbacks.CheckpointSaver("out/Optimization_result.pkl")], # a callback function to be called after each iteration
                      )


def save_optimization_result(res, path='', add_filename=''):
    with open(path+'Final_optimization_result'+add_filename+'.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(res, outp, pickle.HIGHEST_PROTOCOL)


def load_optimization_result(path='', add_filename=''):
    with open(path+'Final_optimization_result'+add_filename+'.pkl', 'rb') as inp:
        res = pickle.load(inp)
    return res