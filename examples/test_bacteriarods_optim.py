import cr_mech_coli.crm_fit as crm_fit

from cr_bayesian_optim.optimize_bacterialrods import main_bacterialrods_optimization
from cr_bayesian_optim.plotting import *
from cr_bayesian_optim.optimization import *
#from skopt import gp_minimize

def update_test_ABM_framework(settings):
    n_vertices = 8
    settings.constants.n_vertices = n_vertices
    settings.constants.n_saves = 15
    #settings.parameters.damping = crm_fit.SampledFloat(min=0, max=2.5, initial=1.5)
    #settings.parameters.damping = 2.0
    settings.parameters.potential_type.Mie.en = 10.
    settings.parameters.potential_type.Mie.em = 1.5
    settings.others = crm_fit.Others(True)
    return settings

'''
def custom_optimizer(cost, bnds):
    return gp_minimize(cost,
                      bnds,
                      acq_func="LCB",
                      n_calls=20,
                      n_random_starts=3               
                      random_state=1234)
    # Possible variations:
    # - Different acquisition functions/kappa/xi/noise-?/
    # - Different optimization methods (e.g. differential evolution)
    # - Different parameter bounds or initial conditions / different constants in settings - ?
'''


if __name__ == "__main__":
    res = main_bacterialrods_optimization(optimization_bayes)#, update_ABM=update_test_ABM_framework)
    #res = main_bacterialrods_optimization(optimization_diff_evolution, update_ABM=update_test_ABM_framework)

    # Aquisition functions: # EI/PI: faster but less accurate (?)
                            # MES: slower, smaller cost / does not allow parallelization
                            # PVRS: very slow

    path_output =  'out/'#bacterialrods_optim/'
    add_name = '_EI'

    # Save and load optimization result
    save_optimization_result(res, path=path_output, add_filename=add_name)
    res = load_optimization_result(path=path_output, add_filename=add_name)

    # Plotting (only for bayesian optimization):
    plot_optimization_convergence_bayes(res, path=path_output, add_name=add_name)
    #plot_1D_cost_approximation_bayes(res, path=path_output, add_name=add_name)
    plot_objective_projection_bayes(res, path=path_output, add_name=add_name)