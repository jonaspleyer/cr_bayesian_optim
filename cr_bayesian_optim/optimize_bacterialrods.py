import cr_mech_coli as crm
import cr_mech_coli.crm_fit as crm_fit

import numpy as np
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization, acquisition


def extract_data(image_timesteps, n_vertices):
    masks  = [np.loadtxt(f"data/crm_fit/0001/masks/image0010{im}-markers.csv", delimiter=",").T
              for im in image_timesteps]
    pos_data = [crm.extract_positions(mask, n_vertices)[0] for mask in masks]
    iterations_data = [float(im) for im in image_timesteps]
    data = [np.array(iterations_data)[1:], np.array(pos_data)[1:]]
    return data


def cost(data, settings, init_pos, *param):
    (days, x_target) = data
    container = crm_fit.predict(param, init_pos, settings)
    if container is None:
        print("Simulation Failed")
        exit()
    iterations = container.get_all_iterations()
    x_prediction = np.zeros(np.shape(x_target))
    delta_iter = np.mean(np.array(iterations)[1:]-np.array(iterations)[:-1])
    ## TODO why is  saved iterations step changes from 952 to 953 ??
    iter_data = delta_iter*(np.array(days)-days[0]+1)
    ind_last = np.argmin(np.abs(iter_data[-1]-iterations))
    i = 0
    for iter in iterations[:ind_last+1]:
        if np.any(np.abs(iter_data-iter) <= 1.):
            cells = container.get_cells_at_iteration(iter)
            keys = sorted(cells.keys())
            # what is the last dimension: why 3 and not 2 ?
            pos = np.array([cells[key][0].pos for key in keys])[:, :, :-1]
            x_prediction[i] = pos
            i += 1
    return np.mean(squared_difference(x_target, x_prediction))


def squared_difference(x_target, x_prediction):
    return (x_target-x_prediction)**2


def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_objective_GP(optimizer, bnds, name=''):
    for k in bnds.keys():
        fig, ax = plt.subplots()
        x_gp = np.linspace(*bnds[k], 100)
        mean_gp, sigma_gp = posterior(optimizer, x_gp.reshape(-1, 1))
        ax.plot(x_gp, mean_gp, label=k)
        ax.fill_between(x_gp, mean_gp + sigma_gp, mean_gp - sigma_gp, alpha=0.1)
        ax.scatter(optimizer.space.params.flatten(), optimizer.space.target, c="red", s=50, zorder=10)
        ax.legend(fontsize=12)
        plt.savefig(f'{k}_{name}'+'.png', bbox_inches='tight')
        plt.close(fig)



def optimize_bacterialrods_main():
    n_vertices = 8
    # Extract data from masks which have been previously generated
    image_timesteps = ['42', '43', '44', '45', '46', '47', '48', '49', '52']
    data = extract_data(image_timesteps, n_vertices)
    
    # Target/model/simulation 
    # Define settings required to run simulation 
    settings = crm_fit.Settings.from_toml("data/crm_fit/0001/settings.toml")
    settings.constants.n_vertices = n_vertices
    settings.constants.n_saves = 15
    settings.others = crm_fit.Others(True)

    #settings.parameters.damping = crm_fit.SampledFloat(min=0, max=2.5, initial=1.5)
    settings.parameters.damping = 2.0
    settings.parameters.potential_type.Mie.en = 10.
    settings.parameters.potential_type.Mie.em = 1.5
    lower, upper, x0, param_infos, constants, constant_infos = settings.generate_optimization_infos(len(data[1][0]))
    print(param_infos)

    # Define the cost function with arguments as optimizes parameters:
    #cost_for_optimization = lambda Damping, Strength: cost(data, settings, data[1][0], Damping, Strength)
    #cost_for_optimization = lambda Damping: cost(data, settings, data[1][0], Damping)
    cost_for_optimization = lambda Strength: cost(data, settings, data[1][0], Strength)

    N_iter = 20
    acq = acquisition.ExpectedImprovement(1.) #ProbabilityOfImprovement(1.) #UpperConfidenceBound(kappa=1.)#
    bnds = {p_inf[0]: (u_b, l_b) for u_b, l_b, p_inf in zip(lower, upper, param_infos)}
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function=acq,
        pbounds=bnds,
        verbose=2,
        random_state=17695,
    )
    for j in range(N_iter):
        next_params = optimizer.suggest()
        target = cost_for_optimization(**next_params)
        optimizer.register(
            params=next_params,
            target=target,
        )
        plot_objective_GP(optimizer, bnds, name=f'EI_{j}')

   
if __name__ == "__main__":
    optimize_bacterialrods_main()
