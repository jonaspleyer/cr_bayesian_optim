import cr_mech_coli as crm
import cr_mech_coli.crm_fit as crm_fit
from functools import partial
import numpy as np


def extract_data(image_timesteps, n_vertices):
    masks = [
        np.loadtxt(
            f"data/crm_fit/0001/masks/image0010{im}-markers.csv", delimiter=","
        ).T
        for im in image_timesteps
    ]
    pos_data = [crm.extract_positions(mask, n_vertices)[0] for mask in masks]
    iterations_data = [float(im) for im in image_timesteps]
    data = [np.array(iterations_data)[1:], np.array(pos_data)[1:]]
    return data


def cost_bacterialrods(data, settings, init_pos, param):
    (days, x_target) = data
    container = crm_fit.predict(param, init_pos, settings)
    if container is None:
        print("Simulation Failed")
        return 1e10
    else:
        iterations = container.get_all_iterations()
        x_prediction = np.zeros(np.shape(x_target))
        delta_iter = np.mean(np.array(iterations)[1:] - np.array(iterations)[:-1])
        ## TODO why is  saved iterations step changes from 952 to 953 ??
        iter_data = delta_iter * (np.array(days) - days[0] + 1)
        ind_last = np.argmin(np.abs(iter_data[-1] - iterations))
        i = 0
        for iter in iterations[: ind_last + 1]:
            if np.any(np.abs(iter_data - iter) <= 1.0):
                cells = container.get_cells_at_iteration(iter)
                keys = sorted(cells.keys())
                # what is the last dimension: why 3 and not 2 ?
                pos = np.array([cells[key][0].pos for key in keys])[:, :, :-1]
                x_prediction[i] = pos
                i += 1
        return np.mean(squared_difference(x_target, x_prediction))


def squared_difference(x_target, x_prediction):
    return (x_target - x_prediction) ** 2


def create_test_ABM_framework():
    n_vertices = 8
    # Extract data from masks which have been previously generated
    image_timesteps = ["42", "43", "44", "45", "46", "47", "48", "49", "52"]
    data = extract_data(image_timesteps, n_vertices)

    # Target/model/simulation
    # Define settings required to run simulation
    settings = crm_fit.Settings.from_toml("data/crm_fit/0001/settings.toml")
    settings.constants.n_saves = 15
    settings.others = crm_fit.Others(True)
    return data, settings


def main_bacterialrods_optimization(optimizer, update_ABM=None):
    # Define ABM framework
    data, settings = create_test_ABM_framework()
    if update_ABM is not None:
        settings = update_ABM(settings)
    lower, upper, x0, param_infos, constants, constant_infos = (
        settings.generate_optimization_infos(len(data[1][0]))
    )
    bnds_dict = {
        p_inf[0]: (u_b, l_b) for u_b, l_b, p_inf in zip(lower, upper, param_infos)
    }
    print(param_infos)
    cost_for_optimization = partial(cost_bacterialrods, data, settings, data[1][0])
    res = optimizer(cost_for_optimization, [bnds_dict[k] for k in bnds_dict.keys()])
    print(res.x, res.fun)
    return res
