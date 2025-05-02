import cr_mech_coli as crm
import cr_mech_coli.crm_fit as crm_fit
import numpy as np

import matplotlib.pyplot as plt


def calculate_cost(pos, pos_fin_true):
    cost = 0
    for p2i in pos_fin_true:
        values = np.linalg.norm(pos[::-1, :, :2] - p2i, axis=(1, 2))
        cost += np.min(values)
    return cost


def main():
    n_vertices = 8

    # Extract data from masks which have been previously generated
    mask1 = np.loadtxt(
        "data/crm_fit/0001/masks/image001042-markers.csv", delimiter=","
    ).T
    mask2 = np.loadtxt(
        "data/crm_fit/0001/masks/image001052-markers.csv", delimiter=","
    ).T

    p1 = crm.extract_positions(mask1, n_vertices)[0]
    p2 = crm.extract_positions(mask2, n_vertices)[0]

    # Define settings required to run simulation
    settings = crm_fit.Settings.from_toml("data/crm_fit/0001/settings.toml")

    settings.parameters.damping = crm_fit.SampledFloat(min=0, max=2.5, initial=1.5)
    settings.constants.n_vertices = n_vertices
    settings.constants.n_saves = 20

    settings.others = crm_fit.Others(True)

    # Some values here are commented out. See the data/crm_fit/0001/settings.toml file
    parameters = [
        # 4.0,  # Radius
        2.0,  # Damping
        4.0,  # Strength
        # The growth rates are set individually as parameters.
        # This means that we need to assign values for each agent.
        # *[0.07] * p1.shape[0],  # Growth Rate
        4.0,  # Exponent n
        4.5,  # Exponent m
    ]

    # Do the preduction with the supplied values and initial positions
    container = crm_fit.predict(parameters, p1, settings)

    if container is None:
        print("Simulation Failed")
        exit()

    # Obtain Cost Function over the Simulation Time
    iterations = container.get_all_iterations()
    costs = []
    for i in iterations:
        cells = container.get_cells_at_iteration(i)
        keys = sorted(cells.keys())
        pos = np.array([cells[key][0].pos for key in keys])

        costs.append(calculate_cost(pos, p2))

    t = np.array(iterations) * settings.constants.dt / 60

    _, ax = plt.subplots(figsize=(8, 8))
    ax.plot(t, costs, c="k", label="Cost")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Cost [µm]")
    plt.show()


if __name__ == "__main__":
    main()
