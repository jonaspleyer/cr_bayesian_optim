import cr_bayesian_optim as crb
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Define Parameters for Simulation
    options = crb.Options(show_progressbar=True)
    options.domain.domain_size = 2000
    options.time.t_max = 2000

    # Load Results or Run new Simulation if needed
    cells, out_path = crb.sim_branching.load_or_compute_full(options)

    # Specify bounds for visualizing the intracellular
    # and extracellular concentrations
    intra_bounds = (0.5, 1)
    extra_bounds = (0, 10.0)

    # Plots the last iteration of the simulation
    last_iter = sorted(cells.keys())[-1]
    crb.plotting.plot_iteration(
        last_iter,
        intra_bounds,
        extra_bounds,
        out_path,
    )

    # Simple Plot to visualize the Growth Curve
    x = np.array(sorted(cells.keys())) * options.time.dt / 60
    y = [len(cells[i]) for i in cells.keys()]

    fig, ax = plt.subplots()
    ax.set_title("Number of Cells")
    ax.plot(x, y, color=crb.plotting.COLOR1)
    ax.set_xlabel("Time [min]")
    plt.show()
