import numpy as np
import cr_bayesian_optim as crb
from cr_bayesian_optim.plotting import COLOR1, COLOR2, COLOR3, COLOR5
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sp


def produce_options():
    options = crb.Options(
        show_progressbar=True, storage_location="out/fractal_dim_multi"
    )
    options.time.t_max = 2000
    options.domain.domain_size = 2000
    options.time.dt = 0.3
    return options


def fractal_dim_over_time():
    options = produce_options()

    t = []
    y1 = []
    y1_err = []
    y2 = []

    diffusion_constants = [80, 5, 0.5]
    for diffusion_constant in diffusion_constants:
        options.domain.diffusion_constant = diffusion_constant
        cells, _ = crb.sim_branching.load_or_compute_full(options)

        iterations = sorted(cells.keys())[::4]
        colony_diam = []
        dims_mean = []
        dims_std = []
        for i in tqdm(iterations, desc="Calculating dim(t)"):
            pos = np.array([c[0].mechanics.pos for c in cells[i].values()])

            # Calculate Diameter of colony with convex hull
            hull = sp.spatial.ConvexHull(pos)
            hull_points = pos[hull.vertices]

            diam = 0
            n_points = int(np.ceil(len(hull_points) / 2))
            for p in hull_points[:n_points]:
                d = np.linalg.norm(hull_points - p, axis=1)
                diam = max(diam, np.max(d))
            colony_diam.append(diam)

            _, _, popt, pcov = crb.sim_branching.calculate_fractal_dim_for_pos(
                pos, options, None
            )
            dims_mean.append(-popt[0])
            dims_std.append(pcov[0, 0] ** 0.5)

        t.append(np.array(iterations) * options.time.dt / 60)
        y1.append(np.array(dims_mean))
        y1_err.append(np.array(dims_std))
        y2.append(np.array(colony_diam) / 1000)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Introduce new y-axis for colony size
    ax2 = ax.twinx()
    ax2.set_ylabel("Diameter [mm]")
    ax2.set_yscale("log")
    for i in range(len(t)):
        ax2.plot(
            t[i], y2[i], label="Colony Size", linestyle="--", color=COLOR5, linewidth=2
        )

    # Plot Fractal Dimension
    for i in range(len(t)):
        ax.plot(t[i], y1[i], label="dim", color=COLOR1)
        ax.fill_between(
            t[i], y1[i] - y1_err[i], y1[i] + y1_err[i], color=COLOR3, alpha=0.3
        )
        ind = int(np.round(0.2 * len(t[i])))
        angle = (
            360
            / (2 * np.pi)
            * np.atan(
                (y1[i][ind + 1] - y1[i][ind])
                / (np.max(y1) - np.min(y1))
                / (t[i][ind + 1] - t[i][ind])
                * (np.max(t) - np.min(t))
            )
        )
        y = y1[i][ind] + 0.1 * (np.max(y1[i]) - np.min(y1[i]))
        ax.text(t[i][ind], y, f"D={diffusion_constants[i]}", rotation=angle)
    ax.set_ylabel("Fractal Dimension")
    ax.set_xlabel("Time [min]")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = [handles1[0], handles2[0]]
    labels = [labels1[0], labels2[0]]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        frameon=False,
    )

    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(options.storage_location / "fractal-dim-over-time.pdf")
    plt.close(fig)


def fractal_dim_comparison():
    # Initialize Graph
    fig, ax = plt.subplots(figsize=(8, 8))

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf

    options = produce_options()
    diffusion_constants = [80, 5, 0.5]

    results = []
    for diffusion_constant in diffusion_constants:
        options.domain.diffusion_constant = diffusion_constant

        cells, out_path = crb.sim_branching.load_or_compute_last_iter(options)
        last_pos = np.array([c[0].mechanics.pos for c in cells.values()])

        x, y, popt, _ = crb.sim_branching.calculate_fractal_dim_for_pos(
            last_pos, options, out_path
        )

        results.append((x, y, popt))
        xmin = min(np.min(x), xmin)
        xmax = max(np.max(x), xmax)
        ymin = min(np.min(y), ymin)
        ymax = max(np.max(y), ymax)

    for (x, y, popt), diffusion_constant in zip(results, diffusion_constants):
        ax.plot(x, y, color=COLOR1, linestyle="-", label=f"D={diffusion_constant:2}")

        a, b = popt
        ax.plot(
            x,
            np.exp(a * np.log(x) + b),
            label="LR",
            color=COLOR5,
            linestyle=(0, (6, 4)),
            linewidth=2,
        )
        r = np.atan(-a / np.abs(np.log(ymax / ymin)) * np.abs(np.log(xmax / xmin)))
        r *= 360 / (2 * np.pi)
        ax.text(
            np.exp(0.50 * (np.log(xmin) + np.log(xmax))),
            np.exp(np.log(np.min(y)) + 0.55 * (np.log(np.max(y)) - np.log(np.min(y)))),
            f"D={diffusion_constant} dim={-a:.3}",
            verticalalignment="center",
            horizontalalignment="center",
            rotation=-r,
        )

    ax.vlines(
        2 * options.bacteria.cell_radius,
        ymin,
        ymax,
        color=COLOR2,
        linestyle="--",
        label="2x Radius",
    )

    ax.legend()
    ax.set_xlabel("Voxel Size [µm]")
    ax.set_ylabel("Count")
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((xmin, xmax))
    ax.set_xscale("log")
    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[1], handles[-1]]
    labels = [
        "Data",
        labels[1],
        labels[-1],
    ]

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        frameon=False,
    )
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(options.storage_location / "fractal-dim-box-size-scaling.pdf")
    plt.close(fig)


def fractal_dim_main():
    plt.rcParams.update(
        {
            "font.family": "Courier New",  # monospace font
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.titlesize": 20,
        }
    )
    fractal_dim_comparison()
    fractal_dim_over_time()
