import numpy as np
import cr_bayesian_optim as crb
from cr_bayesian_optim.plotting import COLOR1, COLOR2, COLOR3, COLOR4, COLOR5
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sp
from glob import glob
import os


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

    diffusion_constants = [80, 5, 0.5]
    for diffusion_constant in diffusion_constants:
        options.domain.diffusion_constant = diffusion_constant
        cells, _ = crb.sim_branching.load_or_compute_full(options)

        iterations = sorted(cells.keys())
        dims_mean = []
        dims_std = []
        for i in tqdm(iterations, desc="Calculating dim(t)"):
            pos = np.array([c[0].mechanics.pos for c in cells[i].values()])

            _, _, popt, pcov = crb.sim_branching.calculate_fractal_dim_for_pos(
                pos, options, None
            )
            dims_mean.append(-popt[0])
            dims_std.append(pcov[0, 0] ** 0.5)

        t.append(np.array(iterations) * options.time.dt / 60)
        y1.append(np.array(dims_mean))
        y1_err.append(np.array(dims_std))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Fractal Dimension
    for i in range(len(t)):
        ax.plot(t[i], y1[i], label="dim", color=COLOR1)
        ax.fill_between(
            t[i], y1[i] - y1_err[i], y1[i] + y1_err[i], color=COLOR3, alpha=0.3
        )

        # Plot Fit
        popt, pcov = sp.optimize.curve_fit(
            lambda t, a, b, c: (a - c) * (1 - np.exp(-b * t)) + c,
            t[i],
            y1[i],
            sigma=y1_err[i],
            absolute_sigma=True,
        )

        a, b, c = popt
        yfit = (a - c) * (1 - np.exp(-b * t[i])) + c
        ax.plot(
            t[i],
            yfit,
            label="BG",
            color=COLOR5,
            linestyle="--",
            linewidth=2,
        )

        ind = int(np.round(0.3 * len(t[i])))
        angle = (
            360
            / (2 * np.pi)
            * np.atan(
                (yfit[ind + 1] - yfit[ind])
                / (np.max(y1) - np.min(y1))
                / (t[i][ind + 1] - t[i][ind])
                * (np.max(t) - np.min(t))
            )
        )
        y = yfit[ind] + 0.15 * (np.max(yfit) - np.min(yfit))
        ax.text(
            t[i][ind],
            y,
            f"D={diffusion_constants[i]} dim$\\rightarrow${a:.4}",
            rotation=angle,
            horizontalalignment="center",
            verticalalignment="center",
        )

    ax.set_xlim(np.min(t[0]), np.max(t[0]))
    ax.set_ylabel("Fractal Dimension")
    ax.set_xlabel("Time [min]")

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[1]]
    labels = [labels[0], labels[1]]
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
            f"D={diffusion_constant} dim={-a:.4}",
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


def runtime_plot():
    options = produce_options()
    diffusion_constants = [80, 5, 0.5]

    fig, ax = plt.subplots(figsize=(8, 8))

    for diffusion_constant in diffusion_constants:
        options.domain.diffusion_constant = diffusion_constant
        cells, out_path = crb.sim_branching.load_or_compute_full(options)

        files = sorted(glob(str(out_path / "cells/json/*/*.json")))
        times = np.array([os.path.getmtime(f) for f in files])

        t = np.array(list(cells.keys())) * options.time.dt / 60
        dt = times - times[0]

        # Plot Data
        ax.plot(t, dt, color=crb.plotting.COLOR1, label="Data")

        ind = int(np.round(0.5 * len(t)))
        tfit = t[ind:]
        popt, pcov = sp.optimize.curve_fit(lambda t, a, b: a * t + b, tfit, dt[ind:])

        a, b = popt
        # da = pcov[0, 0] ** 0.5
        # db = pcov[1, 1] ** 0.5
        yfit = a * tfit + b
        # yfit_low = (a - da) * tfit + (b - db)
        # yfit_high = (a + da) * tfit + (b + db)

        # Plot Fit
        ax.plot(
            tfit,
            yfit,
            label="LR",
            color=COLOR5,
            linestyle=(0, (6, 4)),
            linewidth=2,
        )
        ax.set_xlim(np.min(t).astype(float), np.max(t).astype(float))
        # ax.fill_between(tfit, yfit_low, yfit_high, color=COLOR4, alpha=0.3)

    ax.set_ylabel("Runtime [s]")
    ax.set_xlabel("Simulation Time [min]")
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)

    ax.legend()

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[1]]
    labels = [labels[0], labels[1]]

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(options.storage_location / "runtime-sim-branching.pdf")
    plt.close(fig)


def fractal_dim_vs_diffusion_constant():
    options = produce_options()

    fig, ax = plt.subplots(figsize=(8, 8))

    data = []
    data_err = []
    diffusion_constants = [0.1, 0.5, 2.0, 5.0, 80.0]
    for diffusion_constant in diffusion_constants:
        options.domain.diffusion_constant = diffusion_constant

        cells, out_path = crb.sim_branching.load_or_compute_last_iter(options)
        pos = np.array([c[0].mechanics.pos for c in cells.values()])
        _, _, popt, pcov = crb.sim_branching.calculate_fractal_dim_for_pos(pos, options)
        dim = popt[0]
        ddim = pcov[0, 0] ** 0.5

        data.append(-dim)
        data_err.append(ddim)

    ax.plot(diffusion_constants, data, color=COLOR1, label="dim")
    ax.fill_between(
        diffusion_constants,
        np.array(data) - np.array(data_err),
        np.array(data) + np.array(data_err),
        color=COLOR2,
        alpha=0.3,
    )

    ax.set_xlim(np.min(diffusion_constants), np.max(diffusion_constants))
    ax.set_xlabel("Diffusion Constant")
    ax.set_ylabel("Fractal Dimension")
    ax.set_xscale("log")
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(options.storage_location / "fractal-dim-vs-diffusion-constant.pdf")
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
    runtime_plot()
    fractal_dim_vs_diffusion_constant()
