import numpy as np
import cr_bayesian_optim as crb
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import scipy as sp
from glob import glob
from pathlib import Path


# Define colors
COLOR1 = "#0c457d"
COLOR2 = "#0ea7b5"
COLOR3 = "#6bd2db"
COLOR4 = "#ffbe4f"
COLOR5 = "#e8702a"

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


def calculate_fields(
    positions: np.ndarray,
    n_voxels: int,
    options: crb.Options,
    raw: bool = False,
):
    radius = options.bacteria.cell_radius / options.domain.domain_size
    voxels = positions * n_voxels / options.domain.domain_size
    # Calculate padding
    dist = 2 * n_voxels * radius

    y = np.zeros((n_voxels, n_voxels))
    for vox in voxels:
        xmin = max(vox[0] - dist, 0)
        xmax = min(vox[0] + dist + 1, n_voxels)
        ymin = max(vox[1] - dist, 0)
        ymax = min(vox[1] + dist + 1, n_voxels)
        for i in range(int(xmin), np.ceil(xmax).astype(int)):
            for j in range(int(ymin), np.ceil(ymax).astype(int)):
                if i == int(vox[0]) and j == int(vox[1]):
                    y[i, j] += 1
                elif ((i - vox[0]) ** 2 + (j - vox[1]) ** 2) ** 0.5 <= dist:
                    y[i, j] += 1
    if raw:
        return y
    else:
        return y > 0


def plot_discretizations(last_pos, n_voxels_list, options, out_path: Path):
    # Plot Snapshots of Discretization to calculate Fractal Dimension
    n = len(n_voxels_list)
    indices = np.round(np.linspace(0, n - 1, min(10, n), endpoint=True)).astype(int)
    counts = []
    for n_voxels in tqdm(n_voxels_list[indices], desc="Plotting Discretizations"):
        y = calculate_fields(
            last_pos,
            n_voxels,
            options,
            raw=True,
        )
        counts.append((n_voxels, y))

    max_overall = np.max([np.max(c[1]) for c in counts])
    for n_voxels, y in counts:
        img = Image.fromarray(
            np.round(y / max_overall * 125 + (y > 0) * 130).astype(np.uint8)
        )
        img.save(out_path / f"discretization-nvoxels-{n_voxels:06}.png")


def load_or_compute(options):
    for file in glob(str(options.storage_location / "**/options.toml")):
        file_path = Path(file)
        opt_loaded = crb.Options.load_from_toml(file_path)
        if opt_loaded == options:
            print("Reading Files")
            out_path = file_path.parent
            last_iter = crb.get_all_iterations(out_path)[-1]
            return crb.load_results_at_iteration(out_path, last_iter), out_path
    else:
        print("Running Simulation")
        cells, out_path = crb.run_sim_branching(options)
        last_iter = sorted(cells.keys())[-1]
        return cells[last_iter], out_path


def calculate_fractal_dim_over_time(cells: crb.CellOutput, options: crb.Options):
    iterations = sorted(cells.keys())
    dims = []
    for i in iterations:
        pos = np.array([c[0].mechanics.pos for c in cells[i].values()])

        x, popt, pcov, count_boxes = calculate_fractal_dim_for_pos(pos, options, None)
        dims.append(popt[0])


def calculate_fractal_dim_for_pos(
    pos, options: crb.Options, out_path: Path | None = None
):
    x = np.linspace(
        options.bacteria.cell_radius / 2.0,
        options.domain.domain_size / 10,
        10,
        dtype=float,
    )
    n_voxels_list = np.floor(options.domain.domain_size / x).astype(int)
    fields = [calculate_fields(pos, n, options) for n in n_voxels_list]
    count_boxes = np.array([np.sum(yi > 0) for yi in fields]).astype(int)

    if out_path is not None:
        plot_discretizations(pos, n_voxels_list, options, out_path)

    popt, pcov = sp.optimize.curve_fit(
        lambda x, a, b: a * x + b, np.log(x), np.log(count_boxes)
    )
    return x, count_boxes, popt, pcov


def fractal_dim_main():
    # Initialize Graph
    fig, ax = plt.subplots(figsize=(8, 8))

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf

    options = crb.Options(
        show_progressbar=True, storage_location="out/fractal_dim_multi"
    )
    options.time.t_max = 2000
    options.domain.domain_size = 2000
    options.time.dt = 0.3
    diffusion_constants = [80, 5, 0.5]

    results = []
    for diffusion_constant in diffusion_constants:
        options.domain.diffusion_constant = diffusion_constant

        cells, out_path = load_or_compute(options)
        last_pos = np.array([c[0].mechanics.pos for c in cells.values()])

        x, y, popt, _ = calculate_fractal_dim_for_pos(last_pos, options, out_path)

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
            linestyle="--",
            linewidth=1.5,
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
        "D=" + ",".join([str(i) for i in sorted(diffusion_constants)]),
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
    fig.savefig(options.storage_location / "fractal-dimension.pdf")
