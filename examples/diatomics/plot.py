"""Plot MLIP pair repulsion curves in a periodic table layout.

All credit for this code to Tamas Stenczel. Authored in https://github.com/stenczelt/MACE-MP-work
for MACE-MP paper https://arxiv.org/abs/2401.00096
"""

# %%
import json
import lzma
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from matplotlib.backends.backend_pdf import PdfPages
from pymatgen.core import Element

import pymatviz as pmv
from pymatviz import df_ptable, ptable_lines


module_dir = os.path.dirname(__file__)
__date__ = "2024-03-31"


# %%
def plot_on_ax(
    ax: plt.Axes,
    distances: np.ndarray,
    energy: np.ndarray,
    formula: str,
) -> None:
    """Plot pair repulsion curve on a given axes.

    Args:
        ax (plt.Axes): Axes object to plot on.
        distances (np.ndarray): Array of distances.
        energy (np.ndarray): Array of energy values.
        formula (str): Chemical formula of the pair.
    """
    shift = energy[-1]
    energy -= shift

    ax.plot(distances, energy, marker=".")
    ax.axhline(0, color="tab:gray")
    ax.set_title(f"{formula}", fontsize=28)
    ax.set(ylim=(-20, 20))


def plot_homo_nuclear(model_size: str) -> None:
    """Plot homonuclear pair repulsion curves in a periodic table layout.

    Args:
        model_size (str): Size of the model (e.g., "small", "medium").
    """
    n_rows, n_columns, size_factor = 10, 18, 3

    fig = plt.figure(
        figsize=(0.75 * n_columns * size_factor, 0.7 * n_rows * size_factor),
    )
    gs = plt.GridSpec(figure=fig, nrows=n_rows, ncols=n_columns)

    lzma_path = f"{module_dir}/homo-nuclear-mace-{model_size}.json.lzma"
    with lzma.open(lzma_path, "rt") as file:
        data: dict[str, list[float]] = json.load(file)

    distances = np.array(data.pop("distances"))
    elements_ok = [int(key.split("-")[0]) for key in data]

    for symbol, row, column, _name, atomic_number, *_ in df_ptable.itertuples():
        if atomic_number in elements_ok:
            if row > 7:
                row -= 2  # noqa: PLW2901
            ax = fig.add_subplot(gs[row - 1, column - 1])
            plot_on_ax(
                ax,
                distances,
                np.array(data[f"{atomic_number}-{atomic_number}"]),
                symbol,
            )

    fig.tight_layout()
    fig.savefig(f"homo-nuclear-mace-{model_size}-periodic-table.svg")


def plot_hetero_nuclear(model_size: str) -> None:
    """Plot hetero-nuclear pair repulsion curves for each element.

    Args:
        model_size (str): Size of the model (e.g., "small", "medium").
    """
    n_rows, n_columns, size_factor = 10, 18, 3

    z_calculated = sorted(
        [
            int(fn.name.split("-")[2])
            for fn in Path("simulations/").glob(f"results-{model_size}-*-X.json")
        ],
    )
    with PdfPages(f"{model_size}-hetero-nuclear.pdf") as pdf:
        for z_main in z_calculated:
            fig = plt.figure(
                figsize=(0.75 * n_columns * size_factor, 0.7 * n_rows * size_factor),
            )
            gs = plt.GridSpec(figure=fig, nrows=n_rows, ncols=n_columns)
            plot_element_heteronuclear(fig, gs, model_size, z_main)
            fig.suptitle(
                f"{chemical_symbols[z_main]}-X Pair repulsion curves with "
                f"MACE-MP0 ({model_size})",
                fontsize=70,
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.savefig(f"heteronuclear-{model_size}-{chemical_symbols[z_main]}-X.pdf")


def plot_element_heteronuclear(
    fig: plt.Figure,
    gs: plt.GridSpec,
    model_size: str,
    atomic_number: int,
) -> None:
    """Plot heteronuclear pair repulsion curves for a specific element.

    Args:
        fig (plt.Figure): Figure object to plot on.
        gs (plt.GridSpec): GridSpec object for subplot layout.
        model_size (str): Size of the model (e.g., "small", "medium").
        atomic_number (int): Atomic number of the main element.
    """
    if gs.nrows != 10 or gs.ncols != 18:
        raise ValueError("GridSpec must have 10 rows and 18 columns.")

    with open(f"simulations/results-{model_size}-{atomic_number}-x.json") as file:
        data: dict[str, list[float]] = json.load(file)

    distances = np.array(data.pop("distances"))
    elements_ok = [int(key.split("-")[1]) for key in data]

    for symbol, row, column, _name, z_other, *_ in df_ptable.itertuples():
        if z_other in elements_ok:
            ax = fig.add_subplot(gs[row - 1, column - 1])
            plot_on_ax(
                ax,
                distances,
                np.array(data[f"{atomic_number}-{z_other}"]),
                f"{chemical_symbols[atomic_number]}-{symbol}",
            )

            if atomic_number == z_other:
                ax.patch.set_facecolor("tab:blue")
                ax.patch.set_alpha(0.3)


# %% plot homo-nuclear and heteronuclear pair repulsion curves
# model_size = "small"
model_size = "medium"
lzma_path = f"{module_dir}/homo-nuclear-mace-{model_size}.json.lzma"
with lzma.open(lzma_path, "rt") as file:
    data: dict[str, list[float]] = json.load(file)
    x_dists = data.pop("distances")

if len(x_dists) != 119:
    raise ValueError(f"Unexpected {len(x_dists)=}")

diatomic_curves = {
    Element.from_Z(int(key.split("-")[0])).symbol: [
        x_dists,
        np.array(e_pot) - e_pot[-1],  # shift energies so energy at max separation is 0
    ]
    for key, e_pot in data.items()
}
# plot_homo_nuclear("small")
# plot_hetero_nuclear("small")
# plot_homo_nuclear("medium")
# plot_hetero_nuclear("medium")

ax = ptable_lines(
    diatomic_curves,
    ax_kwargs=dict(ylim=(-10, 10), yticklabels=[]),
    child_kwargs=dict(color="darkblue", linestyle="solid"),
    color_elem_strategy="symbol",
    add_elem_type_legend=False,
)
pmv.save_fig(ax, f"{module_dir}/homo-nuclear-mace-{model_size}.svg")


# %% count number of elements with energies below E_TOO_LOW
E_TOO_LOW = -20
for model_size in ("small", "medium"):
    lzma_path = f"{module_dir}/homo-nuclear-mace-{model_size}.json.lzma"
    with lzma.open(lzma_path, "rt") as file:
        homo_nuc_diatomics = json.load(file)

    x_dists = homo_nuc_diatomics.pop("distances")
    min_energies = {
        Element.from_Z(int(key.split("-")[0])).symbol: min(y_vals)
        for key, y_vals in homo_nuc_diatomics.items()
    }
    n_lt_10 = sum(val < E_TOO_LOW for val in min_energies.values())
    print(f"{model_size=} {n_lt_10=}")
