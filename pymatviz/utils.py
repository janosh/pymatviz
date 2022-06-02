from __future__ import annotations

import ast
import subprocess
from os.path import abspath, dirname
from shutil import which
from typing import Any, Literal, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from numpy.typing import NDArray
from plotly.graph_objs._figure import Figure
from sklearn.metrics import r2_score


ROOT = dirname(dirname(abspath(__file__)))

NumArray = NDArray[Union[np.float64, np.int_]]

df_ptable = pd.read_csv(f"{ROOT}/pymatviz/elements.csv", comment="#").set_index(
    "symbol"
)

# http://jmol.sourceforge.net/jscolors
jmol_colors = df_ptable.jmol_color.dropna().apply(ast.literal_eval)

missing_cov_rad = 0.2
# covalent_radii = df_ptable.covalent_radius.fillna(missing_cov_rad).to_dict()
covalent_radii = df_ptable.covalent_radius.fillna(missing_cov_rad)

atomic_numbers: dict[str, int] = {}
element_symbols: dict[int, str] = {}

for Z, symbol in enumerate(df_ptable.index, 1):
    atomic_numbers[symbol] = Z
    element_symbols[Z] = symbol


def elem_to_Z(elem_sumbol: str) -> int:
    """Map element symbol to atomic number."""
    return atomic_numbers[elem_sumbol]


def Z_to_elem(atom_num: int) -> str:
    """Map atomic number to element symbol."""
    return element_symbols[atom_num]


def with_hist(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    cell: GridSpec = None,
    bins: int = 100,
) -> Axes:
    """Call before creating a plot and use the returned `ax_main` for all
    subsequent plotting ops to create a grid of plots with the main plot in
    the lower left and narrow histograms along its x- and/or y-axes displayed
    above and near the right edge.

    Args:
        xs (array): x values.
        ys (array): y values.
        cell (GridSpec, optional): Cell of a plt GridSpec at which to add the
            grid of plots. Defaults to None.
        bins (int, optional): Resolution/bin count of the histograms. Defaults to 100.

    Returns:
        ax: The matplotlib Axes to be used for the main plot.
    """
    fig = plt.gcf()

    gs = (cell.subgridspec if cell else fig.add_gridspec)(
        2, 2, width_ratios=(6, 1), height_ratios=(1, 5), wspace=0, hspace=0
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # x_hist
    ax_histx.hist(xs, bins=bins, rwidth=0.8)
    ax_histx.axis("off")

    # y_hist
    ax_histy.hist(ys, bins=bins, rwidth=0.8, orientation="horizontal")
    ax_histy.axis("off")

    return ax_main


def annotate_bars(
    ax: Axes = None,
    v_offset: int = 10,
    h_offset: int = 0,
    labels: Sequence[str | int | float] = None,
    fontsize: int = 14,
    **kwargs: Any,
) -> None:
    """Annotate each bar in bar plot with a label.

    Args:
        ax (matplotlib.axes.Axes): The axes to annotate.
        v_offset (int): Vertical offset between the labels and the bars.
        h_offset (int): Horizontal offset between the labels and the bars.
        labels (list[str]): Labels used for annotating bars. If not provided, defaults
            to the y-value of each bar.
        fontsize (int): Annotated text size in pts. Defaults to 14.
        **kwargs: Additional arguments (rotation, arrowprops, etc.) are passed to
            ax.annotate().
    """
    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = [int(patch.get_height()) for patch in ax.patches]

    y_max = 0

    for rect, label in zip(ax.patches, labels):

        y_pos = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2 + h_offset

        if ax.get_yscale() == "log":
            y_pos = y_pos + np.log(v_offset if v_offset > 1 else 1)
        else:
            y_pos = y_pos + v_offset

        y_max = max(y_max, y_pos)

        txt = f"{label:,}" if isinstance(label, (int, float)) else label
        # place label at end of the bar and center horizontally
        ax.annotate(txt, (x_pos, y_pos), ha="center", fontsize=fontsize, **kwargs)

    # ensure enough vertical space to display label above highest bar
    ax.set(ylim=(None, y_max * 1.1))


def add_mae_r2_box(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    ax: Axes = None,
    loc: str = "lower right",
    prefix: str = "",
    suffix: str = "",
    prec: int = 3,
    **kwargs: Any,
) -> AnchoredText:
    """Provide a set of x and y values of equal length and an optional Axes object
    on which to print the values' mean absolute error and R^2 coefficient of
    determination.

    Args:
        xs (array, optional): x values.
        ys (array, optional): y values.
        ax (Axes, optional): matplotlib Axes on which to add the box. Defaults to None.
        loc (str, optional): Where on the plot to place the AnchoredText object.
            Defaults to "lower right".
        prec (int, optional): # of decimal places in printed metrics. Defaults to 3.
        prefix (str, optional): Title or other string to prepend to metrics.
            Defaults to "".
        suffix (str, optional): Text to append after metrics. Defaults to "".

    Returns:
        AnchoredText: Instance containing the metrics.
    """
    if ax is None:
        ax = plt.gca()

    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]

    text = f"{prefix}$\\mathrm{{MAE}} = {np.abs(xs - ys).mean():.{prec}f}$"
    text += f"\n$R^2 = {r2_score(xs, ys):.{prec}f}${suffix}"

    frameon: bool = kwargs.pop("frameon", False)
    text_box = AnchoredText(text, loc=loc, frameon=frameon, **kwargs)
    ax.add_artist(text_box)

    return text_box


def get_crystal_sys(
    spg: int,
) -> Literal[
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]:
    """Get the crystal system for an international space group number."""
    # not using isinstance(n, int) to allow 0-decimal floats
    if not (spg == int(spg) and 0 < spg < 231):
        raise ValueError(f"Invalid space group {spg}")

    if 0 < spg < 3:
        return "triclinic"
    if spg < 16:
        return "monoclinic"
    if spg < 75:
        return "orthorhombic"
    if spg < 143:
        return "tetragonal"
    if spg < 168:
        return "trigonal"
    if spg < 195:
        return "hexagonal"
    return "cubic"


def add_identity_line(
    fig: Figure, trace_idx: int = 0, line_kwds: dict[str, Any] = None
) -> Figure:
    """Add a line shape to the background layer of a plotly figure spanning from
    smallest to largest x/y values in the trace specified by trace_idx.

    Args:
        fig (Figure): Plotly figure.
        trace_idx (int, optional): Index of the trace to use for measuring x/y limits.
            Defaults to 0.
        line_kwds (dict[str, Any], optional): Keyword arguments passed to
            fig.add_shape(line=line_kwds). Defaults to dict(color="gray", width=1,
            dash="dash").

    Returns:
        Figure: Figure with added identity line.
    """
    trace = fig.data[trace_idx]

    # min/max of sequence containing nans = nan so get rid of them first
    df = pd.DataFrame({"x": trace.x, "y": trace.y}).dropna()

    xy_min = min(df.min())
    xy_max = max(df.max())

    fig.add_shape(
        type="line",
        **dict(x0=xy_min, y0=xy_min, x1=xy_max, y1=xy_max),
        layer="below",
        line={**dict(color="gray", width=1, dash="dash"), **(line_kwds or {})},
    )

    return fig


def save_and_compress_svg(filename: str, fig: Figure | None = None) -> None:
    """Save Plotly figure as SVG and HTML to assets/ folder. Compresses SVG file with
    svgo CLI if available in PATH.

    Args:
        fig (Figure): Plotly Figure instance.
        filename (str): Name of SVG file (w/o extension).

    Raises:
        ValueError: If fig is None and plt.gcf() is empty.
    """
    assert not filename.endswith(".svg"), f"{filename = } should not include .svg"
    filepath = f"{ROOT}/assets/{filename}.svg"

    if isinstance(fig, Figure):
        fig.write_image(filepath)
    elif fig is None:
        if len(plt.gcf().axes) == 0:
            raise ValueError(
                "No figure passed explicitly and plt.gcf() contains no axes. "
                "Did you forget to pass a plotly figure instance?"
            )
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
    else:
        raise TypeError(f"{fig = } should be a Plotly Figure or Matplotlib Figure")

    if (svgo := which("svgo")) is not None:
        subprocess.run([svgo, "--multipass", filepath])
