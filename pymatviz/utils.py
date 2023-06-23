from __future__ import annotations

import ast
import os
import subprocess
from os.path import dirname
from shutil import which
from time import sleep
from typing import TYPE_CHECKING, Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score


if TYPE_CHECKING:
    from matplotlib.gridspec import GridSpec
    from numpy.typing import ArrayLike

ROOT = dirname(dirname(__file__))


df_ptable = pd.read_csv(f"{ROOT}/pymatviz/elements.csv", comment="#").set_index(
    "symbol"
)

# http://jmol.sourceforge.net/jscolors
jmol_colors = df_ptable.jmol_color.dropna().map(ast.literal_eval)

# fallback value (in nanometers) for covalent radius of an element
# see https://wikipedia.org/wiki/Atomic_radii_of_the_elements
missing_covalent_radius = 0.2
covalent_radii: pd.Series = df_ptable.covalent_radius.fillna(missing_covalent_radius)

atomic_numbers: dict[str, int] = {}
element_symbols: dict[int, str] = {}

for Z, symbol in enumerate(df_ptable.index, 1):
    atomic_numbers[symbol] = Z
    element_symbols[Z] = symbol


def with_hist(
    xs: ArrayLike,
    ys: ArrayLike,
    cell: GridSpec | None = None,
    bins: int = 100,
) -> plt.Axes:
    """Call before creating a plot and use the returned `ax_main` for all
    subsequent plotting ops to create a grid of plots with the main plot in the
    lower left and narrow histograms along its x- and/or y-axes displayed above
    and near the right edge.

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
    ax: plt.Axes | None = None,
    v_offset: int | float = 10,
    h_offset: int | float = 0,
    labels: Sequence[str | int | float] | None = None,
    fontsize: int = 14,
    y_max_headroom: float = 1.2,
    **kwargs: Any,
) -> None:
    """Annotate each bar in bar plot with a label.

    Args:
        ax (Axes): The matplotlib axes to annotate.
        v_offset (int): Vertical offset between the labels and the bars.
        h_offset (int): Horizontal offset between the labels and the bars.
        labels (list[str]): Labels used for annotating bars. If not provided, defaults
            to the y-value of each bar.
        fontsize (int): Annotated text size in pts. Defaults to 14.
        y_max_headroom (float): Will be multiplied with the y-value of the tallest bar
            to increase the y-max of the plot, thereby making room for text above all
            bars. Defaults to 1.2.
        **kwargs: Additional arguments (rotation, arrowprops, etc.) are passed to
            ax.annotate().
    """
    ax = ax or plt.gca()

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
    ax.set(ylim=(None, y_max * y_max_headroom))


def annotate_metrics(
    xs: ArrayLike,
    ys: ArrayLike,
    ax: plt.Axes | None = None,
    metrics: dict[str, float] | Sequence[str] = ("MAE", "$R^2$"),
    prefix: str = "",
    suffix: str = "",
    prec: int = 3,
    **kwargs: Any,
) -> AnchoredText:
    """Provide a set of x and y values of equal length and an optional Axes
    object on which to print the values' mean absolute error and R^2
    coefficient of determination.

    Args:
        xs (array, optional): x values.
        ys (array, optional): y values.
        metrics (dict[str, float] | list[str], optional): Metrics to show. Can be a
            subset of recognized keys MAE, R2, R2_adj, RMSE, MSE, MAPE or the names of
            sklearn.metrics.regression functions or any dict of metric names and values.
            Defaults to ("MAE", "R2").
        ax (Axes, optional): matplotlib Axes on which to add the box. Defaults to None.
        loc (str, optional): Where on the plot to place the AnchoredText object.
            Defaults to "lower right".
        prec (int, optional): Precision, i.e. decimal places to show in printed metrics.
            Defaults to 3.
        prefix (str, optional): Title or other string to prepend to metrics.
            Defaults to "".
        suffix (str, optional): Text to append after metrics. Defaults to "".
        **kwargs: Additional arguments (rotation, arrowprops, frameon, loc, etc.) are
            passed to matplotlib.offsetbox.AnchoredText. Sets default loc="lower right"
            and frameon=False.

    Returns:
        AnchoredText: Instance containing the metrics.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(metrics, (dict, list, tuple, set)):
        raise TypeError(f"metrics must be dict|list|tuple|set, not {type(metrics)}")
    funcs = {
        "MAE": lambda x, y: np.abs(x - y).mean(),
        "RMSE": lambda x, y: (((x - y) ** 2).mean()) ** 0.5,
        "MSE": lambda x, y: ((x - y) ** 2).mean(),
        "MAPE": mape,
        "R2": r2_score,
        "$R^2$": r2_score,
        # TODO: check this for correctness
        "R2_adj": lambda x, y: 1 - (1 - r2_score(x, y)) * (len(x) - 1) / (len(x) - 2),
    }
    for key in set(metrics) - set(funcs):
        func = getattr(sklearn.metrics, key, None)
        if func:
            funcs[key] = func
    if bad_keys := set(metrics) - set(funcs):
        raise ValueError(f"Unrecognized metrics: {bad_keys}")

    ax = ax or plt.gca()
    nans = np.isnan(xs) | np.isnan(ys)
    xs, ys = xs[~nans], ys[~nans]

    text = prefix
    if isinstance(metrics, dict):
        for key, val in metrics.items():
            text += f"{key} = {val:.{prec}f}\n"
    else:
        for metric in metrics:
            text += f"{metric} = {funcs[metric](xs, ys):.{prec}f}\n"
    text += suffix

    kwargs["frameon"] = kwargs.get("frameon", False)  # default to no frame
    kwargs["loc"] = kwargs.get("loc", "lower right")  # default to lower right
    text_box = AnchoredText(text, **kwargs)
    ax.add_artist(text_box)

    return text_box


CrystalSystem = Literal[
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]


def get_crystal_sys(spg: int) -> CrystalSystem:
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
    fig: go.Figure, line_kwds: dict[str, Any] | None = None, trace_idx: int = 0
) -> go.Figure:
    """Add a line shape to the background layer of a plotly figure spanning
    from smallest to largest x/y values in the trace specified by trace_idx.

    Args:
        fig (Figure): Plotly figure.
        line_kwds (dict[str, Any], optional): Keyword arguments for customizing the line
            shape will be passed to fig.add_shape(line=line_kwds). Defaults to
            dict(color="gray", width=1, dash="dash").
        trace_idx (int, optional): Index of the trace to use for measuring x/y limits.
            Defaults to 0. Unused if kaleido package is installed and the figure's
            actual x/y-range can be obtained from fig.full_figure_for_development().

    Returns:
        Figure: Figure with added identity line.
    """
    # If kaleido is missing, try block raises ValueError: Full figure generation
    # requires the kaleido package. Install with: pip install kaleido
    # If so, we resort to manually computing the xy data ranges which are usually are
    # close to but not the same as the axes limits.
    try:
        # https://stackoverflow.com/a/62042077
        full_fig = fig.full_figure_for_development(warn=False)
        xy_range = full_fig.layout.xaxis.range + full_fig.layout.yaxis.range
        xy_min, xy_max = min(xy_range), max(xy_range)
    except ValueError:
        trace = fig.data[trace_idx]

        # min/max(seq) gives NaN if sequence contains NaNs so get rid of them first
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


def save_fig(
    fig: go.Figure | plt.Figure | plt.Axes,
    path: str,
    plotly_config: dict[str, Any] | None = None,
    env_disable: Sequence[str] = ("CI",),
    pdf_sleep: float = 0.6,
    **kwargs: Any,
) -> None:
    """Write a plotly figure to an HTML file. If the file is has .svelte
    extension, insert `{...$$props}` into the figure's top-level div so it can
    be styled by consuming Svelte code.

    Args:
        fig (go.Figure | plt.Figure | plt.Axes): Plotly or matplotlib Figure or
            matplotlib Axes object.
        path (str): Path to HTML file that will be created.
        plotly_config (dict, optional): Configuration options for fig.write_html().
        Defaults to dict(showTips=False, responsive=True, modeBarButtonsToRemove=
        ["lasso2d", "select2d", "autoScale2d", "toImage"]).
        See https://plotly.com/python/configuration-options.
        env_disable (list[str], optional): Do nothing if any of these environment
            variables are set. Defaults to ("CI",).
        pdf_sleep (float, optional): Minimum time in seconds to wait before
            writing a PDF file. Workaround for this plotly issue
            https://github.com/plotly/plotly.py/issues/3469. Defaults to 0.6. Has no
            effect on matplotlib figures.

        **kwargs: Keyword arguments passed to fig.write_html().
    """
    if any(var in os.environ for var in env_disable):
        return
    # handle matplotlib figures
    if isinstance(fig, (plt.Figure, plt.Axes)):
        if hasattr(fig, "figure"):
            fig = fig.figure  # unwrap Axes
        fig.savefig(path, **kwargs)
        return
    if not isinstance(fig, go.Figure):
        raise TypeError(
            f"Unsupported figure type {type(fig)}, expected plotly or matplotlib Figure"
        )
    is_pdf = path.lower().endswith((".pdf", ".pdfa"))
    if path.lower().endswith((".svelte", ".html")):
        config = dict(
            showTips=False,
            modeBarButtonsToRemove=[
                "lasso2d",
                "select2d",
                "autoScale2d",
                "toImage",
                "toggleSpikelines",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
            ],
            responsive=True,
            displaylogo=False,
        )
        config.update(plotly_config or {})
        defaults = dict(include_plotlyjs=False, full_html=False, config=config)
        defaults.update(kwargs)
        fig.write_html(path, **defaults)
        if path.lower().endswith(".svelte"):
            # insert {...$$props} into top-level div to be able to post-process and
            # style plotly figures from within Svelte files
            with open(path) as file:
                text = file.read().replace("<div>", "<div {...$$props}>", 1)
            with open(path, "w") as file:
                file.write(text + "\n")
    else:
        if is_pdf:
            orig_template = fig.layout.template
            fig.layout.template = "plotly_white"
        # hide click-to-show traces in PDF
        hidden_traces = []
        for trace in fig.data:
            if trace.visible == "legendonly":
                trace.visible = False
                hidden_traces.append(trace)
        fig.write_image(path, **kwargs)
        if is_pdf:
            # write PDFs twice to get rid of "Loading [MathJax]/extensions/MathMenu.js"
            # see https://github.com/plotly/plotly.py/issues/3469#issuecomment-994907721
            sleep(pdf_sleep)
            fig.write_image(path, **kwargs)

            fig.layout.template = orig_template
        for trace in hidden_traces:
            trace.visible = "legendonly"


def save_and_compress_svg(
    fig: go.Figure | plt.Figure | plt.Axes, filename: str
) -> None:
    """Save Plotly figure as SVG and HTML to assets/ folder. Compresses SVG
    file with svgo CLI if available in PATH.

    Args:
        fig (Figure): Plotly or matplotlib Figure/Axes instance.
        filename (str): Name of SVG file (w/o extension).

    Raises:
        ValueError: If fig is None and plt.gcf() is empty.
    """
    assert not filename.endswith(".svg"), f"{filename = } should not include .svg"
    filepath = f"{ROOT}/assets/{filename}.svg"
    if isinstance(fig, plt.Axes):
        fig = fig.figure

    if isinstance(fig, plt.Figure) and not fig.axes:
        raise ValueError("Passed fig contains no axes. Nothing to plot!")
    save_fig(fig, filepath)
    plt.close()

    if (svgo := which("svgo")) is not None:
        subprocess.run([svgo, "--multipass", filepath])


def df_to_arrays(
    df: pd.DataFrame | None,
    *args: str | Sequence[str] | ArrayLike,
    strict: bool = True,
) -> list[ArrayLike | dict[str, ArrayLike]]:
    """If df is None, this is a no-op: args are returned as-is. If df is a
    dataframe, all following args are used as column names and the column data
    returned as arrays (after dropping rows with NaNs in any column).

    Args:
        df (pd.DataFrame | None): Optional pandas DataFrame.
        *args (list[ArrayLike | str]): Arbitrary number of arrays or column names in df.
        strict (bool, optional): If True, raise TypeError if df is not pd.DataFrame
            or None. If False, return args as-is. Defaults to True.

    Raises:
        ValueError: If df is not None and any of the args is not a df column name.
        TypeError: If df is not pd.DataFrame and not None.

    Returns:
        tuple[ArrayLike, ArrayLike]: Input arrays or arrays from dataframe columns.
    """
    if df is None:
        if cols := [arg for arg in args if isinstance(arg, str)]:
            raise ValueError(f"got column names but no df to get data from: {cols}")
        return args  # type: ignore[return-value]

    if not isinstance(df, pd.DataFrame):
        if not strict:
            return args  # type: ignore[return-value]
        raise TypeError(f"df should be pandas DataFrame or None, got {type(df)}")

    if arrays := [arg for arg in args if isinstance(arg, np.ndarray)]:
        raise ValueError(
            "don't pass dataframe and arrays to df_to_arrays(), should be either or, "
            f"got {arrays}"
        )

    flat_args = []
    # tuple doesn't support item assignment
    args = list(args)  # type: ignore[assignment]

    for col_name in args:
        if isinstance(col_name, (str, int)):
            flat_args.append(col_name)
        else:
            flat_args.extend(col_name)

    df_no_nan = df.dropna(subset=flat_args)
    for idx, col_name in enumerate(args):
        if isinstance(col_name, (str, int)):
            args[idx] = df_no_nan[col_name].to_numpy()  # type: ignore[index]
        else:
            col_data = df_no_nan[[*col_name]].to_numpy().T
            args[idx] = dict(zip(col_name, col_data))  # type: ignore[index]

    return args  # type: ignore[return-value]


def bin_df_cols(
    df: pd.DataFrame,
    bin_by_cols: Sequence[str],
    group_by_cols: Sequence[str] = (),
    n_bins: int | Sequence[int] = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Bin columns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to bin.
        bin_by_cols (Sequence[str]): Columns to bin.
        group_by_cols (Sequence[str]): Additional columns to group by. Defaults to ().
        n_bins (int): Number of bins to use. Defaults to 100.
        verbose (bool): If True, report df length reduction. Defaults to True.

    Returns:
        pd.DataFrame: Binned DataFrame.
    """
    if isinstance(n_bins, int):
        n_bins = [n_bins] * len(bin_by_cols)

    if len(bin_by_cols) != len(n_bins):
        raise ValueError(f"{len(bin_by_cols)=} != {len(n_bins)=}")

    index_name = df.index.name

    for col, bins in zip(bin_by_cols, n_bins):
        df[f"{col}_bins"] = pd.cut(df[col], bins=bins)

    df_bin = (
        df.reset_index()
        .groupby([*[f"{c}_bins" for c in bin_by_cols], *group_by_cols])
        .first()
        .dropna()
    )
    if verbose:
        print(f"{len(df_bin)=:,} / {len(df)=:,} = {len(df_bin)/len(df):.1%}")

    if index_name is None:
        return df_bin
    return df_bin.reset_index().set_index(index_name)
