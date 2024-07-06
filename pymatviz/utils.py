"""pymatviz utility functions."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from functools import partial, wraps
from os.path import dirname
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
from matplotlib.offsetbox import AnchoredText


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import ParamSpec, TypeVar

    from numpy.typing import ArrayLike

    P = ParamSpec("P")
    R = TypeVar("R")

PKG_DIR = dirname(__file__)
ROOT = dirname(PKG_DIR)
TEST_FILES = f"{ROOT}/tests/files"
Backend = Literal["matplotlib", "plotly"]
BACKENDS = MATPLOTLIB, PLOTLY = get_args(Backend)

AxOrFig = Union[plt.Axes, plt.Figure, go.Figure]
VALID_FIG_TYPES = get_args(AxOrFig)
VALID_FIG_NAMES = " | ".join(
    f"{t.__module__}.{t.__qualname__}" for t in VALID_FIG_TYPES
)

CrystalSystem = Literal[
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]

elements_csv = f"{ROOT}/pymatviz/elements.csv"
df_ptable = pd.read_csv(elements_csv, comment="#").set_index("symbol")
ElemValues = Union[dict[Union[str, int], float], pd.Series, Sequence[str]]

# fallback value (in nanometers) for covalent radius of an element
# see https://wikipedia.org/wiki/Atomic_radii_of_the_elements
missing_covalent_radius = 0.2
covalent_radii: pd.Series = df_ptable.covalent_radius.fillna(missing_covalent_radius)

atomic_numbers: dict[str, int] = {}
element_symbols: dict[int, str] = {}

for Z, symbol in enumerate(df_ptable.index, 1):
    atomic_numbers[symbol] = Z
    element_symbols[Z] = symbol


class ExperimentalWarning(Warning):
    """Warning for experimental features."""


warnings.simplefilter("once", ExperimentalWarning)


def pretty_label(key: str, backend: Backend) -> str:
    """Map metric keys to their pretty labels."""
    if backend not in BACKENDS:
        raise ValueError(f"Unexpected {backend=}, must be one of {BACKENDS}")

    symbol_mapping = {
        "R2": {MATPLOTLIB: "$R^2$", PLOTLY: "R<sup>2</sup>"},
        "R2_adj": {
            MATPLOTLIB: "$R^2_{adj}$",
            PLOTLY: "R<sup>2</sup><sub>adj</sub>",
        },
    }

    return symbol_mapping.get(key, {}).get(backend, key)


def crystal_sys_from_spg_num(spg: float) -> CrystalSystem:
    """Get the crystal system for an international space group number."""
    # Ensure integer or float with no decimal part
    if not isinstance(spg, (int, float)) or spg != int(spg):
        raise TypeError(f"Expect integer space group number, got {spg=}")

    if not (1 <= spg <= 230):
        raise ValueError(f"Invalid space group number {spg}, must be 1 <= num <= 230")

    if 1 <= spg <= 2:
        return "triclinic"
    if spg <= 15:
        return "monoclinic"
    if spg <= 74:
        return "orthorhombic"
    if spg <= 142:
        return "tetragonal"
    if spg <= 167:
        return "trigonal"
    if spg <= 194:
        return "hexagonal"
    return "cubic"


def df_to_arrays(
    df: pd.DataFrame | None,
    *args: str | Sequence[str] | Sequence[ArrayLike],
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
        list[ArrayLike | dict[str, ArrayLike]]: Array data for each column name or
            dictionary of column names and array data.
    """
    if df is None:
        if cols := [arg for arg in args if isinstance(arg, str)]:
            raise ValueError(f"got column names but no df to get data from: {cols}")
        # pass through args as-is
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
    df_in: pd.DataFrame,
    bin_by_cols: Sequence[str],
    *,
    group_by_cols: Sequence[str] = (),
    n_bins: int | Sequence[int] = 100,
    bin_counts_col: str = "bin_counts",
    kde_col: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Bin columns of a DataFrame.

    Args:
        df_in (pd.DataFrame): Input dataframe to bin.
        bin_by_cols (Sequence[str]): Columns to bin.
        group_by_cols (Sequence[str]): Additional columns to group by. Defaults to ().
        n_bins (int): Number of bins to use. Defaults to 100.
        bin_counts_col (str): Column name for bin counts.
            Defaults to "bin_counts".
        kde_col (str): Column name for KDE bin counts e.g. 'kde_bin_counts'. Defaults to
            "" which means no KDE to speed things up.
        verbose (bool): If True, report df length reduction. Defaults to True.

    Returns:
        pd.DataFrame: Binned DataFrame.
    """
    if isinstance(n_bins, int):
        # broadcast integer n_bins to all bin_by_cols
        n_bins = [n_bins] * len(bin_by_cols)

    if len(bin_by_cols) != len(n_bins):
        raise ValueError(f"{len(bin_by_cols)=} != {len(n_bins)=}")

    index_name = df_in.index.name

    for col, bins in zip(bin_by_cols, n_bins):
        df_in[f"{col}_bins"] = pd.cut(df_in[col].values, bins=bins)

    if df_in.index.name not in df_in:
        df_in = df_in.reset_index()

    group = df_in.groupby(
        [*[f"{c}_bins" for c in bin_by_cols], *group_by_cols], observed=True
    )

    df_bin = group.first().dropna()
    df_bin[bin_counts_col] = group.size()

    if verbose:
        print(  # noqa: T201
            f"{1 - len(df_bin) / len(df_in):.1%} sample reduction from binning: from "
            f"{len(df_in):,} to {len(df_bin):,}"
        )

    if kde_col:
        # compute kernel density estimate for each bin
        values = df_in[bin_by_cols].dropna().T
        model_kde = scipy.stats.gaussian_kde(values)

        xy_binned = df_bin[bin_by_cols].T
        density = model_kde(xy_binned)
        df_bin["cnt_col"] = density / density.sum() * len(values)

    if index_name is None:
        return df_bin
    return df_bin.reset_index().set_index(index_name)


@contextmanager
def patch_dict(
    dct: dict[Any, Any], *args: Any, **kwargs: Any
) -> Generator[dict[Any, Any], None, None]:
    """Context manager to temporarily patch the specified keys in a dictionary and
    restore it to its original state on context exit.

    Useful e.g. for temporary plotly fig.layout mutations:

        with patch_dict(fig.layout, showlegend=False):
            fig.write_image("plot.pdf")

    Args:
        dct (dict): The dictionary to be patched.
        *args: Only first element is read if present. A single dictionary containing the
            key-value pairs to patch.
        **kwargs: The key-value pairs to patch, provided as keyword arguments.

    Yields:
        dict: The patched dictionary incl. temporary updates.
    """
    # if both args and kwargs are passed, kwargs will overwrite args
    updates = {**args[0], **kwargs} if args and isinstance(args[0], dict) else kwargs

    # save original values as shallow copy for speed
    # warning: in-place changes to nested dicts and objects will persist beyond context!
    patched = dct.copy()

    # apply updates
    patched.update(updates)

    yield patched


def luminance(color: tuple[float, float, float]) -> float:
    """Compute the luminance of a color as in https://stackoverflow.com/a/596243.

    Args:
        color (tuple[float, float, float]): RGB color tuple with values in [0, 1].

    Returns:
        float: Luminance of the color.
    """
    red, green, blue, *_ = color  # alpha = 1 - transparency
    return 0.299 * red + 0.587 * green + 0.114 * blue


def pick_bw_for_contrast(
    color: tuple[float, float, float], text_color_threshold: float = 0.7
) -> str:
    """Choose black or white text color for a given background color based on luminance.

    Args:
        color (tuple[float, float, float]): RGB color tuple with values in [0, 1].
        text_color_threshold (float, optional): Luminance threshold for choosing
            black or white text color. Defaults to 0.7.

    Returns:
        str: "black" or "white" depending on the luminance of the background color.
    """
    light_bg = luminance(color) > text_color_threshold
    return "black" if light_bg else "white"


def si_fmt(
    val: float,
    *,
    fmt: str = ".1f",
    sep: str = "",
    binary: bool = False,
    decimal_threshold: float = 0.01,
) -> str:
    """Convert large numbers into human readable format using SI prefixes.

    Supports binary (1024) and metric (1000) mode.

    https://nist.gov/pml/weights-and-measures/metric-si-prefixes

    Args:
        val (int | float): Some numerical value to format.
        binary (bool, optional): If True, scaling factor is 2^10 = 1024 else 1000.
            Defaults to False.
        fmt (str): f-string format specifier. Configure precision and left/right
            padding in returned string. Defaults to ".1f". Can be used to ensure leading
            or trailing whitespace for shorter numbers. See
            https://docs.python.org/3/library/string.html#format-specification-mini-language.
        sep (str): Separator between number and postfix. Defaults to "".
        decimal_threshold (float): abs(value) below 1 but above this threshold will be
            left as decimals. Only below this threshold is a greek suffix added (milli,
            micro, etc.). Defaults to 0.01. i.e. 0.01 -> "0.01" while
            0.0099 -> "9.9m". Setting decimal_threshold=0.1 would format 0.01 as "10m"
            and leave 0.1 as is.

    Returns:
        str: Formatted number.
    """
    factor = 1024 if binary else 1000
    _scale = ""

    if abs(val) >= 1:
        # 1, Kilo, Mega, Giga, Tera, Peta, Exa, Zetta, Yotta
        for _scale in ("", "K", "M", "G", "T", "P", "E", "Z", "Y"):
            if abs(val) < factor:
                break
            val /= factor
    elif val != 0 and abs(val) < decimal_threshold:
        # milli, micro, nano, pico, femto, atto, zepto, yocto
        for _scale in ("", "m", "μ", "n", "p", "f", "a", "z", "y"):
            if abs(val) >= 1:
                break
            val *= factor

    return f"{val:{fmt}}{sep}{_scale}"


si_fmt_int = partial(si_fmt, fmt=".0f")


def styled_html_tag(text: str, tag: str = "span", style: str = "") -> str:
    """Wrap text in a span with custom style.

    Style defaults to decreased font size and weight e.g. to display units
    in plotly labels and annotations.

    Args:
        text (str): Text to wrap in span.
        tag (str, optional): HTML tag name. Defaults to "span".
        style (str, optional): CSS style string. Defaults to
            "font-size: 0.8em; font-weight: lighter;".
    """
    style = style or "font-size: 0.8em; font-weight: lighter;"
    return f"<{tag} {style=}>{text}</{tag}>"


def validate_fig(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to validate the type of fig keyword argument in a function. fig MUST be
    a keyword argument, not a positional argument.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # TODO use typing.ParamSpec to type wrapper once py310 is oldest supported
        fig = kwargs.get("fig", None)
        if fig is not None and not isinstance(fig, (plt.Axes, plt.Figure, go.Figure)):
            raise TypeError(
                f"Unexpected type for fig: {type(fig).__name__}, must be one of None, "
                f"{VALID_FIG_NAMES}"
            )
        return func(*args, **kwargs)

    return wrapper


@validate_fig
def annotate(text: str, fig: AxOrFig | None = None, **kwargs: Any) -> AxOrFig:
    """Annotate a matplotlib or plotly figure.

    Args:
        text (str): The text to use for annotation.
        fig (plt.Axes | plt.Figure | go.Figure | None, optional): The matplotlib Axes,
            Figure or plotly Figure to annotate. If None, the current matplotlib Axes
            will be used. Defaults to None.
        color (str, optional): The color of the text. Defaults to "black".
        **kwargs: Additional arguments to pass to matplotlib's AnchoredText or plotly's
            fig.add_annotation().

    Returns:
        plt.Axes | plt.Figure | go.Figure: The annotated figure.
    """
    backend = PLOTLY if isinstance(fig, go.Figure) else MATPLOTLIB
    color = kwargs.pop("color", "black")

    if backend == MATPLOTLIB:
        ax = fig if isinstance(fig, plt.Axes) else plt.gca()

        defaults = dict(frameon=False, loc="upper left", prop=dict(color=color))
        text_box = AnchoredText(text, **(defaults | kwargs))
        ax.add_artist(text_box)
    elif isinstance(fig, go.Figure):
        defaults = dict(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.96,
            showarrow=False,
            font=dict(size=16, color=color),
            align="left",
        )

        fig.add_annotation(text=text, **(defaults | kwargs))
    else:
        raise ValueError(f"Unexpected {fig=}")

    return fig


@validate_fig
def get_fig_xy_range(
    fig: go.Figure | plt.Figure | plt.Axes, trace_idx: int = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get the x and y range of a plotly or matplotlib figure.

    Args:
        fig (go.Figure | plt.Figure | plt.Axes): plotly/matplotlib figure or axes.
        trace_idx (int, optional): Index of the trace to use for measuring x/y limits.
            Defaults to 0. Unused if kaleido package is installed and the figure's
            actual x/y-range can be obtained from fig.full_figure_for_development().

    Returns:
        tuple[float, float, float, float]: The x and y range of the figure in the format
            (x_min, x_max, y_min, y_max).
    """
    if fig is None:
        fig = plt.gcf()
    if isinstance(fig, (plt.Figure, plt.Axes)):  # handle matplotlib
        ax = fig if isinstance(fig, plt.Axes) else fig.gca()

        return ax.get_xlim(), ax.get_ylim()

    # If kaleido is missing, try block raises ValueError: Full figure generation
    # requires the kaleido package. Install with: pip install kaleido
    # If so, we resort to manually computing the xy data ranges which are usually are
    # close to but not the same as the axes limits.
    try:
        # https://stackoverflow.com/a/62042077
        dev_fig = fig.full_figure_for_development(warn=False)
        xaxis_type = dev_fig.layout.xaxis.type
        yaxis_type = dev_fig.layout.yaxis.type

        x_range = dev_fig.layout.xaxis.range
        y_range = dev_fig.layout.yaxis.range

        # Convert log range to linear if necessary
        if xaxis_type == "log":
            x_range = [10**val for val in x_range]
        if yaxis_type == "log":
            y_range = [10**val for val in y_range]

    except ValueError:
        trace = fig.data[trace_idx]
        df_xy = pd.DataFrame({"x": trace.x, "y": trace.y}).dropna()

        # Determine ranges based on the type of axes
        if fig.layout.xaxis.type == "log":
            x_range = [10**val for val in (min(df_xy.x), max(df_xy.x))]
        else:
            x_range = [min(df_xy.x), max(df_xy.x)]

        if fig.layout.yaxis.type == "log":
            y_range = [10**val for val in (min(df_xy.y), max(df_xy.y))]
        else:
            y_range = [min(df_xy.y), max(df_xy.y)]

    return x_range, y_range
