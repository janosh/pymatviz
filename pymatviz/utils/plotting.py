"""Plotting-related utility functions."""

from __future__ import annotations

import re
import warnings
from functools import wraps
from typing import TYPE_CHECKING, Literal, cast

import matplotlib as mpl


if TYPE_CHECKING:
    from typing import Literal


import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.offsetbox import AnchoredText

from pymatviz.typing import (
    BACKENDS,
    MATPLOTLIB,
    PLOTLY,
    VALID_FIG_NAMES,
    AxOrFig,
    Backend,
    P,
    R,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from matplotlib.ticker import Formatter


def annotate(text: str | Sequence[str], fig: AxOrFig, **kwargs: Any) -> AxOrFig:
    """Annotate a matplotlib or plotly figure. Supports faceted plots plotly figure with
    trace with empty strings skipped.

    Args:
        text (str): The text to use for annotation. If fig is plotly faceted, text can
            be a list of strings to annotate each subplot.
        fig (plt.Axes | plt.Figure | go.Figure | None, optional): The matplotlib Axes,
            Figure or plotly Figure to annotate.
        **kwargs: Additional arguments to pass to matplotlib's AnchoredText or plotly's
            fig.add_annotation().

    Returns:
        plt.Axes | plt.Figure | go.Figure: The annotated figure.

    Raises:
        TypeError: If fig is not a Matplotlib or Plotly figure.
    """
    color = kwargs.pop("color", get_font_color(fig))

    if isinstance(fig, plt.Figure | plt.Axes):
        ax = fig if isinstance(fig, plt.Axes) else plt.gca()
        text_defaults = dict(frameon=False, loc="upper left", prop=dict(color=color))
        text_box = AnchoredText(text, **(text_defaults | kwargs))
        ax.add_artist(text_box)
    elif isinstance(fig, go.Figure):
        text_defaults = dict(
            x=0.02,
            y=0.96,
            showarrow=False,
            font=dict(size=16, color=color),
            align="left",
        )

        # Annotate all subplots or main plot if not faceted
        if any(
            getattr(trace, "xaxis", None) not in (None, "x") for trace in fig.data
        ):  # Faceted plot
            for idx, trace in enumerate(fig.data):
                # if text is str, use it for all subplots though we might want to
                # warn since this will likely rarely be intended
                sub_text = text if isinstance(text, str) else text[idx]
                # skip traces for which no annotations were provided
                if not sub_text:
                    continue

                subplot_idx = trace.xaxis[1:] or ""  # e.g., 'x2' -> '2', 'x' -> ''
                xref = f"x{subplot_idx} domain" if subplot_idx else "x domain"
                yref = f"y{subplot_idx} domain" if subplot_idx else "y domain"
                fig.add_annotation(
                    text=sub_text,
                    **(dict(xref=xref, yref=yref) | text_defaults | kwargs),
                )
        else:  # Non-faceted plot
            if not isinstance(text, str):
                text_type = type(text).__name__
                raise ValueError(
                    f"Unexpected {text_type=} for non-faceted plot, must be str"
                )
            fig.add_annotation(
                text=text, **(dict(xref="paper", yref="paper") | text_defaults | kwargs)
            )
    else:
        raise TypeError(f"Unexpected {fig=}")

    return fig


def apply_matplotlib_template() -> None:
    """Set default matplotlib configurations for consistency.
    - Font size: 14 for readability.
    - Savefig: Tight bounding box and 200 DPI for high-quality saved plots.
    - Axes: Title size 16, bold weight for emphasis.
    - Figure: DPI 200, title size 20, bold weight for better visibility.
    - Layout: Enables constrained layout to reduce element overlap.
    """
    plt.rc("font", size=14)
    plt.rc("savefig", bbox="tight", dpi=200)
    plt.rc("axes", titlesize=16, titleweight="bold")
    plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
    plt.rcParams["figure.constrained_layout.use"] = True


def get_cbar_label_formatter(
    *,
    cbar_label_fmt: str,
    values_fmt: str,
    values_show_mode: Literal["value", "fraction", "percent", "off"],
    sci_notation: bool,
    default_decimal_places: int = 1,
) -> Formatter:
    """Generate colorbar tick label formatter.

    Work differently for different values_show_mode:
        - "value/fraction" mode: Use cbar_label_fmt (or values_fmt) as is.
        - "percent" mode: Get number of decimal places to keep from fmt
            string, for example 1 from ".1%".

    Args:
        cbar_label_fmt (str): f-string option for colorbar tick labels.
        values_fmt (str): f-string option for tile values, would be used if
            cbar_label_fmt is "auto".
        values_show_mode (str): The values display mode:
            - "off": Hide values.
            - "value": Display values as is.
            - "fraction": As a fraction of the total (0.10).
            - "percent": As a percentage of the total (10%).
        sci_notation (bool): Whether to use scientific notation for values and
            colorbar tick labels.
        default_decimal_places (int): Default number of decimal places
            to use if above fmt is invalid.

    Returns:
        PercentFormatter or FormatStrFormatter.
    """
    from matplotlib.ticker import FormatStrFormatter, PercentFormatter, ScalarFormatter

    cbar_label_fmt = values_fmt if cbar_label_fmt == "auto" else cbar_label_fmt

    if values_show_mode == "percent":
        if match := re.search(r"\.(\d+)%", cbar_label_fmt):
            decimal_places = int(match[1])
        else:
            warnings.warn(
                f"Invalid {cbar_label_fmt=}, use {default_decimal_places=}",
                stacklevel=2,
            )
            decimal_places = default_decimal_places
        return PercentFormatter(xmax=1, decimals=decimal_places)

    if sci_notation:
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_powerlimits((0, 0))
        return formatter

    return FormatStrFormatter(f"%{cbar_label_fmt}")


def _get_plotly_font_color(fig: go.Figure) -> str:
    """Get the font color used in a Plotly figure.

    Args:
        fig (go.Figure): A Plotly figure object.

    Returns:
        str: The font color as a string (e.g., 'black', '#000000').
    """
    if fig.layout.font and fig.layout.font.color:
        return fig.layout.font.color

    if (
        fig.layout.template
        and fig.layout.template.layout
        and fig.layout.template.layout.font
        and fig.layout.template.layout.font.color
    ):
        return fig.layout.template.layout.font.color

    template = pio.templates.default
    if isinstance(template, str):
        template = pio.templates[template]
    if template.layout and template.layout.font and template.layout.font.color:
        return template.layout.font.color

    return "black"


def _get_matplotlib_font_color(fig: plt.Figure | plt.Axes) -> str:
    """Get the font color used in a Matplotlib figure/axes.

    Args:
        fig (plt.Figure | plt.Axes): A Matplotlib figure or axes object.

    Returns:
        str: The font color as a string (e.g., 'black', '#000000').
    """
    ax = fig if isinstance(fig, plt.Axes) else fig.gca()

    # Check axes text color
    for text_element in (ax.xaxis.label, ax.yaxis.label, ax.title):
        text_color = text_element.get_color()
        if text_color != "auto":
            return text_color

    # Check tick label color
    x_labels = ax.xaxis.get_ticklabels()
    tick_color = x_labels[0].get_color() if x_labels else None
    if tick_color is not None and tick_color != "auto":
        return tick_color

    # Check rcParams
    return plt.rcParams.get("text.color", "black")


def get_font_color(fig: AxOrFig) -> str:
    """Get the font color used in a Matplotlib figure/axes or a Plotly figure.

    Args:
        fig (plt.Figure | plt.Axes | go.Figure): A Matplotlib or Plotly figure object.

    Returns:
        str: The font color as a string (e.g., 'black', '#000000').

    Raises:
        TypeError: If fig is not a Matplotlib or Plotly figure.
    """
    if isinstance(fig, go.Figure):
        return _get_plotly_font_color(fig)
    if isinstance(fig, plt.Figure | plt.Axes):
        return _get_matplotlib_font_color(fig)
    raise TypeError(f"Input must be {VALID_FIG_NAMES}, got {type(fig)=}")


def luminance(color: str | tuple[float, float, float]) -> float:
    """Compute the luminance of a color as in https://stackoverflow.com/a/596243.

    Args:
        color (tuple[float, float, float]): RGB color tuple with values in [0, 1].

    Returns:
        float: Luminance of the color.
    """
    # raises ValueError if color invalid
    red, green, blue = mpl.colors.to_rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def pick_bw_for_contrast(
    color: tuple[float, float, float] | str,
    text_color_threshold: float = 0.7,
) -> Literal["black", "white"]:
    """Choose black or white text color for a given background color based on luminance.

    Args:
        color (tuple[float, float, float] | str): RGB color tuple with values in [0, 1].
        text_color_threshold (float, optional): Luminance threshold for choosing
            black or white text color. Defaults to 0.7.

    Returns:
        "black" | "white": depending on the luminance of the background color.
    """
    if isinstance(color, str):
        color = mpl.colors.to_rgb(color)

    light_bg = luminance(cast(tuple[float, float, float], color)) > text_color_threshold
    return "black" if light_bg else "white"


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


def validate_fig(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to validate the type of fig keyword argument in a function. fig MUST be
    a keyword argument, not a positional argument.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # TODO use typing.ParamSpec to type wrapper once py310 is oldest supported
        fig = kwargs.get("fig")
        if fig is not None and not isinstance(fig, plt.Axes | plt.Figure | go.Figure):
            raise TypeError(
                f"Unexpected type for fig: {type(fig).__name__}, must be one of None, "
                f"{VALID_FIG_NAMES}"
            )
        return func(*args, **kwargs)

    return wrapper


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
    if isinstance(fig, plt.Figure | plt.Axes):  # handle matplotlib
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
