"""Plotting-related utility functions.

Available functions:
    - annotate: Annotate a matplotlib or plotly figure with text.
    - apply_matplotlib_template: Set default matplotlib configurations for consistency.
    - get_font_color: Get the font color used in a Matplotlib or Plotly figure.
    - get_fig_xy_range: Get the x and y range of a plotly or matplotlib figure.
    - luminance: Compute the luminance of a color.
    - pick_max_contrast_color: Choose black or white text color for contrast.
    - pretty_label: Map metric keys to their pretty labels.
    - validate_fig: Decorator to validate the type of fig keyword argument.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import matplotlib.colors
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
    ColorType,
    P,
    R,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any


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

                subplot_idx = trace.xaxis[1:] or ""  # e.g. 'x2' -> '2', 'x' -> ''
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


def _get_plotly_font_color(fig: go.Figure) -> str:
    """Get the font color used in a Plotly figure.

    Args:
        fig (go.Figure): A Plotly figure object.

    Returns:
        str: The font color as a string (e.g. 'black', '#000000').
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
        str: The font color as a string (e.g. 'black', '#000000').
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
        str: The font color as a string (e.g. 'black', '#000000').

    Raises:
        TypeError: If fig is not a Matplotlib or Plotly figure.
    """
    if isinstance(fig, go.Figure):
        return _get_plotly_font_color(fig)
    if isinstance(fig, plt.Figure | plt.Axes):
        return _get_matplotlib_font_color(fig)
    raise TypeError(f"Input must be {VALID_FIG_NAMES}, got {type(fig)=}")


def luminance(color: ColorType) -> float:
    """Compute the relative luminance of a color using the WCAG 2.0 formula.

    Args:
        color (ColorType): RGB color tuple with values in [0, 1] or [0, 255], or a color
            string that can be converted to RGB.

    Returns:
        float: Relative luminance of the color in range [0, 1].
    """
    if isinstance(color, str) and color.startswith("rgb("):
        r, g, b, *_a = map(float, color.strip("rgb()").split(","))
        if r > 1 or g > 1 or b > 1:
            r, g, b = r / 255, g / 255, b / 255
    elif isinstance(color, tuple) and len(color) >= 3:
        # Check if any value is > 1, indicating 0-255 range
        if any(c > 1 for c in color[:3]):
            r, g, b = color[0] / 255, color[1] / 255, color[2] / 255
        else:
            r, g, b = color[:3]
    else:
        # raises ValueError if color invalid
        r, g, b, *_a = matplotlib.colors.to_rgba(color)

    def _convert_rgb_to_linear(rgb: float) -> float:
        """Convert an RGB value to linear RGB (remove gamma correction)."""
        return rgb / 12.92 if rgb <= 0.03928 else ((rgb + 0.055) / 1.055) ** 2.4

    # Convert RGB to linear RGB (remove gamma correction)
    r, g, b = map(_convert_rgb_to_linear, (r, g, b))

    # Calculate relative luminance using WCAG 2.0 coefficients
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color1: ColorType, color2: ColorType) -> float:
    """Calculate the contrast ratio between two colors according to WCAG 2.0.

    Args:
        color1 (ColorType): First color (RGB tuple with values in [0, 1] or [0, 255],
            or a color string that can be converted to RGB).
        color2 (ColorType): Second color (RGB tuple with values in [0, 1] or [0, 255],
            or a color string that can be converted to RGB).

    Returns:
        float: Contrast ratio between the two colors, ranging from 1:1 to 21:1.
    """
    lum1 = luminance(color1)
    lum2 = luminance(color2)

    # Ensure lighter color is first for the formula
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)

    # Calculate contrast ratio: (L1 + 0.05) / (L2 + 0.05)
    return (lighter + 0.05) / (darker + 0.05)


def pick_max_contrast_color(
    bg_color: ColorType,
    colors: tuple[ColorType, ColorType] = ("white", "black"),
    min_contrast_ratio: float = 2.0,  # Lower threshold makes dark colors get white text
) -> ColorType:
    """Choose text color for a given background color based on WCAG 2.0 contrast ratio.

    This function calculates the contrast ratio between the background color and each
    of the provided text colors, then returns the color with the highest contrast ratio.
    If the contrast ratio with white is above the minimum contrast ratio, white will be
    chosen even if black has a slightly higher contrast ratio. This ensures that darker
    colors always get white text, which is often more readable in 3D visualizations.

    Args:
        bg_color (ColorType): Background color.
        colors (tuple[ColorType, ColorType], optional): Text colors to choose
            from. Defaults to ("white", "black").
        min_contrast_ratio (float, optional): Minimum contrast ratio to prefer white
            over black text. Defaults to 2.0 (lower than WCAG AA standard to ensure
            dark colors get white text).

    Returns:
        ColorType: item in `colors` that provides the best contrast with bg_color.
    """
    # Calculate contrast ratios for each potential text color
    contrast_ratios = [contrast_ratio(bg_color, color) for color in colors]

    # If the contrast ratio with white is above the minimum contrast ratio,
    # prefer white text even if black has a slightly higher contrast ratio
    if contrast_ratios[0] >= min_contrast_ratio:
        return colors[0]

    # Otherwise, return the color with the highest contrast ratio
    return colors[contrast_ratios.index(max(contrast_ratios))]


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
    fig: go.Figure | plt.Figure | plt.Axes,
    traces: int | slice | Sequence[int] | Callable[[go.Scatter], bool] = 0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get the x and y range of a plotly or matplotlib figure.

    Args:
        fig (go.Figure | plt.Figure | plt.Axes): plotly/matplotlib figure or axes.
        traces (int | slice | Sequence[int] | Callable, optional): Specifies which
            trace(s) to use for determining the x/y range. Can be:
            - int: A single trace index (default: 0)
            - slice: A slice object to select a range of traces
            - list[int]: A list of specific trace indices
            - Callable: A function that takes a trace and returns True/False
            Only used if kaleido package is not installed or for matplotlib figures.

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
        x_axis_type = dev_fig.layout.xaxis.type
        y_axis_type = dev_fig.layout.yaxis.type

        x_range = dev_fig.layout.xaxis.range
        y_range = dev_fig.layout.yaxis.range

        # Convert log range to linear if necessary
        if x_axis_type == "log":
            x_range = [10**val for val in x_range]
        if y_axis_type == "log":
            y_range = [10**val for val in y_range]

    except ValueError:
        # Select a trace to use for determining the range
        trace_index = 0
        if isinstance(traces, int):
            trace_index = traces
        elif isinstance(traces, slice):
            indices = list(range(*traces.indices(len(fig.data))))
            trace_index = indices[0] if indices else 0
        elif isinstance(traces, list):
            trace_index = traces[0] if traces else 0
        elif callable(traces):
            for idx, trace in enumerate(fig.data):
                if traces(trace):
                    trace_index = idx
                    break

        trace = fig.data[trace_index]
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
