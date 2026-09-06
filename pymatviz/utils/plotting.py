"""Plotting-related utility functions.

Available functions:
    - annotate: Annotate a plotly figure with text.
    - annotated_heatmap: Heatmap with per-cell text, replacing the removed
      plotly.figure_factory.create_annotated_heatmap.
    - get_font_color: Get the font color used in a Plotly figure.
    - get_fig_xy_range: Get the x and y range of a plotly figure.
    - luminance: Compute the luminance of a color.
    - pick_max_contrast_color: Choose black or white text color for contrast.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from pymatviz.colors import _CSS_NAMED_COLORS


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from pymatviz.typing import ColorType


PRETTY_LABELS: Final[dict[str, str]] = {
    "R2": "R<sup>2</sup>",
    "R2_adj": "R<sup>2</sup><sub>adj</sub>",
}

PLOTLY_LINE_STYLES: Final[tuple[str, ...]] = (
    "solid",
    "dot",
    "dash",
    "longdash",
    "dashdot",
    "longdashdot",
)


def annotate(text: str | Sequence[str], fig: go.Figure, **kwargs: Any) -> go.Figure:
    """Annotate a Plotly figure or its facets, skipping empty facet labels.

    Args:
        text (str): The text to use for annotation. If fig is plotly faceted, text can
            be a list of strings to annotate each subplot.
        fig (go.Figure): The plotly Figure to annotate.
        **kwargs: Additional arguments to pass to plotly's fig.add_annotation().

    Returns:
        go.Figure: The annotated figure.

    Raises:
        TypeError: If fig is not a Plotly figure.
    """
    if not isinstance(fig, go.Figure):
        raise TypeError(f"Expected plotly Figure, got {type(fig)}")

    color = kwargs.pop("color", get_font_color(fig))

    text_defaults = dict(
        x=0.02, y=0.96, showarrow=False, font=dict(size=16, color=color), align="left"
    )

    # Annotate all subplots or main plot if not faceted (i.e. if any trace is
    # assigned to an x-axis other than the primary "x")
    if any(getattr(trace, "xaxis", None) not in (None, "x") for trace in fig.data):
        for idx, trace in enumerate(fig.data):
            sub_text = text if isinstance(text, str) else text[idx]
            # skip traces for which no annotations were provided
            if not sub_text:
                continue

            subplot_idx = trace.xaxis[1:]  # e.g. 'x2' -> '2', 'x' -> ''
            xref = f"x{subplot_idx} domain"
            yref = f"y{subplot_idx} domain"
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

    return fig


def get_font_color(fig: go.Figure) -> str:
    """Get the font color from a Plotly figure, its template, or the global template.

    Defaults to black when no font color is set. Raises TypeError for non-figures.
    """
    if not isinstance(fig, go.Figure):
        raise TypeError(f"Input must be plotly Figure, got {type(fig)=}")
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


def _rgb_components(color: ColorType) -> tuple[float, float, float]:
    """Normalize RGB(A) tuples, CSS names, hex, and rgb/rgba byte channels."""
    if isinstance(color, str):
        color = color.strip().lower()
        color = _CSS_NAMED_COLORS.get(color, color)
        if color.startswith("#"):
            if not re.fullmatch(r"#[0-9a-f]{3}(?:[0-9a-f]{3})?", color):
                raise ValueError(f"Invalid hex color: {color}")
            hex_value = color[1:]
            step = len(hex_value) // 3
            channels = tuple(
                int(hex_value[idx : idx + step], 16) / (16**step - 1)
                for idx in range(0, len(hex_value), step)
            )
        elif match := re.fullmatch(r"rgba?\(([^()]*)\)", color):
            try:
                values = tuple(float(value) for value in match[1].split(","))
            except ValueError as exc:
                raise ValueError(f"Invalid RGB color: {color}") from exc
            if len(values) not in (3, 4) or (
                len(values) == 4 and not 0 <= values[3] <= 1
            ):
                raise ValueError(f"Invalid RGB color: {color}")
            channels = tuple(value / 255 for value in values[:3])
        else:
            raise ValueError(f"Unsupported color format: {color}")
    elif isinstance(color, tuple) and len(color) in (3, 4):
        try:
            channels = tuple(float(cast("Any", channel)) for channel in color[:3])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported color tuple: {color}") from exc
        if any(channel > 1 for channel in channels):
            channels = tuple(channel / 255 for channel in channels)
    else:
        raise ValueError(f"Unsupported color type: {type(color)}")

    if not all(0 <= channel <= 1 for channel in channels):
        raise ValueError(f"RGB channels must be finite and in [0, 255]: {color}")
    red, green, blue = channels
    return red, green, blue


def luminance(color: ColorType) -> float:
    """Compute the relative luminance of a color using the WCAG 2.0 formula.

    Args:
        color (ColorType): RGB color tuple with values in [0, 1] or [0, 255], or a color
            string (CSS name, hex, or rgb/rgba with channels in [0, 255]).
            Alpha is ignored.

    Returns:
        float: Relative luminance of the color in range [0, 1].
    """
    red, green, blue = (
        channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in _rgb_components(color)
    )
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


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


def get_fig_xy_range(
    fig: go.Figure,
    traces: int | slice | Sequence[int] | Callable[[go.Scatter], bool] = 0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get the x and y range of a plotly figure.

    Args:
        fig (go.Figure): plotly figure.
        traces (int | slice | Sequence[int] | Callable, optional): Specifies which
            trace(s) to use for determining the x/y range. Can be:
            - int: A single trace index (default: 0)
            - slice: A slice object to select a range of traces
            - list[int]: A list of specific trace indices
            - Callable: A function that takes a trace and returns True/False

    Returns:
        tuple: The axis ranges as ((x_min, x_max), (y_min, y_max)).
    """
    if not isinstance(fig, go.Figure):
        raise TypeError(f"Expected plotly Figure, got {type(fig)}")

    if isinstance(traces, int):
        selected = [fig.data[traces]]
    elif isinstance(traces, slice):
        selected = list(fig.data[traces])
    elif callable(traces):
        trace_predicate = cast("Callable[[go.Scatter], bool]", traces)
        selected = [trace for trace in fig.data if trace_predicate(trace)]
    else:
        selected = [fig.data[idx] for idx in traces]
    if not selected:
        raise ValueError(f"No valid traces with required data found for {traces=}")

    # Without Kaleido, use data bounds; these omit Plotly's autorange padding.
    try:
        selected_fig = go.Figure(data=selected, layout=fig.layout)
        dev_fig = selected_fig.full_figure_for_development(warn=False)
        x_axis_type = dev_fig.layout.xaxis.type
        y_axis_type = dev_fig.layout.yaxis.type

        x_range: tuple[float, float] = dev_fig.layout.xaxis.range
        y_range: tuple[float, float] = dev_fig.layout.yaxis.range

        # Convert log range to linear if necessary
        if x_axis_type == "log":
            x_range = (10 ** x_range[0], 10 ** x_range[1])
        if y_axis_type == "log":
            y_range = (10 ** y_range[0], 10 ** y_range[1])

    except ValueError:
        df_xy = pd.concat(
            [pd.DataFrame({"x": trace.x, "y": trace.y}) for trace in selected]
        ).dropna()
        if df_xy.empty:
            raise ValueError("No valid traces with required data found") from None

        ranges = []
        for axis in ("x", "y"):
            layout_axis = fig.layout[f"{axis}axis"]
            values = df_xy[axis]
            if layout_axis.type == "log":
                values = values[values > 0]
            if values.empty:
                raise ValueError(f"No valid data for {axis}-axis") from None
            bounds = (values.min(), values.max())
            if layout_axis.range is not None:
                bounds = tuple(
                    auto
                    if bound is None
                    else 10**bound
                    if layout_axis.type == "log"
                    else bound
                    for auto, bound in zip(bounds, layout_axis.range, strict=True)
                )
            ranges.append(bounds)
        x_range, y_range = ranges

    return x_range, y_range


def _scale_end_colors(
    colorscale: Any, font_colors: Any, *, reversescale: bool
) -> tuple[str, str]:
    """Annotation colors for the low and high ends of a colorscale."""
    white, black = "#FFFFFF", "#000000"
    if font_colors is not None and len(font_colors):
        return font_colors[0], font_colors[-1]
    if colorscale is None:
        return black, black
    low, high = colorscale[0][1], colorscale[-1][1]
    if reversescale:
        low, high = high, low
    return (
        black if luminance(low) > 0.7 else white,
        black if luminance(high) > 0.7 else white,
    )


def annotated_heatmap(
    z: Any,
    x: Any = None,
    y: Any = None,
    annotation_text: Any = None,
    colorscale: Any = "Plasma",
    font_colors: Any = None,
    *,
    showscale: bool = False,
    reversescale: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """Heatmap with a text annotation centered on every cell.

    Args:
        z: 2D array of values driving the cell colors.
        x: Column labels. Defaults to positional indices.
        y: Row labels. Defaults to positional indices.
        annotation_text: 2D array of per-cell text. Defaults to z.
        colorscale: Any plotly colorscale, named or as a list of stops.
        font_colors: Annotation colors for the low and high ends of the scale. Chosen
            from the colorscale for contrast when omitted.
        showscale: Whether to draw the colorbar.
        reversescale: Whether to reverse the colorscale.
        **kwargs: Forwarded to the heatmap trace. zmin/zmax/zmid also move the
            threshold at which annotations switch between the two font colors.

    Returns:
        go.Figure: The annotated heatmap.
    """
    z_arr = np.asarray(z)
    annotation_text = z_arr if annotation_text is None else np.asarray(annotation_text)
    x_labels = list(x) if x is not None else range(z_arr.shape[1])
    y_labels = list(y) if y is not None else range(z_arr.shape[0])

    # the value at which annotations flip from the low to the high font color
    z_min = np.nanmin(z_arr) if (kw_min := kwargs.get("zmin")) is None else kw_min
    z_max = np.nanmax(z_arr) if (kw_max := kwargs.get("zmax")) is None else kw_max
    z_mid = (z_max + z_min) / 2 if (kw_mid := kwargs.get("zmid")) is None else kw_mid

    labelled = x is not None or y is not None
    trace = go.Heatmap(
        z=z,
        colorscale=colorscale,
        showscale=showscale,
        reversescale=reversescale,
        **({"x": x, "y": y} if labelled else {}),
        **kwargs,
    )

    low_color, high_color = _scale_end_colors(
        trace.colorscale, font_colors, reversescale=reversescale
    )
    annotations = [
        go.layout.Annotation(
            text=str(annotation_text[row_idx][col_idx]),
            x=x_labels[col_idx],
            y=y_labels[row_idx],
            xref="x1",
            yref="y1",
            font=dict(color=low_color if val < z_mid else high_color),
            showarrow=False,
        )
        for row_idx, row in enumerate(z_arr)
        for col_idx, val in enumerate(row)
    ]

    axis_extra = {} if labelled else dict(showticklabels=False)
    layout = dict(
        annotations=annotations,
        xaxis=dict(
            ticks="",
            side="top",
            gridcolor="rgb(0, 0, 0)",
            **({"dtick": 1} if labelled else {}),
            **axis_extra,
        ),
        yaxis=dict(
            ticks="",
            ticksuffix="  ",
            **({"dtick": 1} if labelled else {}),
            **axis_extra,
        ),
    )
    return go.Figure(data=[trace], layout=layout)
