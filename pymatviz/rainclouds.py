"""Raincloud plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots
from scipy import stats


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal


def rainclouds(
    data: Mapping[str, Sequence[float] | tuple[pd.DataFrame, str]],
    *,
    orientation: Literal["h", "v"] = "h",
    alpha: float = 0.7,
    width_viol: float = 0.3,
    width_box: float = 0.05,
    jitter: float = 0.01,
    point_size: float = 3,
    bw: float = 0.2,
    cut: float = 0.0,
    scale: Literal["area", "count", "width"] = "area",
    rain_offset: float = -0.25,
    offset: float | None = None,
    hover_data: Sequence[str] | dict[str, Sequence[str]] | None = None,
    show_violin: bool = True,
    show_box: bool = True,
    show_points: bool = True,
) -> go.Figure:
    """Create a raincloud plot for multiple datasets using Plotly.

    This plot type was proposed in https://wellcomeopenresearch.org/articles/4-63/v2.
    It is a vertical stack of:

    1. violin plot (the cloud)
    2. box plot (the umbrella)
    3. strip plot (the rain)

    Args:
        data (dict[str, Union[Sequence[float], tuple[pd.DataFrame, str]]]): A dictionary
            where keys are labels and values are either sequences of float data or
            tuples containing a DataFrame and the column name to plot. Dataframes can
            hold additional columns to be used in hover tooltips.
        orientation ("h" | "v", optional): Orientation of the plot.
            "h" for horizontal, "v" for vertical. Defaults to "h".
        alpha (float, optional): Transparency of the violin plots. Defaults to 0.7.
        width_viol (float, optional): Width of the violin plots. Defaults to 0.3.
        width_box (float, optional): Width of the box plots. Defaults to 0.05.
        jitter (float, optional): Amount of jitter for the strip plot. Defaults to 0.01.
        point_size (float, optional): Size of the points in the strip plot.
            Defaults to 3.
        bw (float, optional): Bandwidth for the KDE. Defaults to 0.2.
        cut (float, optional): Distance past extreme data points to extend KDE. Defaults
            to 0.0.
        scale ("area" | "count" | "width", optional): Method to scale the width
            of each violin. Defaults to "area".
        rain_offset (float, optional): Shift the strip plot position. Defaults to -0.25.
        offset (float | None, optional): Shift the violin plot position.
            Defaults to None.
        hover_data (Sequence[str] | dict[str, Sequence[str]] | None, optional):
            Additional data to be shown in hover tooltips. Can be a list of column names
            or a dict with the same keys as data and different column names for each
            trace.
        show_violin (bool, optional): Whether to show the violin plot. Defaults to True.
        show_box (bool, optional): Whether to show the box plot. Defaults to True.
        show_points (bool, optional): Whether to show the strip plot points.
            Defaults to True.
        **kwargs: Additional keyword arguments to pass to the plotting functions.

    Returns:
        go.Figure: The Plotly figure containing the raincloud plot.
    """

    def rgba_from_hex(hex_color: str, alpha: float) -> str:
        """Convert hex color to rgba."""
        rgb = pc.hex_to_rgb(hex_color)
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"

    fig = make_subplots(rows=1, cols=1)

    offset = max(width_box / 1.8, 0.15) + 0.05 if offset is None else offset
    positions = np.arange(len(data)) * 0.6

    for idx, (label, data_itm) in enumerate(data.items()):
        color = qualitative.Plotly[idx % len(qualitative.Plotly)]
        rgba_color = rgba_from_hex(color, alpha)
        pos = positions[idx]

        if (
            len(data_itm) == 2
            and isinstance(df_i := data_itm[0], pd.DataFrame)
            and isinstance(col := data_itm[1], str)
        ):
            values = df_i[col]
            if hover_data is None:
                hover_data = [col]
            elif isinstance(hover_data, list) and col not in hover_data:
                hover_data = [col, *hover_data]
            elif (
                isinstance(hover_data, dict)
                and label in hover_data
                and col not in hover_data[label]
            ):
                hover_data[label] = [col, *hover_data[label]]
        else:
            values = data_itm

        if show_violin:  # the cloud
            kde = stats.gaussian_kde(values, bw_method=bw)
            x_range = np.linspace(min(values) - cut, max(values) + cut, 100)
            y_range = kde(x_range)

            if scale == "area":
                y_range /= y_range.max()
            elif scale == "count":
                y_range *= len(values) / y_range.max()
            elif scale == "width":
                y_range /= y_range.max()

            common_violin_kwargs = dict(
                fill="toself",
                fillcolor=rgba_color,
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=True,  # Show in legend as the group representative
                name=label,
                legendgroup=label,
                hoverinfo="x+y",
                hoverlabel=dict(namelength=-1),
                hovertemplate=(f"{label}<br>%{{x:.3g}}<br>%{{y:.3g}}<extra></extra>"),
            )

            x_data = np.concatenate([x_range, x_range[::-1]])
            y_data = np.concatenate(
                [
                    [pos] * len(x_range),
                    (pos + y_range * width_viol)[::-1],
                ]
            )

            fig.add_scatter(
                x=x_data if orientation == "h" else y_data,
                y=y_data if orientation == "h" else x_data,
                **common_violin_kwargs,
            )

        if show_box:  # the umbrella
            common_box_kwargs = dict(
                name=label,
                boxpoints=False,
                width=width_box,
                fillcolor=rgba_color,
                line=dict(color=color),
                orientation=orientation,
                showlegend=False,
                legendgroup=label,
            )
            fig.add_box(
                x=values if orientation == "h" else [pos - offset / 2] * len(values),
                y=[pos - offset / 2] * len(values) if orientation == "h" else values,
                **common_box_kwargs,
            )

        if show_points:  # the rain
            rng = np.random.default_rng(seed=0)
            jitter_values = rng.normal(0, jitter, size=len(values))
            hover_key = data_itm[1] if isinstance(data_itm, tuple) else "value"
            hover_text = [f"{label}<br>{hover_key}: {val:.3g}" for val in values]

            if hover_data is not None:
                if isinstance(hover_data, dict):
                    cols_to_show = hover_data.get(label, [])
                else:
                    cols_to_show = hover_data

                if isinstance(data_itm, tuple):
                    df_i, col = data_itm
                    for col in cols_to_show:
                        if col in df_i:
                            for val_idx, val in enumerate(df_i[col]):
                                hover_text[val_idx] += f"<br>{col}: {val}"
                elif isinstance(hover_data, dict):
                    for col, col_data in hover_data.get(label, {}).items():  # type: ignore[union-attr]
                        for val_idx, val in enumerate(col_data):
                            hover_text[val_idx] += f"<br>{col}: {val}"

            common_scatter_kwargs = dict(
                mode="markers",
                marker=dict(color=color, size=point_size, opacity=0.5),
                showlegend=False,
                name=label,
                legendgroup=label,
                hoverinfo="text",
                hovertext=hover_text,
            )

            fig.add_scatter(
                x=values if orientation == "h" else pos + rain_offset + jitter_values,
                y=pos + rain_offset + jitter_values if orientation == "h" else values,
                **common_scatter_kwargs,
            )

    # Determine if labels should be horizontal or vertical
    labels = list(data)
    max_label_len = max(len(label) for label in labels)
    label_orientation = "v" if max_label_len > 10 else "h"

    # Calculate the range based on visible elements
    range_adjustment = 0.0
    if show_violin:
        range_adjustment += 0.2
    if show_box:
        range_adjustment += 0.1
    if show_points:
        range_adjustment += 0.1

    if orientation == "h":
        fig.update_yaxes(
            ticktext=labels,
            tickvals=positions,
            range=[positions[0] - range_adjustment, positions[-1] + range_adjustment],
            tickangle=0 if label_orientation == "h" else -90,
        )
        fig.update_xaxes(zeroline=False)
    else:
        fig.update_xaxes(
            ticktext=labels,
            tickvals=positions,
            range=[positions[0] - range_adjustment, positions[-1] + range_adjustment],
            tickangle=0 if label_orientation == "h" else -90,
        )
        fig.update_yaxes(zeroline=False)

    return fig
