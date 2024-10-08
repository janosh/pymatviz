"""Raincloud plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots
from scipy import stats


if TYPE_CHECKING:
    from collections.abc import Sequence


def rainclouds(
    data: dict[str, Sequence[float] | tuple[pd.DataFrame, str]],
    *,
    orientation: Literal["h", "v"] = "h",
    alpha: float = 0.7,
    width_viol: float = 0.3,
    width_box: float = 0.1,
    jitter: float = 0.01,
    point_size: float = 3,
    bw: float = 0.2,
    cut: float = 0.0,
    scale: Literal["area", "count", "width"] = "area",
    move: float = -0.15,
    offset: float | None = None,
    hover_data: Sequence[str] | dict[str, Sequence[str]] | None = None,
) -> go.Figure:
    """Create a raincloud plot for multiple datasets using Plotly.

    Args:
        data (dict[str, Union[Sequence[float], tuple[pd.DataFrame, str]]]): A dictionary
            where keys are labels and values are either sequences of float data or
            tuples containing a DataFrame and the column name to plot. Dataframes can
            hold additional columns to be used in hover tooltips.
        orientation ("h" | "v", optional): Orientation of the plot.
            "h" for horizontal, "v" for vertical. Defaults to "h".
        alpha (float, optional): Transparency of the violin plots. Defaults to 0.7.
        width_viol (float, optional): Width of the violin plots. Defaults to 0.3.
        width_box (float, optional): Width of the box plots. Defaults to 0.1.
        jitter (float, optional): Amount of jitter for the strip plot. Defaults to 0.01.
        point_size (float, optional): Size of the points in the strip plot.
            Defaults to 3.
        bw (float, optional): Bandwidth for the KDE. Defaults to 0.2.
        cut (float, optional): Distance past extreme data points to extend KDE. Defaults
            to 0.0.
        scale ("area" | "count" | "width", optional): Method to scale the width
            of each violin. Defaults to "area".
        move (float, optional): Adjustment for the strip plot position.
            Defaults to -0.15.
        offset (float | None, optional): Adjustment for the violin plot position.
            Defaults to None.
        hover_data (Sequence[str] | dict[str, Sequence[str]] | None, optional):
            Additional data to be shown in hover tooltips. Can be a list of column names
            or a dict with the same keys as data and different column names for each
            trace.
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

    for idx, (label, data_item) in enumerate(data.items()):
        color = qualitative.Plotly[idx % len(qualitative.Plotly)]
        rgba_color = rgba_from_hex(color, alpha)
        pos = positions[idx]

        if (
            len(data_item) == 2
            and isinstance(df_i := data_item[0], pd.DataFrame)
            and isinstance(col := data_item[1], str)
        ):
            values = df_i[col]
            if hover_data is None:
                hover_data = [col]
            elif isinstance(hover_data, list) and col not in hover_data:
                hover_data.insert(0, col)
            elif (
                isinstance(hover_data, dict)
                and label in hover_data
                and col not in hover_data[label]
            ):
                hover_data[label] = [col, *hover_data[label]]
        else:
            values = data_item

        # Violin plot (half cloud)
        kde = stats.gaussian_kde(values, bw_method=bw)
        x_range = np.linspace(min(values) - cut, max(values) + cut, 100)
        y_range = kde(x_range)

        if scale == "area":
            y_range /= y_range.max()
        elif scale == "count":
            y_range *= len(values) / y_range.max()
        elif scale == "width":
            y_range /= y_range.max()

        violin_offset = 0.1

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
                [pos + violin_offset] * len(x_range),
                (pos + violin_offset + y_range * width_viol)[::-1],
            ]
        )

        fig.add_scatter(
            x=x_data if orientation == "h" else y_data,
            y=y_data if orientation == "h" else x_data,
            **common_violin_kwargs,
        )

        # Box plot (umbrella)
        common_box_kwargs = dict(
            name=label,
            boxpoints=False,
            width=width_box,
            fillcolor=rgba_color,
            line=dict(color=color),
            orientation=orientation,
            showlegend=False,  # Hide from legend
            legendgroup=label,
        )
        fig.add_box(
            x=values if orientation == "h" else [pos] * len(values),
            y=[pos] * len(values) if orientation == "h" else values,
            **common_box_kwargs,
        )

        # Strip plot (rain)
        jitter_values = np.random.default_rng().normal(0, jitter, size=len(values))
        hover_text = [
            f"{label}<br>"
            f"{data_item[1] if isinstance(data_item, tuple) else 'value'}: {val:.3g}"
            for val in values
        ]

        if hover_data is not None:
            if isinstance(hover_data, dict):
                columns_to_show = hover_data.get(label, [])
            else:
                columns_to_show = hover_data

            if isinstance(data_item, tuple):
                df_i, col = data_item
                for col in columns_to_show:
                    if col in df_i.columns:
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
            x=values if orientation == "h" else pos + move + jitter_values,
            y=pos + move + jitter_values if orientation == "h" else values,
            **common_scatter_kwargs,
        )

    # Determine if labels should be horizontal or vertical
    labels = list(data)
    max_label_len = max(len(label) for label in labels)
    label_orientation = "v" if max_label_len > 10 else "h"

    if orientation == "h":
        fig.update_yaxes(
            ticktext=labels,
            tickvals=positions,
            range=[positions[0] - 0.4, positions[-1] + 0.4],
            tickangle=0 if label_orientation == "h" else -90,
        )
        fig.update_xaxes(zeroline=False)
    else:
        fig.update_xaxes(
            ticktext=labels,
            tickvals=positions,
            range=[positions[0] - 0.4, positions[-1] + 0.4],
            tickangle=0 if label_orientation == "h" else -90,
        )
        fig.update_yaxes(zeroline=False)

    return fig
