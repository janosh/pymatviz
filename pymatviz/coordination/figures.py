"""Plotly plots of coordination numbers distributions."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from inspect import isclass
from typing import TYPE_CHECKING, cast

import numpy as np
import plotly.graph_objects as go
from plotly.colors import label_rgb
from plotly.subplots import make_subplots
from pymatgen.analysis.local_env import NearNeighbors

from pymatviz.colors import ELEM_COLORS_JMOL
from pymatviz.coordination.helpers import (
    CnSplitMode,
    calculate_average_cn,
    create_hover_text,
    normalize_get_neighbors,
)
from pymatviz.enums import ElemColorScheme
from pymatviz.process_data import normalize_structures
from pymatviz.structure.helpers import get_site_elements, get_site_symbol


if TYPE_CHECKING:
    from typing import Any, Literal

    import pandas as pd

    from pymatviz.typing import AnyStructure


def _resolve_element_colors(
    element_color_scheme: ElemColorScheme | dict[str, str],
) -> dict[str, Any]:
    """Resolve element colors from a scheme enum or custom dict (merged over Jmol)
    with RGB tuples converted to plotly 'rgb(r, g, b)' strings.
    """
    if isinstance(element_color_scheme, dict):
        color_map = ELEM_COLORS_JMOL | element_color_scheme
    elif isinstance(element_color_scheme, ElemColorScheme):
        color_map = element_color_scheme.color_map
    else:
        raise TypeError(
            f"Invalid {element_color_scheme=}. Must be {', '.join(ElemColorScheme)} "
            "or a custom dict."
        )
    return {
        elem: label_rgb(color)
        if isinstance(color, tuple) and len(color) == 3
        else color
        for elem, color in color_map.items()
    }


def coordination_hist(
    structures: AnyStructure
    | Mapping[str, AnyStructure]
    | Sequence[AnyStructure]
    | pd.Series,
    *,
    strategy: float | NearNeighbors | type[NearNeighbors] = 3.0,
    split_mode: CnSplitMode | str = CnSplitMode.by_element,
    bar_mode: Literal["group", "stack"] = "stack",
    hover_data: Sequence[str] | dict[str, str] | None = None,
    element_color_scheme: ElemColorScheme | dict[str, str] = ElemColorScheme.pastel,
    annotate_bars: bool | dict[str, Any] = False,
    bar_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a plotly histogram of coordination numbers for given structure(s).

    Args:
        structures (AnyStructure | dict[str, AnyStructure] | Sequence[AnyStructure]):
            A single structure or a dictionary or sequence of structures.
        strategy (float | NearNeighbors | type[NearNeighbors]): Neighbor-finding
            strategy. Can be one of:
            - float: Cutoff distance for neighbor search in Angstroms.
            - NearNeighbors: An instance of a NearNeighbors subclass.
            - Type[NearNeighbors]: A NearNeighbors subclass (will be instantiated).
            Defaults to 3.0 (Angstroms cutoff).
        split_mode (CnSplitMode | str): How to split the data into subplots or color
            groups. Can be one of:
            "none": Single plot with all data. All elements of all structures (if
                multiple were passed) will be shown in the same plot.
            "by element": Split into subplots by element. Matching colors across
                subplots for different elements indicate those elements belong to the
                same structure.
            "by structure": Split into subplots by structure, i.e. each structure
                gets its own subplot with coordination numbers for all sites plotted
                in the same color.
            "by structure and element": Like "by structure", each structure gets its
                own subplot, but elements are colored differently within each structure.
        bar_mode ("group" | "stack"): How to arrange bars at the same
            coordination number. Can be one of:
            "group": Bars are grouped side by side.
            "stack": Bars are stacked on top of each other.
        hover_data (Sequence[str] | dict[str, str] | None): Sequence of keys or dict
            mapping keys to pretty labels for additional data to be shown in the hover
            tooltip. The keys must exist in the site properties or properties dict of
            the structure.
        element_color_scheme (ElemColorScheme | dict[str, str]): Color scheme for
            elements. Can be "jmol", "vesta", or a
            custom dict.
        annotate_bars (bool | dict[str, Any]): If True, annotate bars with element
            symbols when split_mode is 'by_element' or 'by_structure_and_element'. If a
            dict, used as keywords for bar annotations, e.g. {"font_size": 12,
            "font_color": "red"}.
        bar_kwargs (dict[str, Any] | None): Dictionary of keyword arguments to
            customize bar appearance.
            These will be passed to go.Bar().
        subplot_kwargs (dict[str, Any] | None): Additional keyword arguments to pass to
            make_subplots().

    Returns:
        go.Figure: A plotly Figure object containing the histogram.
    """
    struct_dict = normalize_structures(structures)

    # coord_data: coordination numbers and hover data for each structure and element
    coord_data: dict[Hashable, dict[str, Any]] = {}
    min_cn, max_cn = float("inf"), 0  # will be updated in the loop below

    # Map hover_data keys to their display labels
    hover_labels: dict[str, str]
    if isinstance(hover_data, Sequence):
        hover_labels = {key: key for key in hover_data}
    elif hover_data is None:
        hover_labels = {}
    elif isinstance(hover_data, dict):
        hover_labels = hover_data
    else:
        raise TypeError(f"Invalid {hover_data=}")

    get_neighbors = normalize_get_neighbors(strategy)

    for struct_key, structure in struct_dict.items():
        coord_data[struct_key] = {}
        for idx, site in enumerate(structure):
            cn = len(get_neighbors(site, structure))
            min_cn = min(min_cn, cn)
            max_cn = max(max_cn, cn)
            elem_data = coord_data[struct_key].setdefault(
                get_site_symbol(site), {"cn": [], "hover_data": {}}
            )
            elem_data["cn"].append(cn)
            for key in hover_labels:
                elem_data["hover_data"].setdefault(key, []).append(
                    structure.site_properties.get(key, [None] * len(structure))[idx]
                )

    x_range = list(range(int(min_cn), int(max_cn) + 2))

    elements = sorted({elem for struct in coord_data.values() for elem in struct})
    if split_mode == CnSplitMode.by_element:
        n_subplots = len(elements)
    elif split_mode in (CnSplitMode.by_structure, CnSplitMode.by_structure_and_element):
        n_subplots = len(coord_data)
    else:
        if split_mode != CnSplitMode.none:
            raise ValueError(f"Invalid {split_mode=}")
        n_subplots = 1

    n_cols = min(3, n_subplots)
    n_rows = math.ceil(n_subplots / n_cols)

    subplot_kwargs = dict(
        subplot_titles=(
            elements if split_mode == CnSplitMode.by_element else list(coord_data)
        ),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    ) | (subplot_kwargs or {})

    if split_mode != CnSplitMode.none:
        fig = make_subplots(rows=n_rows, cols=n_cols, **subplot_kwargs)
    else:
        fig = go.Figure()

    element_colors = _resolve_element_colors(element_color_scheme)

    row, col = 1, 1
    # use the normalized dict: len() on a raw Structure input counts sites
    is_single_structure = len(coord_data) == 1
    if annotate_bars is True:
        annotate_bars = {}

    bar_kwargs = dict(width=0.8 if bar_mode == "stack" else 0.6, opacity=0.95) | (
        bar_kwargs or {}
    )

    def histogram_bar(
        struct_key: Hashable, elem_symbol: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Bar counts per CN in x_range with matching hover texts."""
        counts = Counter(data["cn"])
        y_vals = [counts.get(cn, 0) for cn in x_range]
        hover_text = [
            create_hover_text(
                struct_key=struct_key,
                elem_symbol=elem_symbol,
                cn=cn,
                count=count,
                hover_data=hover_labels,
                data=data,
                is_single_structure=is_single_structure,
            )
            for cn, count in zip(x_range, y_vals, strict=True)
        ]
        return dict(
            x=x_range,
            y=y_vals,
            text=y_vals,
            textposition="auto",
            hovertext=hover_text,
            hoverinfo="text",
        )

    # Show each element once in the legend across structures.
    legend_elements_shown = set()

    for struct_key, struct_data in coord_data.items():
        if split_mode == CnSplitMode.by_structure:
            combined_data = {
                "cn": [
                    cn for elem_data in struct_data.values() for cn in elem_data["cn"]
                ],
                "hover_data": {
                    key: [
                        value
                        for elem_data in struct_data.values()
                        for value in elem_data["hover_data"][key]
                    ]
                    for key in hover_labels
                },
            }
            fig.add_bar(
                **histogram_bar(struct_key, "", combined_data),
                name=struct_key,
                **bar_kwargs,
                row=row,
                col=col,
            )
        else:
            if split_mode == CnSplitMode.by_element:
                elem_items = [
                    (elem, struct_data[elem])
                    for elem in elements
                    if elem in struct_data
                ]
            else:
                elem_items = list(struct_data.items())
            for elem_symbol, data in elem_items:
                trace_kwargs = histogram_bar(struct_key, elem_symbol, data) | dict(
                    marker_color=element_colors.get(elem_symbol)
                )
                if split_mode == CnSplitMode.by_element:
                    subplot_row, subplot_col = divmod(
                        elements.index(elem_symbol), n_cols
                    )
                    trace_kwargs |= dict(
                        name=f"{struct_key} - {elem_symbol}",
                        legendgroup=struct_key,
                        legendgrouptitle_text=struct_key,
                        row=subplot_row + 1,
                        col=subplot_col + 1,
                    )
                elif split_mode == CnSplitMode.by_structure_and_element:
                    trace_kwargs |= dict(
                        name=elem_symbol,
                        legendgroup=elem_symbol,
                        showlegend=elem_symbol not in legend_elements_shown,
                        row=row,
                        col=col,
                    )
                    legend_elements_shown.add(elem_symbol)
                else:  # No split
                    trace_kwargs["name"] = f"{struct_key} - {elem_symbol}"
                if annotate_bars is not False and split_mode != CnSplitMode.none:
                    trace_kwargs |= dict(text=elem_symbol, textfont=annotate_bars)
                fig.add_bar(**trace_kwargs, **bar_kwargs)

        if split_mode in (
            CnSplitMode.by_structure,
            CnSplitMode.by_structure_and_element,
        ):
            col += 1
            if col > n_cols:
                col = 1
                row += 1

    fig.update_layout(barmode=bar_mode, bargap=0.15, bargroupgap=0.1)
    fig.layout.legend.tracegroupgap = 2  # reduce gap between legend groups

    # Add figure-level axis titles using annotations
    anno_kwargs = dict(font_size=15, xref="paper", yref="paper", showarrow=False)
    fig.add_annotation(  # X-axis title
        text="Coordination Number", x=0.5, y=-0.15, **anno_kwargs
    )
    fig.add_annotation(  # Y-axis title
        text="Count", x=-0.1, y=0.5, textangle=-90, **anno_kwargs
    )
    # Adjust L/B for titles
    fig.layout.margin.update(l=60, b=60, t=fig.layout.margin.t or 30)

    # start x-axis just below the smallest observed CN
    if max_cn - min_cn < 20:
        fig.update_xaxes(tick0=int(min_cn), dtick=1)
    fig.update_yaxes(rangemode="tozero")  # Ensure y_min=0, not <0

    return fig


def coordination_vs_cutoff_line(
    structures: AnyStructure
    | Mapping[str, AnyStructure]
    | Sequence[AnyStructure]
    | pd.Series,
    *,
    strategy: tuple[float, float] | NearNeighbors | type[NearNeighbors] = (1, 5),
    num_points: int = 50,
    element_color_scheme: ElemColorScheme | dict[str, str] = ElemColorScheme.jmol,
    subplot_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a plotly line plot of cumulative coordination numbers vs cutoff distance.

    Args:
        structures (AnyStructure | dict[str, AnyStructure] | Sequence[AnyStructure]): A
            single structure or a dictionary or sequence of structures.
        strategy (tuple[float, float] | NearNeighbors | type[NearNeighbors]):
            Neighbor-finding strategy. Can be one of:
            - tuple[float, float]: (min_cutoff, max_cutoff) range in Angstroms.
            - NearNeighbors: An instance of a NearNeighbors subclass.
            - Type[NearNeighbors]: A NearNeighbors subclass (will be instantiated).
            Defaults to (1, 5) Angstrom range.
        num_points (int): Number of points to calculate between min and max cutoff.
        element_color_scheme (ElemColorScheme | dict[str, str]): Color scheme for
            elements. Can be "jmol", "vesta", or a custom dict.
        subplot_kwargs (dict[str, Any] | None): Additional keyword arguments to pass to
            make_subplots().

    Returns:
        A plotly Figure object containing the line plot.
    """
    struct_dict = normalize_structures(structures)

    # Determine cutoff range based on strategy
    if (
        isinstance(strategy, tuple)
        and len(strategy) == 2
        and isinstance(strategy[0], int | float)
        and isinstance(strategy[1], int | float)
        and not isinstance(strategy[0], bool)
        and not isinstance(strategy[1], bool)
    ):
        cutoff_range = (float(strategy[0]), float(strategy[1]))

    elif isinstance(strategy, NearNeighbors) or (
        isclass(strategy) and issubclass(strategy, NearNeighbors)
    ):
        nn_instance = strategy if isinstance(strategy, NearNeighbors) else strategy()
        if hasattr(nn_instance, "cutoff"):
            max_cutoff = nn_instance.cutoff
        elif hasattr(nn_instance, "distance_cutoffs"):
            distance_cutoffs = nn_instance.distance_cutoffs
            if (
                isinstance(distance_cutoffs, (list, tuple))
                and len(distance_cutoffs) > 1
            ):
                max_cutoff = distance_cutoffs[1]
            else:
                max_cutoff = (
                    distance_cutoffs
                    if isinstance(distance_cutoffs, (int, float))
                    else 5.0
                )
        else:
            raise AttributeError(f"Could not determine cutoff for {nn_instance=}")
        # cast since hasattr narrowing types the cutoff attributes as plain object
        cutoff_range = (0.0, cast("float", max_cutoff))

    else:
        raise TypeError(
            f"Invalid {strategy=}. Expected float, tuple of floats, NearNeighbors "
            "instance, or NearNeighbors subclass."
        )

    cutoffs = np.linspace(cutoff_range[0], cutoff_range[1], num_points)
    neighbor_fns = [normalize_get_neighbors(strategy=cutoff) for cutoff in cutoffs]

    element_colors = _resolve_element_colors(element_color_scheme)

    n_cols = min(3, len(struct_dict))
    n_rows = math.ceil(len(struct_dict) / n_cols)
    subplot_kwargs = dict(
        cols=n_cols,
        rows=n_rows,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[str(key) for key in struct_dict],
    ) | (subplot_kwargs or {})
    fig = make_subplots(**subplot_kwargs)

    for idx, (struct_name, structure) in enumerate(struct_dict.items(), start=1):
        elements = sorted(
            {
                elem_symbol
                for site in structure
                for elem_symbol in get_site_elements(site)
            }
        )

        for element in elements:
            coord_numbers = [
                calculate_average_cn(structure, element, get_neighbors_fn)
                for get_neighbors_fn in neighbor_fns
            ]

            fig.add_scatter(
                x=cutoffs,
                y=coord_numbers,
                mode="lines",
                name=element,
                line=dict(color=element_colors.get(element)),
                legendgroup=struct_name,
                legendgrouptitle_text=struct_name,
                row=(idx - 1) // n_cols + 1,
                col=(idx - 1) % n_cols + 1,
            )

    fig.update_xaxes(title_text="Cutoff Distance [Å]", row=n_rows)
    fig.update_yaxes(title_text="Coordination Number")

    return fig
