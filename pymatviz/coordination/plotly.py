"""Plotly plots of coordination numbers distributions."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from inspect import isclass
from typing import TYPE_CHECKING

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
from pymatviz.utils import normalize_to_dict


if TYPE_CHECKING:
    from typing import Any, Literal

    from pymatgen.core import Structure


def coordination_hist(
    structures: Structure | dict[str, Structure] | Sequence[Structure],
    *,
    strategy: float | NearNeighbors | type[NearNeighbors] = 3.0,
    split_mode: CnSplitMode | str = CnSplitMode.by_element,
    bar_mode: Literal["group", "stack"] = "stack",
    hover_data: Sequence[str] | dict[str, str] | None = None,
    element_color_scheme: ElemColorScheme | dict[str, str] = ElemColorScheme.jmol,
    annotate_bars: bool | dict[str, Any] = False,
    bar_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a plotly histogram of coordination numbers for given structure(s).

    Args:
        structures: A single structure or a dictionary or sequence of structures.
        strategy: Neighbor-finding strategy. Can be one of:
            - float: Cutoff distance for neighbor search in Angstroms.
            - NearNeighbors: An instance of a NearNeighbors subclass.
            - Type[NearNeighbors]: A NearNeighbors subclass (will be instantiated).
            Defaults to 3.0 (Angstroms cutoff).
        split_mode: How to split the data into subplots or color groups.
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
        bar_mode: How to arrange bars at the same coordination number.
            "group": Bars are stacked and grouped side by side.
            "stack": Bars are stacked on top of each other.
        hover_data: Sequence of keys or dict mapping keys to pretty labels for
            additional data to be shown in the hover tooltip. The keys must exist in the
            site properties or properties dict of the structure.
        element_color_scheme: Color scheme for elements. Can be "jmol", "vesta", or a
            custom dict.
        annotate_bars: If True, annotate bars with element symbols when split_mode
            is 'by_element' or 'by_structure_and_element'. If a dict, used as keywords
            for bar annotations, e.g. {"font_size": 12, "font_color": "red"}.
        bar_kwargs: Dictionary of keyword arguments to customize bar appearance.
            These will be passed to go.Bar().

    Returns:
        A plotly Figure object containing the histogram.
    """
    structures = normalize_to_dict(structures)

    # coord_data: coordination numbers and hover data for each structure and element
    coord_data: dict[str, dict[str, Any]] = {}
    min_cn, max_cn = float("inf"), 0  # will be updated in the loop below

    # Process hover_data
    if isinstance(hover_data, Sequence):
        hover_data = {key: key for key in hover_data}
    elif hover_data is None:
        hover_data = {}
    elif not isinstance(hover_data, dict):
        raise TypeError(f"Invalid {hover_data=}")

    get_neighbors = normalize_get_neighbors(strategy)

    for struct_key, structure in structures.items():
        coord_data[struct_key] = {}
        for idx, site in enumerate(structure):
            cn = len(get_neighbors(site, structure))
            min_cn = min(min_cn, cn)
            max_cn = max(max_cn, cn)
            elem_symbol = site.specie.symbol
            if elem_symbol not in coord_data[struct_key]:
                coord_data[struct_key][elem_symbol] = {"cn": [], "hover_data": {}}
            coord_data[struct_key][elem_symbol]["cn"].append(cn)
            for key in hover_data or ():
                if key not in coord_data[struct_key][elem_symbol]["hover_data"]:
                    coord_data[struct_key][elem_symbol]["hover_data"][key] = []
                coord_data[struct_key][elem_symbol]["hover_data"][key].append(
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

    if split_mode != CnSplitMode.none:
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=(
                elements if split_mode == CnSplitMode.by_element else list(coord_data)
            ),
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.03,
            vertical_spacing=0.05,
        )
    else:
        fig = go.Figure()

    if isinstance(element_color_scheme, dict):
        # Merge custom colors with default Jmol colors to get a complete color scheme
        element_colors = ELEM_COLORS_JMOL | element_color_scheme
    elif isinstance(element_color_scheme, ElemColorScheme):
        element_colors = element_color_scheme.color_map
    else:
        raise TypeError(
            f"Invalid {element_color_scheme=}. Must be {', '.join(ElemColorScheme)} "
            f"or a custom dict."
        )

    row, col = 1, 1
    is_single_structure = len(structures) == 1
    if annotate_bars is True:
        annotate_bars = {}

    bar_kwargs = {"width": 0.8 if bar_mode == "stack" else 0.6} | (bar_kwargs or {})

    for struct_key, struct_data in coord_data.items():
        if split_mode == CnSplitMode.by_element:
            for elem_symbol in elements:
                if elem_symbol in struct_data:
                    data = struct_data[elem_symbol]
                    counts = Counter(data["cn"])
                    y = [counts.get(idx, 0) for idx in x_range]

                    hover_text = [
                        create_hover_text(
                            struct_key,
                            elem_symbol,
                            cn,
                            count,
                            hover_data,
                            data,
                            is_single_structure,
                        )
                        for cn, count in zip(x_range, y, strict=False)
                    ]

                    bar_color = element_colors.get(elem_symbol)
                    if isinstance(bar_color, tuple) and len(bar_color) == 3:
                        bar_color = label_rgb(bar_color)

                    trace = go.Bar(
                        x=x_range,
                        y=y,
                        name=f"{struct_key} - {elem_symbol}",
                        text=y,
                        textposition="auto",
                        hovertext=hover_text,
                        hoverinfo="text",
                        marker_color=bar_color,
                        legendgroup=struct_key,
                        legendgrouptitle_text=struct_key,
                        **bar_kwargs,
                    )
                    subplot_idx = elements.index(elem_symbol) + 1
                    row, col = (
                        (subplot_idx - 1) // n_cols + 1,
                        (subplot_idx - 1) % n_cols + 1,
                    )
                    if annotate_bars is not False:
                        trace.update(text=elem_symbol, textfont=annotate_bars)

                    fig.add_trace(trace, row=row, col=col)

        elif split_mode == CnSplitMode.by_structure:
            all_cn = [
                cn for elem_data in struct_data.values() for cn in elem_data["cn"]
            ]
            counts = Counter(all_cn)
            y = [counts.get(idx, 0) for idx in x_range]

            hover_text = [
                create_hover_text(
                    struct_key, "", cn, count, hover_data, {}, is_single_structure
                )
                for cn, count in zip(x_range, y, strict=False)
            ]

            trace = go.Bar(
                x=x_range,
                y=y,
                name=struct_key,
                text=y,
                textposition="auto",
                hovertext=hover_text,
                hoverinfo="text",
                **bar_kwargs,
            )
            fig.add_trace(trace, row=row, col=col)
        elif split_mode == CnSplitMode.by_structure_and_element:
            for elem_symbol, data in struct_data.items():
                counts = Counter(data["cn"])
                y = [counts.get(idx, 0) for idx in x_range]

                hover_text = [
                    create_hover_text(
                        struct_key,
                        elem_symbol,
                        cn,
                        count,
                        hover_data,
                        data,
                        is_single_structure,
                    )
                    for cn, count in zip(x_range, y, strict=False)
                ]

                bar_color = element_colors.get(elem_symbol)
                if isinstance(bar_color, tuple) and len(bar_color) == 3:
                    bar_color = label_rgb(bar_color)

                trace = go.Bar(
                    x=x_range,
                    y=y,
                    name=elem_symbol,
                    text=y,
                    textposition="auto",
                    hovertext=hover_text,
                    hoverinfo="text",
                    marker_color=bar_color,
                    **bar_kwargs,
                )

                if annotate_bars is not False:
                    trace.update(text=elem_symbol, textfont=annotate_bars)

                fig.add_trace(trace, row=row, col=col)
        else:  # No split
            for elem_symbol, data in struct_data.items():
                counts = Counter(data["cn"])
                y = [counts.get(idx, 0) for idx in x_range]

                hover_text = [
                    create_hover_text(
                        struct_key,
                        elem_symbol,
                        cn,
                        count,
                        hover_data,
                        data,
                        is_single_structure,
                    )
                    for cn, count in zip(x_range, y, strict=False)
                ]

                bar_color = element_colors.get(elem_symbol)
                if isinstance(bar_color, tuple) and len(bar_color) == 3:
                    bar_color = label_rgb(bar_color)
                fig.add_bar(
                    x=x_range,
                    y=y,
                    name=f"{struct_key} - {elem_symbol}",
                    text=y,
                    textposition="auto",
                    hovertext=hover_text,
                    hoverinfo="text",
                    marker_color=bar_color,
                    **bar_kwargs,
                )

        if split_mode in (
            CnSplitMode.by_structure,
            CnSplitMode.by_structure_and_element,
        ):
            col += 1
            if col > n_cols:
                col = 1
                row += 1

    fig.update_layout(barmode=bar_mode, bargap=0.15, bargroupgap=0.1)

    # start x-axis just below the smallest observed CN
    fig.update_xaxes(tick0=int(min_cn), dtick=1, range=[min_cn - 0.5, max_cn + 0.5])
    fig.update_yaxes(title="Count", rangemode="tozero")  # Ensure y_min=0, not <0

    # Add title "Coordination Number" to x axes of the last n_cols subplots
    for idx in range(n_subplots - n_cols + 1, n_subplots + 1):
        fig.layout[f"xaxis{idx}"].update(title="Coordination Number")

    # Remove axis labels for non-edge subplots
    if split_mode != CnSplitMode.none:
        for idx in range(1, n_rows * n_cols + 1):
            if idx % n_cols != 1:  # Not in the first column
                fig.update_yaxes(title_text="", row=idx // n_cols + 1, col=idx % n_cols)
            if idx <= (n_rows - 1) * n_cols:  # Not in the last row
                fig.update_xaxes(title_text="", row=idx // n_cols + 1, col=idx % n_cols)

    return fig


def coordination_vs_cutoff_line(
    structures: Structure | dict[str, Structure] | Sequence[Structure],
    *,
    strategy: tuple[float, float] | NearNeighbors | type[NearNeighbors] = (1, 5),
    num_points: int = 50,
    element_color_scheme: ElemColorScheme | dict[str, str] = ElemColorScheme.jmol,
    subplot_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a plotly line plot of cumulative coordination numbers vs cutoff distance.

    Args:
        structures: A single structure or a dictionary or sequence of structures.
        strategy: Neighbor-finding strategy. Can be one of:
            - float: Single cutoff distance for neighbor search in Angstroms.
            - tuple[float, float]: (min_cutoff, max_cutoff) range in Angstroms.
            - NearNeighbors: An instance of a NearNeighbors subclass.
            - Type[NearNeighbors]: A NearNeighbors subclass (will be instantiated).
            Defaults to (1, 5) Angstrom range.
        num_points: Number of points to calculate between min and max cutoff.
        element_color_scheme: Color scheme for elements. Can be "jmol", "vesta", or a
            custom dict.
        subplot_kwargs: Additional keyword arguments to pass to make_subplots().

    Returns:
        A plotly Figure object containing the line plot.
    """
    structures = normalize_to_dict(structures)

    # Determine cutoff range based on strategy
    if (
        isinstance(strategy, tuple)
        and len(strategy) == 2
        and {*map(type, strategy)} <= {int, float}
    ):
        cutoff_range = strategy

    elif isinstance(strategy, NearNeighbors) or (
        isclass(strategy) and issubclass(strategy, NearNeighbors)
    ):
        nn_instance = strategy if isinstance(strategy, NearNeighbors) else strategy()
        if hasattr(nn_instance, "cutoff"):
            max_cutoff = nn_instance.cutoff
        elif hasattr(nn_instance, "distance_cutoffs"):
            max_cutoff = nn_instance.distance_cutoffs[1]
        else:
            raise AttributeError(f"Could not determine cutoff for {nn_instance=}")
        cutoff_range = (0, max_cutoff)

    else:
        raise TypeError(
            f"Invalid {strategy=}. Expected float, tuple of floats, NearNeighbors "
            "instance, or NearNeighbors subclass."
        )

    cutoffs = np.linspace(cutoff_range[0], cutoff_range[1], num_points)

    if isinstance(element_color_scheme, dict):
        element_colors = ELEM_COLORS_JMOL | element_color_scheme
    elif isinstance(element_color_scheme, ElemColorScheme):
        element_colors = element_color_scheme.color_map
    else:
        raise TypeError(
            f"Invalid {element_color_scheme=}. Must be {', '.join(ElemColorScheme)} "
            "or a custom dict."
        )

    n_cols = min(3, len(structures))
    n_rows = math.ceil(len(structures) / n_cols)
    subplot_kwargs = dict(
        cols=n_cols,
        rows=n_rows,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=list(structures),
    ) | (subplot_kwargs or {})
    fig = make_subplots(**subplot_kwargs)

    for idx, (struct_name, structure) in enumerate(structures.items(), start=1):
        elements = sorted({site.specie.symbol for site in structure})

        for element in elements:
            coord_numbers = []
            for cutoff in cutoffs:
                get_neighbors_fn = normalize_get_neighbors(strategy=cutoff)
                avg_cn = calculate_average_cn(structure, element, get_neighbors_fn)
                coord_numbers.append(avg_cn)

            color = element_colors.get(element)
            if isinstance(color, tuple) and len(color) == 3:
                color = label_rgb(color)

            fig.add_scatter(
                x=cutoffs,
                y=coord_numbers,
                mode="lines",
                name=element,
                line=dict(color=color),
                legendgroup=struct_name,
                legendgrouptitle_text=struct_name,
                row=(idx - 1) // n_cols + 1,
                col=(idx - 1) % n_cols + 1,
            )

    fig.update_xaxes(title_text="Cutoff Distance [Å]", row=n_rows)
    fig.update_yaxes(title_text="Coordination Number")

    return fig
