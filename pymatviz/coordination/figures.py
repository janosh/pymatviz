"""Coordination number distributions as MatterViz bar and line plot widgets."""

from __future__ import annotations

from collections import Counter
from inspect import isclass
from typing import TYPE_CHECKING, cast

import numpy as np
from pymatgen.analysis.local_env import NearNeighbors

from pymatviz.colors import ELEM_COLORS_JMOL
from pymatviz.coordination.helpers import (
    CnSplitMode,
    calculate_average_cn,
    coordination_nums_in_structure,
    normalize_get_neighbors,
)
from pymatviz.enums import ElemColorScheme
from pymatviz.process_data import normalize_structures
from pymatviz.structure.helpers import get_site_elements
from pymatviz.widgets.bar_plot import BarPlotWidget, bar_axis_defaults
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import pandas as pd

    from pymatviz.typing import AnyStructure

# Line dash per structure in coordination_vs_cutoff_line (colors encode elements)
_STRUCT_DASHES = ("none", "6 3", "2 3", "8 3 2 3", "12 4")


def _element_colors(
    element_color_scheme: ElemColorScheme | dict[str, str],
) -> dict[str, str]:
    """Map element symbols to CSS colors from a scheme enum or a custom dict
    (merged over Jmol). RGB tuples of fractions in [0, 1] become ``rgb(r, g, b)``;
    any other non-string color raises ValueError.
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

    def to_css(color: Any) -> str:
        """Format an RGB fraction tuple as ``rgb(r, g, b)``, pass strings through."""
        if isinstance(color, str):
            return color
        if not (isinstance(color, tuple) and len(color) == 3) or not all(
            0 <= chan <= 1 for chan in color
        ):
            raise ValueError(
                f"{color=} must be a CSS color string or an (r, g, b) tuple of "
                "fractions in [0, 1]"
            )
        red, green, blue = np.rint(np.multiply(color, 255)).astype(int)
        return f"rgb({red}, {green}, {blue})"

    return {elem: to_css(color) for elem, color in color_map.items()}


def coordination_hist(
    structures: AnyStructure
    | Mapping[str, AnyStructure]
    | Sequence[AnyStructure]
    | pd.Series,
    *,
    strategy: float | NearNeighbors | type[NearNeighbors] = 3.0,
    split_mode: CnSplitMode | str = CnSplitMode.by_element,
    element_color_scheme: ElemColorScheme | dict[str, str] = ElemColorScheme.pastel,
    annotate_bars: bool = False,
    **kwargs: Any,
) -> BarPlotWidget:
    """Bar chart of how many sites have each coordination number (CN).

    CNs are computed in Python with a cutoff radius or a pymatgen ``NearNeighbors``
    strategy, neither of which MatterViz has. To bond structures in the frontend
    with MatterViz's ``electroneg_ratio``/``explicit_only`` strategies instead, use
    ``pymatviz.widgets.CoordinationBarPlotWidget`` (same ``split_mode`` names).

    Args:
        structures (AnyStructure | dict[str, AnyStructure] | Sequence[AnyStructure]):
            A single structure or a dict, sequence or pd.Series of structures.
        strategy (float | NearNeighbors | type[NearNeighbors]): Neighbor-finding
            strategy. Can be one of:
            - float: Cutoff distance for neighbor search in Angstroms.
            - NearNeighbors: An instance of a NearNeighbors subclass.
            - Type[NearNeighbors]: A NearNeighbors subclass (will be instantiated).
            Defaults to 3.0 (Angstroms cutoff).
        split_mode (CnSplitMode | str): Which bar series to draw:
            "by_element": one series per element, summed over all structures.
            "by_structure": one series per structure.
            "none": one series holding all sites of all structures.
        element_color_scheme (ElemColorScheme | dict[str, str]): Bar colors in
            "by_element" mode, a scheme or a custom ``{element: color}`` dict merged
            over Jmol colors. Defaults to pastel.
        annotate_bars (bool): Label each non-empty bar with its element symbol.
            Only valid with split_mode="by_element".
        **kwargs: BarPlotWidget traits, e.g. ``mode`` (``"stacked"`` default,
            ``"grouped"`` or ``"overlay"``), ``orientation``, ``x_axis``, ``y_axis``
            (merged over the defaults set here), ``legend``, ``style``.

    Returns:
        BarPlotWidget: The coordination histogram.
    """
    split_mode = CnSplitMode(split_mode)
    if annotate_bars and split_mode != CnSplitMode.by_element:
        raise ValueError(
            f"annotate_bars needs split_mode='by_element', got {split_mode=}"
        )
    struct_dict = normalize_structures(structures)

    # {struct_key: {element: [CN of each site]}}
    struct_cns = {
        key: coordination_nums_in_structure(struct, strategy=strategy)
        for key, struct in struct_dict.items()
    }
    # {series label: CN counts}; by_element sums each element over all structures
    groups: dict[str, Counter[int]] = {}
    if split_mode == CnSplitMode.by_element:
        for elem_cns in struct_cns.values():
            for elem, cns in elem_cns.items():
                groups.setdefault(elem, Counter()).update(cns)
        groups = dict(sorted(groups.items()))
    elif split_mode == CnSplitMode.by_structure:
        for key, elem_cns in struct_cns.items():
            groups[str(key)] = Counter(cn for cns in elem_cns.values() for cn in cns)
    else:
        groups[CnSplitMode.none.label] = Counter(
            cn
            for elem_cns in struct_cns.values()
            for cns in elem_cns.values()
            for cn in cns
        )

    all_cns = {cn for counts in groups.values() for cn in counts}
    cn_values = list(range(min(all_cns), max(all_cns) + 1))
    elem_colors = _element_colors(element_color_scheme)
    series: list[dict[str, Any]] = []
    for label, counts in groups.items():
        entry: dict[str, Any] = {
            "x": cn_values,
            "y": [counts.get(cn, 0) for cn in cn_values],
            "label": label,
            "bar_width": 0.8,
        }
        if split_mode == CnSplitMode.by_element:
            # elements missing from the scheme (e.g. dummy species) take the palette
            if color := elem_colors.get(label):
                entry["color"] = color
            if annotate_bars:
                entry["labels"] = [
                    label if counts.get(cn) else None for cn in cn_values
                ]
        series.append(entry)

    axes = bar_axis_defaults(
        kwargs,
        category={"label": "Coordination Number", "format": "d"},
        value={"label": "Count", "format": "d", "range": [0, None]},
    )
    return BarPlotWidget(series, **axes | {"mode": "stacked"} | kwargs)


def coordination_vs_cutoff_line(
    structures: AnyStructure
    | Mapping[str, AnyStructure]
    | Sequence[AnyStructure]
    | pd.Series,
    *,
    strategy: tuple[float, float] | NearNeighbors | type[NearNeighbors] = (1, 5),
    num_points: int = 50,
    element_color_scheme: ElemColorScheme | dict[str, str] = ElemColorScheme.jmol,
    **kwargs: Any,
) -> ScatterPlotWidget:
    """Line plot of each element's average coordination number vs cutoff distance.

    Lines are colored by element. With several structures, each gets its own
    legend group and line dash pattern.

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
        **kwargs: ScatterPlotWidget traits, e.g. ``x_axis``, ``y_axis`` (merged over
            the defaults set here), ``legend``, ``style``.

    Returns:
        ScatterPlotWidget: one line per structure and element.
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
    elem_colors = _element_colors(element_color_scheme)
    multi_struct = len(struct_dict) > 1

    series: list[dict[str, Any]] = []
    for struct_idx, (struct_name, structure) in enumerate(struct_dict.items()):
        elements = sorted(
            {elem for site in structure for elem in get_site_elements(site)}
        )
        dash = _STRUCT_DASHES[struct_idx % len(_STRUCT_DASHES)]
        for element in elements:
            entry: dict[str, Any] = {
                "x": cutoffs,
                "y": [
                    calculate_average_cn(structure, element, get_neighbors_fn)
                    for get_neighbors_fn in neighbor_fns
                ],
                "label": element,
                "markers": "line",
                "line_style": {"color": elem_colors.get(element), "dash": dash},
            }
            if multi_struct:
                entry["legend_group"] = str(struct_name)
            series.append(entry)

    return ScatterPlotWidget(
        series,
        **{
            "x_axis": {"label": "Cutoff Distance (Å)"}
            | (kwargs.pop("x_axis", None) or {}),
            "y_axis": {"label": "Coordination Number"}
            | (kwargs.pop("y_axis", None) or {}),
        }
        | kwargs,
    )
