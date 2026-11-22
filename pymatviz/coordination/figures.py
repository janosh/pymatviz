"""Coordination number distributions as MatterViz bar and line plot widgets."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import matscikit as mk
import numpy as np

from pymatviz.colors import ELEM_COLORS_JMOL
from pymatviz.coordination.helpers import (
    PYTHON_CN_STRATEGIES,
    CnSplitMode,
    CnStrategy,
    coordination_nums_in_structure,
)
from pymatviz.enums import ElemColorScheme
from pymatviz.structures import normalize_structures, site_elements, str_labels
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.bar_plot import BarPlotWidget, bar_axis_defaults
from pymatviz.widgets.coordination_bar_plot import CoordinationBarPlotWidget
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


def coordination_bar_plot(
    structures: AnyStructure
    | Mapping[str, AnyStructure]
    | Sequence[AnyStructure]
    | pd.Series,
    *,
    strategy: str | CnStrategy = "electroneg_ratio",
    split_mode: CnSplitMode | str = CnSplitMode.by_element,
    element_color_scheme: ElemColorScheme | dict[str, str] | None = None,
    annotate_bars: bool = False,
    **kwargs: Any,
) -> BarPlotWidget | CoordinationBarPlotWidget:
    """Bar chart of how many sites have each coordination number (CN).

    A MatterViz bonding strategy name (``"electroneg_ratio"``, ``"explicit_only"``,
    ``"pair_cutoffs"``) bonds the structures in the browser, where the controls pane
    switches strategy and split mode live. A cutoff radius, ``"crystal_nn"`` or
    ``"voronoi"`` computes the CNs in Python (with matscikit) instead.

    Args:
        structures (AnyStructure | dict[str, AnyStructure] | Sequence[AnyStructure]):
            A single structure or a dict, sequence or pd.Series of structures.
        strategy (str | float): A MatterViz bonding strategy name (computed in the
            browser), or for Python: a cutoff distance in Å, "crystal_nn" (pymatgen's
            CrystalNN) or "voronoi" (VoronoiNN). Defaults to "electroneg_ratio".
        split_mode (CnSplitMode | str): Which bar series to draw:
            "by_element": one series per element, summed over all structures.
            "by_structure": one series per structure.
            "none": one series holding all sites of all structures.
        element_color_scheme (ElemColorScheme | dict[str, str]): Bar colors in
            "by_element" mode (Python strategies only), a scheme or a custom
            ``{element: color}`` dict merged over Jmol colors. Defaults to pastel.
        annotate_bars (bool): Label each non-empty bar with its element symbol.
            Only valid with split_mode="by_element" and a Python strategy.
        **kwargs: Widget traits, e.g. ``mode`` (``"stacked"`` default in Python,
            ``"grouped"`` or ``"overlay"``), ``orientation``, ``x_axis``, ``y_axis``
            (merged over the defaults set here), ``legend``, ``style``.

    Returns:
        CoordinationBarPlotWidget | BarPlotWidget: The coordination histogram,
            computed in the browser or Python.

    Raises:
        ValueError: For Python-only options with a MatterViz strategy, or
            annotate_bars without split_mode="by_element".
    """
    split_mode = CnSplitMode(split_mode)
    struct_dict = normalize_structures(structures)
    if isinstance(strategy, str) and strategy not in PYTHON_CN_STRATEGIES:
        if element_color_scheme is not None or annotate_bars:
            raise ValueError(
                "element_color_scheme and annotate_bars need a Python strategy (cutoff "
                f"radius, 'crystal_nn' or 'voronoi'), got {strategy=}"
            )
        labeled = str_labels(struct_dict)
        return CoordinationBarPlotWidget(
            labeled, strategy=strategy, split_mode=split_mode.value, **kwargs
        )
    if annotate_bars and split_mode != CnSplitMode.by_element:
        raise ValueError(
            f"annotate_bars needs split_mode='by_element', got {split_mode=}"
        )

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
    elem_colors = _element_colors(element_color_scheme or ElemColorScheme.pastel)
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
    return BarPlotWidget(series, **with_defaults(axes | {"mode": "stacked"}, kwargs))


def coordination_vs_cutoff_line(
    structures: AnyStructure
    | Mapping[str, AnyStructure]
    | Sequence[AnyStructure]
    | pd.Series,
    *,
    cutoff_range: tuple[float, float] = (1, 5),
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
        cutoff_range (tuple[float, float]): (min_cutoff, max_cutoff) in Å. Defaults
            to (1, 5).
        num_points (int): Number of points to calculate between min and max cutoff.
        element_color_scheme (ElemColorScheme | dict[str, str]): Color scheme for
            elements. Can be "jmol", "vesta", or a custom dict.
        **kwargs: ScatterPlotWidget traits, e.g. ``x_axis``, ``y_axis`` (merged over
            the defaults set here), ``legend``, ``style``.

    Returns:
        ScatterPlotWidget: one line per structure and element.
    """
    if len(cutoff_range) != 2 or not all(
        isinstance(cutoff, int | float) and not isinstance(cutoff, bool)
        for cutoff in cutoff_range
    ):
        raise TypeError(f"Invalid {cutoff_range=}, expected (min_cutoff, max_cutoff)")
    struct_dict = normalize_structures(structures)
    cutoffs = np.linspace(float(cutoff_range[0]), float(cutoff_range[1]), num_points)
    elem_colors = _element_colors(element_color_scheme)
    multi_struct = len(struct_dict) > 1
    # (n_sites, n_cutoffs) CNs per structure from one neighbor search each
    all_site_cns = mk.coordination_numbers(list(struct_dict.values()), cutoffs)

    series: list[dict[str, Any]] = []
    for struct_idx, ((struct_name, structure), site_cns) in enumerate(
        zip(struct_dict.items(), all_site_cns, strict=True)
    ):
        elements_per_site = site_elements(structure)
        dash = _STRUCT_DASHES[struct_idx % len(_STRUCT_DASHES)]
        for element in sorted(set().union(*elements_per_site)):
            has_elem = [element in elems for elems in elements_per_site]
            entry: dict[str, Any] = {
                "x": cutoffs,
                "y": site_cns[has_elem].mean(axis=0),
                "label": element,
                "markers": "line",
                "line_style": {"color": elem_colors.get(element), "dash": dash},
            }
            if multi_struct:
                entry["legend_group"] = str(struct_name)
            series.append(entry)

    return ScatterPlotWidget(
        series,
        **with_defaults(
            {
                "x_axis": {"label": "Cutoff Distance (Å)"},
                "y_axis": {"label": "Coordination Number"},
            },
            kwargs,
        ),
    )
