"""Coordination number distributions as MatterViz bar and line plot widgets."""

from __future__ import annotations

from collections import Counter
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, get_args

import kemio
import numpy as np

from pymatviz.enums import LabelEnum
from pymatviz.structures import normalize_structures, site_elements, site_symbols
from pymatviz.utils import df_ptable, str_labels
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.bar_plot import BarPlotWidget, bar_axis_defaults
from pymatviz.widgets.coordination_bar_plot import CoordinationBarPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from typing_extensions import TypeIs

    from pymatviz.typing import AnyStructure


# Neighbor strategies computed in Python by kemio.coordination_numbers: a cutoff
# in Å, or pymatgen's CrystalNN() and VoronoiNN()
type CnStrategy = float | Literal["crystal_nn", "voronoi"]
# MatterViz bonding strategies, computed in the browser
BrowserCnStrategy = Literal["electroneg_ratio", "explicit_only", "pair_cutoffs"]


def is_browser_strategy(strategy: Any) -> TypeIs[BrowserCnStrategy]:
    """Whether strategy names a MatterViz bonding strategy (computed in the browser)."""
    return strategy in get_args(BrowserCnStrategy)


class CnSplitMode(LabelEnum):
    """How to split coordination number histograms into bar series. Values match
    MatterViz's CoordinationBarPlot ``split_mode``.
    """

    none = "none", "All sites"
    by_element = "by_element", "By element"
    by_structure = "by_structure", "By structure"


# Jmol CSS colors of the elements that have one, for bars and lines colored here
ELEM_COLORS_JMOL: dict[str, str] = {
    symbol: f"rgb({red}, {green}, {blue})"
    for symbol, (red, green, blue) in df_ptable["jmol_color"].items()
    if red >= 0  # -1 marks a missing color
}
# Line dash per structure in coordination_vs_cutoff_line (colors encode elements)
_STRUCT_DASHES = ("none", "6 3", "2 3", "8 3 2 3", "12 4")


def _element_colors(element_colors: Mapping[str, str] | None) -> dict[str, str]:
    """Jmol element CSS colors with custom ``{element: CSS color}`` ones on top.

    Raises:
        TypeError: If a custom color is not a CSS color string.
    """
    if bad := {
        elem: color
        for elem, color in (element_colors or {}).items()
        if not isinstance(color, str)
    }:
        raise TypeError(f"element_colors must map to CSS color strings, got {bad}")
    return ELEM_COLORS_JMOL | dict(element_colors or {})


def coordination_bar_plot(
    structures: AnyStructure
    | Mapping[str, AnyStructure]
    | Sequence[AnyStructure]
    | pd.Series,
    *,
    strategy: BrowserCnStrategy | CnStrategy = "electroneg_ratio",
    split_mode: CnSplitMode | str = CnSplitMode.by_element,
    element_colors: Mapping[str, str] | None = None,
    annotate_bars: bool = False,
    **kwargs: Any,
) -> BarPlotWidget | CoordinationBarPlotWidget:
    """Bar chart of how many sites have each coordination number (CN).

    A MatterViz bonding strategy name (``"electroneg_ratio"``, ``"explicit_only"``,
    ``"pair_cutoffs"``) bonds the structures in the browser, where the controls pane
    switches strategy and split mode live. A cutoff radius, ``"crystal_nn"`` or
    ``"voronoi"`` computes the CNs in Python (with kemio) instead.

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
        element_colors (dict[str, str]): ``{element: CSS color}`` bar colors in
            "by_element" mode (Python strategies only), merged over Jmol colors.
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
    # by_structure series are keyed by label
    labeled = str_labels(normalize_structures(structures))
    if is_browser_strategy(strategy):
        if element_colors is not None or annotate_bars:
            raise ValueError(
                "element_colors and annotate_bars need a Python strategy (cutoff "
                f"radius, 'crystal_nn' or 'voronoi'), got {strategy=}"
            )
        return CoordinationBarPlotWidget(
            labeled, strategy=strategy, split_mode=split_mode.value, **kwargs
        )
    if annotate_bars and split_mode != CnSplitMode.by_element:
        raise ValueError(
            f"annotate_bars needs split_mode='by_element', got {split_mode=}"
        )

    # {series label: CN counts} from one neighbor search over all structures;
    # by_element sums each element (majority one on disordered sites) over them
    groups: dict[str, Counter[int]] = {}
    site_cns = kemio.coordination_numbers(list(labeled.values()), strategy)
    for (label, struct), cns in zip(labeled.items(), site_cns, strict=True):
        if split_mode == CnSplitMode.by_element:
            for symbol, cn in zip(site_symbols(struct), cns.tolist(), strict=True):
                groups.setdefault(symbol, Counter())[cn] += 1
        else:
            if split_mode != CnSplitMode.by_structure:
                label = CnSplitMode.none.label  # noqa: PLW2901
            groups.setdefault(label, Counter()).update(cns.tolist())
    if split_mode == CnSplitMode.by_element:
        groups = dict(sorted(groups.items()))

    observed_cns = {cn for counts in groups.values() for cn in counts}
    if not observed_cns:
        raise ValueError("structures have no sites to count coordination numbers of")
    cn_values = list(range(min(observed_cns), max(observed_cns) + 1))
    elem_colors = _element_colors(element_colors)
    series: list[dict[str, Any]] = []
    for label, counts in groups.items():
        entry: dict[str, Any] = {
            "x": cn_values,
            "y": [counts.get(cn, 0) for cn in cn_values],
            "label": label,
            "bar_width": 0.8,
        }
        if split_mode == CnSplitMode.by_element:
            # elements without a color (e.g. dummy species) take the palette
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
    element_colors: Mapping[str, str] | None = None,
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
        num_points (int): Number of cutoffs from min to max cutoff. Defaults to 50.
        element_colors (dict[str, str]): ``{element: CSS color}`` line colors,
            merged over Jmol colors.
        **kwargs: ScatterPlotWidget traits, e.g. ``x_axis``, ``y_axis`` (merged over
            the defaults set here), ``legend``, ``style``.

    Returns:
        ScatterPlotWidget: one line per structure and element.
    """
    if len(cutoff_range) != 2 or not all(
        isinstance(cutoff, Real) and not isinstance(cutoff, bool)
        for cutoff in cutoff_range
    ):
        raise TypeError(f"Invalid {cutoff_range=}, expected (min_cutoff, max_cutoff)")
    if num_points < 1:
        raise ValueError(f"{num_points=} must be positive")
    struct_dict = str_labels(normalize_structures(structures))
    cutoffs = np.linspace(float(cutoff_range[0]), float(cutoff_range[1]), num_points)
    elem_colors = _element_colors(element_colors)
    multi_struct = len(struct_dict) > 1
    # (n_sites, n_cutoffs) CNs per structure from one neighbor search each
    all_site_cns = kemio.coordination_numbers(list(struct_dict.values()), cutoffs)

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
                entry["legend_group"] = struct_name
            series.append(entry)

    defaults = {
        "x_axis": {"label": "Cutoff Distance (Å)"},
        "y_axis": {"label": "Coordination Number"},
    }
    return ScatterPlotWidget(series, **with_defaults(defaults, kwargs))
