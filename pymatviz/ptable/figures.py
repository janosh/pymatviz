"""Periodic table heatmaps and per-element mini plots as MatterViz widgets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from pymatviz.enums import ElemCountMode
from pymatviz.process_data import count_elements
from pymatviz.utils import df_ptable
from pymatviz.widgets.periodic_table import (
    SPLIT_LAYOUTS,
    PeriodicTableWidget,
    nullable_floats,
)


if TYPE_CHECKING:
    from pymatviz.typing import ElemValues


type ColorScale = str | dict[str, Any]
type XYArray = Sequence[float] | np.ndarray[Any, Any]
# (x, y) or (x, y, color_values) of one series, or an array with those rows
type LineData = (
    tuple[XYArray, XYArray] | tuple[XYArray, XYArray, XYArray] | np.ndarray[Any, Any]
)
# one series, or several keyed by name (each name gets its own color)
type ElemData = LineData | Mapping[str, LineData]
type TileLabels = Mapping[str, str] | Callable[[Any], str | None]

# qualitative palette (d3 schemeSet2) for named series of ptable_scatter
_SERIES_COLORS = (
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
)


def _tile_labels(
    labels: TileLabels | None, data: Mapping[str, Any]
) -> dict[str, str] | None:
    """Per-element labels shown under each plot tile's symbol.

    A mapping is used as is, a callable is called with each element's data and
    None or "" results are skipped.
    """
    if labels is None:
        return None
    if isinstance(labels, Mapping):
        return {str(key): str(val) for key, val in labels.items()}
    texts = {symbol: labels(elem_data) for symbol, elem_data in data.items()}
    return {sym: str(text) for sym, text in texts.items() if text not in (None, "")}


def ptable_heatmap(
    values: ElemValues,
    *,
    count_mode: ElemCountMode = ElemCountMode.composition,
    exclude_elements: Sequence[str] = (),
    fill_value: float | None = None,
    heat_mode: Literal["value", "fraction", "percent"] = "value",
    log: bool = False,
    color_scale: ColorScale = "interpolateViridis",
    labels: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> PeriodicTableWidget:
    """Periodic table heatmap of element values or counts.

    Args:
        values (dict[str, float] | pd.Series | list[str | Composition]): Map from
            element symbols to heatmap values, e.g. dict(Fe=2, O=3), or a list of
            compositions whose elements get counted (see ``count_elements``). A named
            numeric pd.Series titles the color bar.
        count_mode (ElemCountMode): How to count elements of compositions, see
            ``count_elements``. Defaults to "composition".
        exclude_elements (list[str]): Elements to leave uncolored and label "excl.",
            e.g. ["O"] when oxygen dominates. Defaults to ().
        fill_value (float | None): Value for elements missing from values. Defaults
            to None, leaving them uncolored (styled by ``missing``).
        heat_mode ("value" | "fraction" | "percent"): Show values as is, or normalized
            to a fraction or percentage of their (finite) sum, making heatmaps of
            different data comparable. "percent" suffixes the color bar title with
            " (%)". Defaults to "value".
        log (bool): Log-spaced colors. Values <= 0 are left uncolored. Can't be
            combined with heat_mode "fraction" or "percent". Defaults to False.
        color_scale (str | dict): matterviz ColorScale: a d3 interpolator name like
            "interpolateViridis" or a dict with keys scheme, type ("linear", "log",
            "arcsinh"), value_range ([min, max] with None ends fitting the data) and
            palette_range. Defaults to "interpolateViridis".
        labels (dict[str, str]): Text by element symbol, shown only on tiles
            without a heat value (matterviz shows the value instead), e.g. for
            missing or excluded elements. Defaults to None ("excl." on excluded
            tiles).
        **kwargs: Passed to PeriodicTableWidget, e.g. color_bar=dict(title="count")
            (None hides it), missing=dict(color="#eee"), style or tile_props, e.g.
            tile_props=dict(float_fmt=".2f") for value labels (d3-format spec,
            default: matterviz's auto format) or dict(show_number=False).

    Returns:
        PeriodicTableWidget: Interactive periodic table heatmap.

    Raises:
        ValueError: For log with heat_mode "fraction"/"percent" or a non-log color
            scale type, an unknown heat_mode, keys that aren't element symbols, no
            finite values or a zero sum to normalize by.
    """
    if log and heat_mode in ("fraction", "percent"):
        raise ValueError(
            "Combining log color scale and heat_mode='fraction'/'percent' unsupported"
        )
    if heat_mode not in ("value", "fraction", "percent"):
        raise ValueError(f"{heat_mode=} must be 'value', 'fraction' or 'percent'")
    is_numeric = isinstance(values, pd.Series) and is_numeric_dtype(values)
    title = values.name if is_numeric else None

    counts = count_elements(values, count_mode, exclude_elements, fill_value)
    if heat_mode != "value":
        total = counts[np.isfinite(counts)].sum()
        if total == 0:
            raise ValueError(f"{heat_mode=} needs values with a nonzero sum")
        counts = counts / total
        if heat_mode == "percent":
            counts *= 100
            title = None if title is None else f"{title} (%)"
    heatmap_values = {
        str(symbol): val
        for symbol, val in zip(
            counts.index, nullable_floats(counts, "values"), strict=True
        )
        if val is not None
    }
    if not heatmap_values:
        raise ValueError(f"ptable_heatmap: no finite values in {values=}")

    if log:
        scale = {"scheme": color_scale} if isinstance(color_scale, str) else color_scale
        if scale.get("type", "log") != "log":
            raise ValueError(
                f"log=True conflicts with color scale type {scale['type']}"
            )
        color_scale = {**scale, "type": "log"}
    if title is not None and (color_bar := kwargs.get("color_bar", {})) is not None:
        kwargs["color_bar"] = {"title": str(title)} | color_bar
    tile_labels = dict.fromkeys(exclude_elements, "excl.") | dict(labels or {})
    return PeriodicTableWidget(
        heatmap_values=heatmap_values,
        color_scale=color_scale,
        labels=tile_labels,
        **kwargs,
    )


def ptable_heatmap_splits(
    data: pd.DataFrame | pd.Series | Mapping[str, XYArray],
    *,
    split_layout: str | None = None,
    color_scale: ColorScale = "interpolateViridis",
    **kwargs: Any,
) -> PeriodicTableWidget:
    """Periodic table heatmap with each element tile split into 2-4 segments.

    All segments share one color scale and color bar. Each segment shows its value.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, Sequence[float]]): Map from
            element symbols to 2-4 values each, e.g. {"Fe": [1, 2], "Co": [3, 4]}.
            For a DataFrame, the index holds element symbols and each column is a
            segment. NaN/±inf segments are left uncolored.
        split_layout (str | None): How tiles are split, by number of values:
            2 "diagonal" (top-left, bottom-right), 3 "horizontal" (strips top to
            bottom) or "vertical" (left to right), 4 "quadrant" (reading order) or
            "triangular" (top, right, bottom, left). Defaults to None, the first
            layout listed for the count.
        color_scale (str | dict): matterviz ColorScale, see ``ptable_heatmap``.
            Defaults to "interpolateViridis".
        **kwargs: Passed to PeriodicTableWidget, e.g. color_bar=dict(title="..."),
            missing=dict(color="#eee") or tile_props=dict(float_fmt=".2f") to
            format value labels (d3-format spec).

    Returns:
        PeriodicTableWidget: Interactive periodic table with split tiles.

    Raises:
        ValueError: If data is empty, has keys that aren't element symbols, not
            all elements have the same number (2, 3 or 4) of values, or
            split_layout doesn't support that number.
    """
    rows = data.T.to_dict("list") if isinstance(data, pd.DataFrame) else dict(data)
    if not rows:
        raise ValueError(f"ptable_heatmap_splits: {data=} must not be empty")
    if unknown := sorted(map(str, set(rows) - set(df_ptable.index))):
        raise ValueError(f"ptable_heatmap_splits: keys must be elements, got {unknown}")

    heatmap_values = {
        str(symbol): nullable_floats(values, f"ptable_heatmap_splits: {symbol}")
        for symbol, values in rows.items()
    }
    n_splits = len(next(iter(heatmap_values.values())))
    if n_splits not in {2, 3, 4}:
        raise ValueError(f"Number of splits {n_splits} must be 2, 3, or 4")
    for symbol, values in heatmap_values.items():
        if len(values) != n_splits:
            raise ValueError(f"{symbol}: got {len(values)} values, expected {n_splits}")
    if split_layout not in (None, *SPLIT_LAYOUTS[n_splits]):
        raise ValueError(
            f"{split_layout=} invalid for {n_splits} values per element, use one of "
            f"{SPLIT_LAYOUTS[n_splits]}"
        )
    return PeriodicTableWidget(
        heatmap_values=heatmap_values,
        split_layout=split_layout,
        color_scale=color_scale,
        **kwargs,
    )


def ptable_hists(
    data: pd.DataFrame | pd.Series | Mapping[str, XYArray],
    *,
    bins: int = 20,
    x_range: tuple[float | None, float | None] | None = None,
    log_y: bool = False,
    color_scale: ColorScale | None = "interpolateRdBu",
    labels: TileLabels | None = None,
    **kwargs: Any,
) -> PeriodicTableWidget:
    """Histogram of each element's values, laid out as a periodic table.

    All histograms share the same bins (numpy edge semantics: each bin includes its
    left edge, the last one also its right edge), so tiles compare at a glance.
    Hover a tile to see its histogram enlarged with axes.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, Sequence[float]]): Map from
            element symbols to values to histogram, e.g. {"Fe": [1, 2, 3], "O": [4]}.
            A DataFrame histograms each column (named by element symbol). NaN/±inf
            values are skipped.
        bins (int): Number of bins (1-1000). Defaults to 20.
        x_range (tuple[float | None, float | None]): Range binned for all tiles,
            values outside it are dropped. A None end fits the data (keep the
            other end on its side of the data, else matterviz finds lo >= hi
            and throws). Defaults to None (min to max of all values).
        log_y (bool): Bar heights on a log(1 + count) scale. Defaults to False.
        color_scale (str | dict | None): matterviz ColorScale coloring bars by bin
            position, shown in a color bar. None draws bars in the text color
            without a color bar. Defaults to "interpolateRdBu" (red to blue).
        labels (dict[str, str] | Callable[[np.ndarray], str | None]): Label
            under each tile's symbol: a dict keyed by element symbol or a callable
            taking an element's values (NaN removed). Defaults to None.
        **kwargs: Passed to PeriodicTableWidget, e.g. color_bar=dict(title="...")
            (None hides it) or style.

    Returns:
        PeriodicTableWidget: Interactive periodic table of histograms.

    Raises:
        ValueError: For keys that aren't element symbols or values that aren't 1D
            numeric arrays.
        TraitError: For bins outside [1, 1000], an x_range with lo >= hi, an
            invalid color_scale, or numeric heatmap_values beside a color_scale.
    """
    columns = data.to_dict("list") if isinstance(data, pd.DataFrame) else dict(data)
    elem_values = {  # finite values only, so labels callables get clean arrays
        str(symbol): np.array(
            [
                val
                for val in nullable_floats(vals, f"ptable_hists: {symbol}")
                if val is not None
            ]
        )
        for symbol, vals in columns.items()
    }
    config: dict[str, Any] = {"bins": bins, "log_y": log_y}
    if x_range is not None:
        config["x_range"] = list(x_range)
    if color_scale is not None:
        config["color_scale"] = color_scale
    return PeriodicTableWidget(
        tile_plots={symbol: {"values": vals} for symbol, vals in elem_values.items()},
        tile_plot_config=config,
        labels=_tile_labels(labels, elem_values),
        **kwargs,
    )


def ptable_scatter(
    data: Mapping[str, ElemData],
    *,
    mode: Literal["markers", "line", "line+markers"] = "markers",
    x_range: tuple[float | None, float | None] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
    color_scale: ColorScale = "interpolateViridis",
    color: str | None = None,
    labels: TileLabels | None = None,
    **kwargs: Any,
) -> PeriodicTableWidget:
    """Scatter or line plot per element, laid out as a periodic table.

    All tiles share x and y axes. Hover a tile to see its plot enlarged with axes.

    Args:
        data (dict[str, ElemData]): Map from element symbols to either
            1. (x, y) or (x, y, color_values) for one series, e.g.
               {"Fe": ([1, 2, 3], [4, 5, 6])}. Numeric color_values color markers
               and line segments through color_scale and show a color bar.
            2. dict of such tuples for several named series per element, e.g.
               {"Fe": {"up": ([1, 2], [3, 4]), "down": ([1, 2], [2, 1])}}. Each
               name gets its own color (unless it has color_values), shared
               across elements. matterviz shows no legend for the names.
            Points with NaN/±inf coordinates are dropped and break lines.
        mode ("markers" | "line" | "line+markers"): Draw points as markers, lines
            through them or both. Defaults to "markers".
        x_range (tuple[float | None, float | None]): x range shared by all tiles,
            a None end fits the data (as in ptable_hists). Points outside are
            dropped. Defaults to None.
        y_range (tuple[float | None, float | None]): Like x_range for y.
        color_scale (str | dict): matterviz ColorScale for color_values. Defaults
            to "interpolateViridis".
        color (str | None): CSS color of series without color_values or a name
            color. Defaults to None (the tile's text color).
        labels (dict[str, str] | Callable[[ElemData], str | None]): Label under
            each tile's symbol: a dict keyed by element symbol or a callable taking
            an element's data. Defaults to None.
        **kwargs: Passed to PeriodicTableWidget, e.g. color_bar=dict(title="T (K)").

    Returns:
        PeriodicTableWidget: Interactive periodic table of scatter/line plots.

    Raises:
        ValueError: For keys that aren't element symbols, series that aren't 2- or
            3-tuples of numeric 1D arrays or arrays of differing lengths.
        TraitError: For an unknown mode, ranges with lo >= hi, an invalid
            color_scale, or numeric heatmap_values beside color_values.
    """
    series_names = sorted(
        {name for elem in data.values() if isinstance(elem, Mapping) for name in elem}
    )
    name_colors = {
        name: _SERIES_COLORS[idx % len(_SERIES_COLORS)]
        for idx, name in enumerate(series_names)
    }

    def to_series(line: LineData, label: str, name: str | None) -> dict[str, Any]:
        """One tile_plots x/y series from an (x, y[, color_values]) tuple."""
        if isinstance(line, np.ndarray):  # rows x, y[, color_values]
            line = tuple(line)
        if not isinstance(line, Sequence) or len(line) not in (2, 3):
            raise ValueError(
                f"{label} must be an (x, y) or (x, y, color_values) tuple, got {line!r}"
            )
        series: dict[str, Any] = dict(
            zip(("x", "y", "color_values"), line, strict=False)
        )
        if name is not None and "color_values" not in series:
            series["color"] = name_colors[name]
        return series

    tile_plots = {
        str(symbol): [
            to_series(line, f"ptable_scatter: {symbol}[{name!r}]", name)
            for name, line in elem.items()
        ]
        if isinstance(elem, Mapping)
        else to_series(elem, f"ptable_scatter: {symbol}", None)
        for symbol, elem in data.items()
    }
    config: dict[str, Any] = {"mode": mode}
    # matterviz rejects a tile color scale beside numeric heatmap_values, so only set
    # it when some series has color_values to map
    if any(
        "color_values" in series
        for plot in tile_plots.values()
        for series in (plot if isinstance(plot, list) else [plot])
    ):
        config["color_scale"] = color_scale
    for key, axis_range in (("x_range", x_range), ("y_range", y_range)):
        if axis_range is not None:
            config[key] = list(axis_range)
    if color is not None:
        config["color"] = color
    return PeriodicTableWidget(
        tile_plots=tile_plots,
        tile_plot_config=config,
        labels=_tile_labels(labels, data),
        **kwargs,
    )
