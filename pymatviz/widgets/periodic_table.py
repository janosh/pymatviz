"""Interactive periodic table widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import traitlets as tl

from pymatviz.utils import df_ptable
from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets._schema_traits import PeriodicTableTraits
from pymatviz.widgets._traits import color_bar_trait, color_scale_trait, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


# numeric arrays a tile plot series may hold, per series kind
_HIST_KEYS, _XY_KEYS = frozenset({"values"}), frozenset({"x", "y", "color_values"})
_COUNT_KEYS = frozenset({"counts"})
_TILE_PLOT_MODES = ("markers", "line", "line+markers")
# matterviz split layouts per heat value count, the first being its default
SPLIT_LAYOUTS: dict[int, tuple[str, ...]] = {
    2: ("diagonal",),
    3: ("horizontal", "vertical"),
    4: ("quadrant", "triangular"),
}
# ElementTile props the table sets per tile, so tile_props can't override them.
# Mirrors the tile_props Omit<...> list in matterviz
# src/lib/periodic-table/PeriodicTable.svelte. (role, tabindex and style are allowed
# there: the table only overrides role/tabindex on interactive tiles and appends
# tile_props.style to its grid placement.)
_TABLE_TILE_PROPS = frozenset(
    {"active", "backdrop", "element", "href", "label", "segments", "split_layout"}
)
# JSON-settable props tile_props may hold: ElementTile's own (matterviz
# src/lib/element/ElementTile.svelte $props minus _TABLE_TILE_PROPS and callbacks)
# plus HTML attributes it spreads onto the tile element. aria-* and data-* keys pass
# too. Anything else is likely a typo the frontend would silently ignore.
_TILE_PROPS = frozenset(
    {
        "show_symbol",
        "show_number",
        "show_name",
        "symbol_style",
        "text_color",
        "float_fmt",
        "style",
        "class",
        "role",
        "tabindex",
        "title",
    }
)
# matterviz TILE_PLOT_GRID (src/lib/periodic-table/tile-plots.ts): tile plots draw on
# a grid this many units wide, so it caps tile_plot_config.bins
_TILE_PLOT_GRID = 1000


class _ColorScaleCheck(tl.HasTraits):
    """Validates a matterviz ColorScale nested in tile_plot_config on construction."""

    color_scale = color_scale_trait()


def float_array(values: Any, label: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Validate values as a 1D numeric array and return it as float64.

    Raises:
        ValueError: If values isn't a 1D numeric array.
    """
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a 1D numeric array: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D numeric array, got {values!r}")
    return arr


def nullable_floats(values: Any, label: str) -> list[float | None]:
    """1D numeric array as a JSON list with NaN/±inf as None (matterviz skips nulls).

    Raises:
        ValueError: If values isn't a 1D numeric array.
    """
    arr = float_array(values, label)
    out = arr.astype(object)
    out[~np.isfinite(arr)] = None
    return out.tolist()


def normalize_tile_plots(
    tile_plots: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any] | list[dict[str, Any]]] | None:
    """Validate ``tile_plots`` and convert its arrays to JSON lists.

    Each value is one series dict or a list of them: ``{"values"}`` (histogram),
    ``{"counts"}`` (pre-binned histogram: one non-negative count per
    ``tile_plot_config.bins`` bin over an explicit ``x_range``) or
    ``{"x", "y", "color_values"?}`` (markers/line), each with an optional CSS
    ``color``. Arrays may be numpy; NaN and ±inf become None, which matterviz skips.

    Raises:
        TypeError: If a series isn't a dict or its color isn't a string.
        ValueError: For keys that aren't element symbols, series with neither or
            several kinds of keys, unknown keys, x/y/color_values of different
            lengths, counts that aren't finite and non-negative or mixed series
            kinds across tiles.
    """
    if tile_plots is None:
        return None
    if unknown := sorted(map(str, set(tile_plots) - set(df_ptable.index))):
        raise ValueError(f"tile_plots keys must be element symbols, got {unknown}")
    normalized: dict[str, dict[str, Any] | list[dict[str, Any]]] = {}
    kinds: set[str] = set()
    for symbol, plot in tile_plots.items():
        is_list = isinstance(plot, Sequence) and not isinstance(plot, str | Mapping)
        series_list = list(plot) if is_list else [plot]
        out: list[dict[str, Any]] = []
        for idx, series in enumerate(series_list):
            label = f"tile_plots[{symbol!r}]" + (f"[{idx}]" if is_list else "")
            if not isinstance(series, Mapping):
                raise TypeError(f"{label} must be a dict, got {type(series).__name__}")
            arr_keys = set(series) - {"color"}
            if arr_keys == _HIST_KEYS:
                kinds.add("histogram")
            elif arr_keys == _COUNT_KEYS:
                kinds.add("counts")
            elif {"x", "y"} <= arr_keys <= _XY_KEYS:
                kinds.add("x/y")
            else:
                raise ValueError(
                    f"{label} needs keys {{values}}, {{counts}} or "
                    f"{{x, y, color_values?}} (plus "
                    f"optional color), got {sorted(series)}"
                )
            if not isinstance(series.get("color", ""), str):
                raise TypeError(f"{label}.color must be a CSS color string")
            arrays = {  # keep the caller's key order
                key: nullable_floats(val, label)
                for key, val in series.items()
                if key in arr_keys
            }
            if any(count is None or count < 0 for count in arrays.get("counts", ())):
                raise ValueError(
                    f"{label}.counts must be finite and non-negative, got "
                    f"{list(series['counts'])}"
                )
            if len({len(arr) for arr in arrays.values()}) > 1:
                lengths = {key: len(arr) for key, arr in arrays.items()}
                raise ValueError(f"{label}: array lengths differ {lengths}")
            out.append(
                arrays | ({"color": series["color"]} if "color" in series else {})
            )
        normalized[str(symbol)] = out if is_list else out[0]
    if len(kinds) > 1:
        raise ValueError(f"tile_plots mixes {sorted(kinds)} series, use one kind")
    return normalized


def normalize_tile_plot_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate ``tile_plot_config`` against what matterviz accepts.

    Raises:
        tl.TraitError: For unknown keys or invalid values (bins not an integer in
            [1, 1000], unknown mode, ranges that aren't [lo, hi] with lo < hi, ...).
    """
    if config is None:
        return None
    config = normalize_plot_json(dict(config), "PeriodicTable.tile_plot_config")
    allowed = {"bins", "mode", "color", "color_scale", "log_y", "x_range", "y_range"}
    if extra := sorted(set(config) - allowed):
        raise tl.TraitError(f"tile_plot_config: unknown keys {extra}, use {allowed}")
    bins = config.get("bins", 20)
    if (
        isinstance(bins, bool)
        or not isinstance(bins, int)
        or not 1 <= bins <= _TILE_PLOT_GRID
    ):
        raise tl.TraitError(
            f"tile_plot_config.bins must be an int in [1, {_TILE_PLOT_GRID}], {bins=}"
        )
    if config.get("mode", "markers") not in _TILE_PLOT_MODES:
        raise tl.TraitError(f"tile_plot_config.mode must be one of {_TILE_PLOT_MODES}")
    if not isinstance(config.get("log_y", False), bool):
        raise tl.TraitError("tile_plot_config.log_y must be a bool")
    if not isinstance(config.get("color", ""), str):
        raise tl.TraitError("tile_plot_config.color must be a CSS color string")
    if "color_scale" in config:
        _ColorScaleCheck(color_scale=config["color_scale"])
    for key in ("x_range", "y_range"):
        axis_range = config.get(key, [None, None])
        if (
            not isinstance(axis_range, list)
            or len(axis_range) != 2
            or any(
                isinstance(end, bool) or not isinstance(end, int | float | None)
                for end in axis_range
            )
            or (None not in axis_range and not axis_range[0] < axis_range[1])
        ):
            raise tl.TraitError(
                f"tile_plot_config.{key} must be [lo, hi] with lo < hi (None ends fit "
                f"the data), got {axis_range}"
            )
    return config


def _has_heat_numbers(heatmap_values: Any) -> bool:
    """Whether JSON-normalized heatmap_values holds any finite number, numeric
    strings included (matterviz colors those).
    """
    stack = [heatmap_values]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, list | tuple):
            stack.extend(item)
        elif not isinstance(item, bool):
            try:  # None and color strings like "red" aren't numbers
                num = float(item)
            except (TypeError, ValueError):
                continue
            if np.isfinite(num):
                return True
    return False


class PeriodicTableWidget(PeriodicTableTraits, MatterVizWidget):
    """MatterViz widget for interactive periodic table visualizations.

    Supports element heatmaps (scalar or 2-4 split values per tile), custom color
    scales, labels, click interaction and per-element mini plots (``tile_plots``).

    Examples:
        Heatmap by element counts:
        >>> import pymatviz as pmv
        >>> widget = pmv.periodic_table(
        ...     heatmap_values={"Fe": 42, "O": 100, "Li": 15},
        ...     color_scale="interpolateViridis",
        ... )

        Log-spaced colors clamped to a fixed lower bound, color bar hidden:
        >>> widget = pmv.periodic_table(
        ...     heatmap_values={"Fe": 42, "O": 100, "Li": 15},
        ...     color_scale={"type": "log", "value_range": [10, None]},
        ...     color_bar=None,
        ... )

        Histogram per element on shared bins, bars colored by bin position:
        >>> widget = pmv.periodic_table(
        ...     tile_plots={"Fe": {"values": [1, 2, 2, 3]}, "O": {"values": [2, 4]}},
        ...     tile_plot_config={"bins": 10, "color_scale": "interpolateRdBu"},
        ... )
    """

    heatmap_values = tl.Any(allow_none=True).tag(sync=True)
    color_scale = color_scale_trait()
    color_bar = color_bar_trait()
    gap = tl.Unicode(default_value="0.3cqw").tag(sync=True)
    # how tiles with 2-4 heat values are split (None: matterviz default per count, see
    # SPLIT_LAYOUTS); a layout not valid for a tile's count falls back to the default
    split_layout = tl.Enum(
        sorted({layout for layouts in SPLIT_LAYOUTS.values() for layout in layouts}),
        default_value=None,
        allow_none=True,
    ).tag(sync=True)
    # Highlighted element symbols. None (default): tile clicks pin their tooltip and
    # inset plot. A list (link_selection sets one): clicks toggle elements in it
    active_elements = optional_trait(tl.List, trait=tl.Unicode())

    @tl.validate("tile_props")
    def _validate_tile_props(self, proposal: dict[str, Any]) -> Any:
        """Reject props the table sets on each tile itself and unknown props."""
        props = normalize_plot_json(proposal["value"], "PeriodicTable.tile_props")
        keys = set(props or {})
        if clash := sorted(keys & _TABLE_TILE_PROPS):
            raise tl.TraitError(f"tile_props can't set {clash}, the table sets them")
        if unknown := sorted(
            key
            for key in keys - _TILE_PROPS
            if not str(key).startswith(("aria-", "data-"))
        ):
            raise tl.TraitError(
                f"tile_props: unknown keys {unknown}, use {sorted(_TILE_PROPS)} "
                "or aria-*/data-* attributes"
            )
        return props

    @tl.validate("tile_plots")
    def _validate_tile_plots(self, proposal: dict[str, Any]) -> Any:
        """Normalize tile_plots on every assignment (numpy arrays, NaN -> None)."""
        return normalize_tile_plots(proposal["value"])

    @tl.validate("tile_plot_config")
    def _validate_tile_plot_config(self, proposal: dict[str, Any]) -> Any:
        """Validate tile_plot_config, rejecting a color scale beside heat values."""
        config = normalize_tile_plot_config(proposal["value"])
        self._check_one_color_scale(self.heatmap_values, config)
        return config

    @tl.validate("heatmap_values")
    def _validate_heatmap_values(self, proposal: dict[str, Any]) -> Any:
        """Normalize heat values to JSON on every assignment and reject numeric ones
        beside a tile plot color scale. A pandas Series keeps its element index
        (its ``tolist()`` would map values to elements by position).
        """
        import pandas as pd

        value = proposal["value"]
        if isinstance(value, pd.Series):
            value = value.to_dict()
        values = normalize_plot_json(value, "PeriodicTable.heatmap_values")
        self._check_one_color_scale(values, self.tile_plot_config)
        return values

    @staticmethod
    def _check_one_color_scale(heatmap_values: Any, config: Any) -> None:
        """Matterviz shows one color bar, so it throws on numeric heatmap_values
        together with tile_plot_config.color_scale.
        """
        if (config or {}).get("color_scale") and _has_heat_numbers(heatmap_values):
            raise tl.TraitError(
                "PeriodicTable takes numeric heatmap_values or "
                "tile_plot_config.color_scale, not both"
            )

    def __init__(
        self,
        heatmap_values: dict[str, Any] | list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PeriodicTableWidget.

        Args:
            heatmap_values: Element-keyed dict or list of values to color-map
                onto the periodic table. A value can be a list of 2-4 numbers to
                split the tile into segments.
            **kwargs: Widget traits, e.g. ``color_scale``, ``color_bar``, ``labels``,
                ``missing``, ``tile_plots``, ``tile_plot_config``, ``tile_props``.
        """
        kwargs.setdefault(
            "style", "width: 100%; max-width: 900px; aspect-ratio: 18/10;"
        )
        super().__init__(heatmap_values=heatmap_values, **kwargs)
