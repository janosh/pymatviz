"""Shared synced display traits; widget-specific defaults stay on each widget."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import traitlets as tl

from pymatviz.widgets._normalize import (
    materialize_iterator,
    normalize_plot_json,
    with_defaults,
)


if TYPE_CHECKING:
    from collections.abc import Callable


# Enums of traits with non-None defaults (schema-generated traits read theirs from
# widget-model-schema.json): matterviz's ELEMENT_COLOR_SCHEME_NAMES
# (src/lib/constants.ts) and CameraProjection (src/lib/settings.ts)
ELEMENT_COLOR_SCHEMES = ("Vesta", "Jmol", "Alloy", "Pastel", "Muted", "Dark Mode")
CAMERA_PROJECTIONS = ("perspective", "orthographic")
# Mirror matterviz's D3InterpolateName (src/lib/colors/scale-names.ts): the
# d3-scale-chromatic interpolators, e.g. "interpolateViridis" (not plain "viridis")
_D3_NAMES = (
    "Blues", "BrBG", "BuGn", "BuPu", "Cividis", "Cool", "CubehelixDefault", "GnBu",
    "Greens", "Greys", "Inferno", "Magma", "OrRd", "Oranges", "PRGn", "PiYG",
    "Plasma", "PuBu", "PuBuGn", "PuOr", "PuRd", "Purples", "Rainbow", "RdBu",
    "RdGy", "RdPu", "RdYlBu", "RdYlGn", "Reds", "Sinebow", "Spectral", "Turbo",
    "Viridis", "Warm", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd",
)  # fmt: skip
D3_INTERPOLATORS = tuple(f"interpolate{name}" for name in _D3_NAMES)
# Mirror matterviz's ExportFormat (src/lib/io/export.ts)
EXPORT_FORMATS = ("png", "svg", "csv", "json", "md", "tex")
CONTROLS_MODES = ("always", "hover", "never")  # svelte-widgets ControlsConfig.mode


def trait_errors[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """``func(*args, **kwargs)``, its errors re-raised as TraitError. Inside
    ``hold_trait_notifications()`` (which widget constructors use) validators run
    when the batch ends, and traitlets rolls the batch back only for a TraitError:
    any other error (e.g. a KeyError or FileNotFoundError while converting the value)
    would leave the invalid value set and synced.
    """
    try:
        return func(*args, **kwargs)
    except tl.TraitError:
        raise
    except Exception as exc:
        raise tl.TraitError(f"{type(exc).__name__}: {exc}") from exc


def validator(*names: str) -> Callable[[Callable[..., Any]], Any]:
    """``tl.validate`` re-raising any error of the validator as TraitError (see
    ``trait_errors``). An iterator value (e.g. a generator) reaches the validator as
    the list of its items.
    """

    def decorator(func: Callable[..., Any]) -> Any:
        """Wrap func's errors in TraitError."""

        @functools.wraps(func)
        def checked(obj: Any, proposal: tl.Bunch) -> Any:
            """Func's validated value, its errors as TraitError."""
            value = trait_errors(materialize_iterator, proposal["value"])
            return trait_errors(func, obj, tl.Bunch(proposal, value=value))

        return tl.validate(*names)(checked)

    return decorator


# === Index traits ===
# Two-way index traits (trajectory step, histogram series, scatter point, structure
# sites) are checked against the data length Python knows. Python assignments out of
# range raise; frontend updates are clamped (or dropped) instead, since they can race
# a Python data update and a raise would leave the two sides out of sync (ipywidgets
# echoes a value a validator changed back to the frontend). Replacing the data with
# shorter data clamps or resets the dependent index (observers on the data traits).


def from_frontend(widget: Any, name: str) -> bool:
    """Whether trait ``name`` is being set by a frontend state update: ipywidgets'
    ``set_state`` locks the keys it sets in ``_property_lock`` while it sets them.
    """
    return name in getattr(widget, "_property_lock", {})


def index_error(
    widget: Any, name: str, index: Any, n_items: int | None, what: str
) -> IndexError:
    """IndexError naming the widget trait, the bad index and the valid range."""
    if n_items is None:
        valid = "a non-negative int"
    else:
        valid = f"0 to {n_items - 1}" if n_items else f"none: no {what}"
    return IndexError(
        f"{type(widget).__name__}.{name}={index!r} is out of range for "
        f"{'unknown' if n_items is None else n_items} {what} (valid: {valid})"
    )


def checked_position(
    widget: Any, name: str, index: int, n_items: int | None, what: str
) -> int:
    """Validated position index (e.g. a trajectory step) into ``n_items`` items.
    0 always passes, as the position of empty data, and any non-negative index when
    ``n_items`` is None (data Python doesn't hold, e.g. loaded from a ``data_url``).
    Frontend updates out of range are clamped into range.

    Raises:
        IndexError: For an out-of-range index assigned from Python.
    """
    last = None if n_items is None else max(n_items - 1, 0)
    if index >= 0 and (last is None or index <= last):
        return index
    if from_frontend(widget, name):
        return max(0, index if last is None else min(index, last))
    raise index_error(widget, name, index, n_items, what)


def clamped_position(index: int, n_items: int | None) -> int:
    """Position index clamped into range of ``n_items`` items (unchanged for None),
    for observers reacting to data replaced by shorter data.
    """
    return index if n_items is None else min(index, max(n_items - 1, 0))


class JsonFloat(tl.Float):
    """Float trait also taking numpy real scalars (e.g. ``np.float32``, ``np.int64``)
    and rejecting NaN and infinities, which aren't JSON.
    """

    def validate(self, obj: Any, value: Any) -> float:
        """Finite float of value."""
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise tl.TraitError(
                f"The {self.name!r} trait of a {type(obj).__name__} instance must be "
                f"finite, got {value!r}"
            )
        return super().validate(obj, value)


class JsonInt(tl.Int):
    """Int trait also taking numpy integer scalars (e.g. ``np.argmax`` results)."""

    def validate(self, obj: Any, value: Any) -> int:
        """Int of value."""
        if isinstance(value, np.integer):
            value = int(value)
        return super().validate(obj, value)


class JsonBool(tl.Bool):
    """Bool trait also taking numpy booleans (e.g. ``(values > 0).any()``)."""

    def validate(self, obj: Any, value: Any) -> bool:
        """Bool of value."""
        if isinstance(value, np.bool_):
            value = bool(value)
        return super().validate(obj, value)


def _is_stored_value(trait: tl.TraitType, obj: Any, value: Any) -> bool:
    """Whether value is the trait's already validated value being set again:
    ``HasTraits.__init__`` re-sets every constructor kwarg after cross-validation,
    which ran each conversion twice (e.g. 2x the structure dict walks).
    """
    stored = obj._trait_values.get(trait.name)
    return obj._cross_validation_lock and value is stored


class _JsonNormalized(tl.TraitType):
    """Mixin running every assigned value (in the constructor or later) through
    normalize_plot_json (numpy arrays/scalars to lists/floats/ints) before the
    container check and any ``@validator``, unless the widget lists the trait in its
    ``_raw_json_traits`` (e.g. series its validator encodes as binary buffers).
    """

    def _validate(self, obj: Any, value: Any) -> Any:
        """JSON-normalized value, validated by the container trait."""
        if _is_stored_value(self, obj, value):
            return value
        value = materialize_iterator(value)  # raw traits too, before the List check
        mro = type(obj).__mro__
        if not any(self.name in vars(cls).get("_raw_json_traits", ()) for cls in mro):
            label = f"{type(obj).__name__}.{self.name}"
            value = trait_errors(normalize_plot_json, value, label)
        return super()._validate(obj, value)


class JsonDict(_JsonNormalized, tl.Dict):
    """Dict trait whose values are JSON-normalized on every assignment."""


class JsonList(_JsonNormalized, tl.List):
    """List trait whose values are JSON-normalized on every assignment."""


class JsonUnion(_JsonNormalized, tl.Union):
    """Union trait whose values are JSON-normalized on every assignment."""


_JSON_TRAITS = {
    tl.Dict: JsonDict,
    tl.List: JsonList,
    tl.Union: JsonUnion,
    tl.Float: JsonFloat,
    tl.Int: JsonInt,
    tl.Bool: JsonBool,
}


def optional_trait(trait_cls: Any, **kwargs: Any) -> Any:
    """Synced trait defaulting to None so the frontend component default applies.
    Dict, List and Union traits JSON-normalize every assigned value, Float, Int and
    Bool traits take numpy scalars (Float rejects non-finite values).
    """
    trait_cls = _JSON_TRAITS.get(trait_cls, trait_cls)
    return trait_cls(allow_none=True, default_value=None, **kwargs).tag(sync=True)


# JSON container traits a normalized value is checked against
type _OutputTrait = tl.Dict[Any, Any] | tl.List[Any] | tl.Union


class _Normalized(tl.Any):
    """Trait converting every assigned value (in the constructor or later) with a
    normalizer, e.g. structure objects to structure dicts, checked against the
    output trait the frontend expects. None skips both.
    """

    def __init__(self, normalize: Callable[[Any], Any], output: _OutputTrait) -> None:
        """Store the normalizer and output trait, defaulting the trait to None."""
        super().__init__(default_value=None, allow_none=True)
        output.allow_none = True  # None skips it, but its errors say "or None"
        self.normalize, self.output = normalize, output

    def class_init(self, cls: type[tl.HasTraits], name: str | None) -> None:
        """Attach the output trait under this trait's name, so its errors name the
        trait and widget instead of printing the bare trait object.
        """
        super().class_init(cls, name)
        self.output.class_init(cls, name)

    def _validate(self, obj: Any, value: Any) -> Any:
        """Validated value. Unlike traitlets' default, None still runs the widget's
        cross-trait validators (e.g. one-data-source checks), else clearing a trait
        outside ``hold_trait_notifications()`` could leave an invalid state.
        """
        if value is None and not obj._cross_validation_lock:
            return self._cross_validate(obj, value)
        if _is_stored_value(self, obj, value):
            return value
        return super()._validate(obj, value)

    def validate(self, obj: Any, value: Any) -> Any:
        """Normalized value, validated by the output trait. The normalizer's errors
        become TraitError, which rolls back ``hold_trait_notifications()`` batches.
        """
        normalized = trait_errors(self.normalize, materialize_iterator(value))
        return self.output.validate(obj, normalized)


def normalized_trait(
    normalize: Callable[[Any], Any], output: _OutputTrait | None = None
) -> Any:
    """Synced trait passing every assigned value through ``normalize``, whose
    result must be a dict (or match ``output``).
    """
    return _Normalized(normalize, output or tl.Dict()).tag(sync=True)


def color_scale_trait() -> Any:
    """Synced matterviz ``ColorScale``: a d3 interpolator name or a config dict.

    The dict form is matterviz's ``ColorScaleConfig`` with keys ``scheme`` (d3
    interpolator name), ``type`` (``"linear"``, ``"log"``, ``"arcsinh"``),
    ``value_range`` (``[min, max]`` data values mapped to the palette ends, a None
    end fits the data) and ``palette_range`` (fraction of the palette to use).
    None (default) keeps the component's own (viridis for continuous scales).
    """
    return optional_trait(
        tl.Union,
        trait_types=[
            tl.CaselessStrEnum(D3_INTERPOLATORS),
            tl.Dict(per_key_traits={"scheme": tl.CaselessStrEnum(D3_INTERPOLATORS)}),
        ],
    )


def color_bar_trait() -> Any:
    """Synced matterviz ``ColorBar`` props dict: ``{}`` (default) shows the default
    color bar, ``None`` hides it, e.g. ``{"title": "count", "orientation":
    "vertical"}`` customizes it.
    """
    return JsonDict(default_value={}, allow_none=True).tag(sync=True)


class StructureVizTraits(tl.HasTraits):
    """Display options shared by structure-rendering widgets (synced to JS)."""

    show_atoms = JsonBool(default_value=True).tag(sync=True)

    # Cell
    cell_edge_opacity = JsonFloat(0.1).tag(sync=True)
    cell_surface_opacity = JsonFloat(0.05).tag(sync=True)
    cell_edge_width = JsonFloat(1.5).tag(sync=True)

    # Appearance
    color_scheme = tl.CaselessStrEnum(
        values=ELEMENT_COLOR_SCHEMES, default_value="Vesta"
    ).tag(sync=True)

    # UI controls. gizmo: bool or a matterviz GizmoOptions dict
    gizmo = optional_trait(tl.Union, trait_types=[tl.Bool(), tl.Dict()])


class PlotControlsTraits(tl.HasTraits):
    """Two-way ``controls_open`` state of matterviz plot components' control pane."""

    controls_open = JsonBool(default_value=False).tag(sync=True)


class HierarchyPlotTraits(PlotControlsTraits):
    """Native data, coloring, label and zoom traits for hierarchy charts."""

    data = tl.Any(allow_none=True).tag(sync=True)
    # keep the N largest children per parent, bucketing the rest into "Other" (0 = all)
    max_children = optional_trait(tl.Int, min=0)
    # formats offered in the export pane ([] hides it; None = all) + their base filename
    export_formats = optional_trait(tl.List, trait=tl.Enum(EXPORT_FORMATS))
    # two-way synced zoom state: id (string or number) of the cell the view is rooted on
    zoom_root_id = optional_trait(
        tl.Union, trait_types=[tl.Int(), tl.Float(), tl.Unicode()]
    )
    _positional_traits = ("data",)  # node dict or list of root nodes

    @validator("zoom_root_id")
    def _reject_bool_zoom_root_id(self, proposal: tl.Bunch) -> Any:
        """Reject bools, which the Int and Float options would take as 1 and 0."""
        if isinstance(proposal["value"], bool):
            raise TypeError(
                f"zoom_root_id must be a string or number, got {proposal['value']!r}"
            )
        return proposal["value"]

    @validator("data")
    def _normalize_hierarchy_data(self, proposal: dict[str, Any]) -> Any:
        """Normalize data on construction and subsequent assignments. A standalone
        leaf is wrapped in a list: MatterViz hides a single node's depth-0 root,
        which would leave a leaf without a cell.
        """
        value = proposal["value"]
        if isinstance(value, dict) and not value.get("children"):
            value = [value]
        return normalize_plot_json(value, f"{type(self).__name__}.data")


_AXIS_OF_KEY = {"x": "x_axis", "values": "x_axis", "y": "y_axis"}


def df_series(
    data: Any, color: str | None, kwargs: dict[str, Any], **columns: str | None
) -> Any:
    """Series from DataFrame columns (``columns`` maps series keys to column names),
    one per value of the ``color`` column, dropping rows with NaN in any of them.
    Column names become default axis labels in ``kwargs``. Non-DataFrame data passes
    through unchanged.

    Raises:
        TypeError: If columns are named but data is no DataFrame.
        ValueError: If data is a DataFrame but a column name is missing.
    """
    import pandas as pd

    named = {key: col for key, col in columns.items() if col is not None}
    if not isinstance(data, pd.DataFrame):
        if named or color is not None:
            raise TypeError(
                f"columns {[*named.values(), color]} need a DataFrame, got "
                f"{type(data).__name__}"
            )
        return data
    if missing := sorted(columns.keys() - named.keys()):
        raise ValueError(f"name the DataFrame column(s) for {missing}")
    # one column may fill several roles (e.g. color=x), but is selected once
    grouped = color is not None  # "" is a valid column name
    used_cols = list(dict.fromkeys([*named.values(), *([color] if grouped else [])]))
    df_used = data[used_cols].dropna()
    # horizontal bars and boxes put the x column's values on the y axis
    flip = str(kwargs.get("orientation", "")).lower() == "horizontal"
    axes = {"x_axis": "y_axis", "y_axis": "x_axis"} if flip else {}
    axis_labels = {
        axes.get(_AXIS_OF_KEY[key], _AXIS_OF_KEY[key]): {"label": col}
        for key, col in named.items()
    }
    kwargs.update(with_defaults(axis_labels, kwargs))
    groups = (
        df_used.groupby(color, sort=False, observed=True)
        if grouped
        else [(None, df_used)]
    )
    return [
        {key: group[col].to_numpy() for key, col in named.items()}
        | ({"label": str(name)} if grouped else {})
        for name, group in groups
    ]
