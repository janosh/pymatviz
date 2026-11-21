"""Shared synced display traits; widget-specific defaults stay on each widget."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json


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


def optional_trait(trait_cls: Any, **kwargs: Any) -> Any:
    """Synced trait defaulting to None so the frontend component default applies."""
    return trait_cls(allow_none=True, default_value=None, **kwargs).tag(sync=True)


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
    return tl.Dict(default_value={}, allow_none=True).tag(sync=True)


class StructureVizTraits(tl.HasTraits):
    """Display options shared by structure-rendering widgets (synced to JS)."""

    show_atoms = tl.Bool(default_value=True).tag(sync=True)

    # Cell
    cell_edge_opacity = tl.Float(0.1).tag(sync=True)
    cell_surface_opacity = tl.Float(0.05).tag(sync=True)
    cell_edge_width = tl.Float(1.5).tag(sync=True)

    # Appearance
    color_scheme = tl.CaselessStrEnum(
        values=ELEMENT_COLOR_SCHEMES, default_value="Vesta"
    ).tag(sync=True)

    # UI controls. gizmo: bool or a matterviz GizmoOptions dict
    gizmo = tl.Union([tl.Bool(), tl.Dict()], allow_none=True, default_value=None).tag(
        sync=True
    )


class PlotControlsTraits(tl.HasTraits):
    """Control-pane traits shared by matterviz plot components: two-way
    ``controls_open`` plus HTML attribute dicts for the toggle button and the pane.
    """

    controls_open = tl.Bool(default_value=False).tag(sync=True)


class HierarchyPlotTraits(PlotControlsTraits):
    """Native data, coloring, label and zoom traits for hierarchy charts."""

    data = tl.Any(allow_none=True).tag(sync=True)
    # keep the N largest children per parent, bucketing the rest into "Other" (0 = all)
    max_children = optional_trait(tl.Int, min=0)
    # formats offered in the export pane ([] hides it; None = all) + their base filename
    export_formats = optional_trait(tl.List, trait=tl.Enum(EXPORT_FORMATS))
    # two-way synced zoom state: id of the cell the view is rooted on
    zoom_root_id = optional_trait(tl.Any)

    def __init__(
        self, data: dict[str, Any] | list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> None:
        """Display labeled nodes with leaf values and optional children."""
        super().__init__(data=data, **kwargs)

    @tl.validate("data", "legend", "padding")
    def _normalize_hierarchy_json(self, proposal: dict[str, Any]) -> Any:
        """Normalize JSON values on construction and subsequent assignments. A
        standalone leaf passed as ``data`` is wrapped in a list: MatterViz hides a
        single node's depth-0 root, which would leave a leaf without a cell.
        """
        value = proposal["value"]
        if (
            proposal["trait"].name == "data"
            and isinstance(value, dict)
            and not value.get("children")
        ):
            value = [value]
        return normalize_plot_json(
            value, f"{type(self).__name__}.{proposal['trait'].name}"
        )


class CartesianPlotTraits(PlotControlsTraits):
    """Series, axes, ranges and decorations shared by matterviz 2D plot components
    (``ScatterPlot``, ``BarPlot``, ``Histogram``). Axis/legend/display/padding are
    matterviz config dicts, ranges are ``[min, max]`` lists (None = auto).
    """

    # config kwargs passed through normalize_plot_json (numpy -> JSON) on init
    _JSON_KWARGS: tuple[str, ...] = (
        "x_axis", "x2_axis", "y_axis", "y2_axis", "display", "legend", "ref_lines",
        "padding", "x_range", "x2_range", "y_range", "y2_range",
    )  # fmt: skip
