"""Shared synced display traits; widget-specific defaults stay on each widget."""

from __future__ import annotations

import re
from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json


# Mirror matterviz's ELEMENT_COLOR_SCHEME_NAMES (src/lib/constants.ts) and
# BONDING_STRATEGIES (src/lib/structure/bonding.ts). The renderer throws on any other
# name, so traits validate against these (case-insensitively, stored canonical).
ELEMENT_COLOR_SCHEMES = ("Vesta", "Jmol", "Alloy", "Pastel", "Muted", "Dark Mode")
BONDING_STRATEGIES = ("electroneg_ratio", "explicit_only")
# Mirror matterviz's ExportFormat (src/lib/io/export.ts)
EXPORT_FORMATS = ("png", "svg", "csv", "json", "md", "tex")
CONTROLS_MODES = ("always", "hover", "never")  # svelte-widgets ControlsConfig.mode
# Mirror matterviz's StructurePane (src/lib/structure/index.ts) and TrajectoryPane
# (src/lib/trajectory/Trajectory.svelte): the floating panes active_pane can open
STRUCTURE_PANES = ("controls", "info", "export", "flight")
TRAJECTORY_PANES = (
    "controls", "info", "msd", "vacf", "rdf", "spectroscopy", "hotspots",
    "structure-id", "export", "flight",
)  # fmt: skip


def optional_trait(trait_cls: Any, **kwargs: Any) -> Any:
    """Synced trait defaulting to None so the frontend component default applies."""
    return trait_cls(allow_none=True, default_value=None, **kwargs).tag(sync=True)


class D3Interpolator(tl.Unicode):
    """d3-scale-chromatic interpolator name; matterviz throws on any other string."""

    info_text = "a d3 interpolator name like 'interpolateViridis'"

    def validate(self, obj: Any, value: Any) -> str:
        """Reject strings that aren't ``interpolate<Name>``, e.g. plain ``viridis``."""
        value = super().validate(obj, value)
        if not re.fullmatch(r"interpolate[A-Z]\w*", value):
            self.error(obj, value)
        return value


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
            D3Interpolator(),
            tl.Dict(per_key_traits={"scheme": D3Interpolator()}),
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

    # None (not "") so the frontend does not try to fetch an empty URL
    data_url = optional_trait(tl.Unicode)

    # Atom visualization
    atom_radius = optional_trait(tl.Float)
    show_atoms = tl.Bool(default_value=True).tag(sync=True)
    show_bonds = optional_trait(tl.Bool)
    show_site_labels = optional_trait(tl.Bool)
    show_site_indices = optional_trait(tl.Bool)
    same_size_atoms = optional_trait(tl.Bool)

    # Per-property vectors: {property_name: {visible?, color?, scale?}}.
    # The frontend populates omitted configs from site properties.
    vector_configs = optional_trait(tl.Dict)
    vector_scale = optional_trait(tl.Float)
    vector_color = optional_trait(tl.Unicode)
    vector_normalize = optional_trait(tl.Bool)
    vector_uniform_thickness = optional_trait(tl.Bool)
    vector_origin_gap = optional_trait(tl.Float)

    # Bonds
    bond_thickness = optional_trait(tl.Float)
    bond_color = optional_trait(tl.Unicode)
    # None defers to the frontend default (electroneg_ratio)
    bonding_strategy = optional_trait(tl.CaselessStrEnum, values=BONDING_STRATEGIES)

    # Cell
    cell_edge_opacity = tl.Float(0.1).tag(sync=True)
    cell_surface_opacity = tl.Float(0.05).tag(sync=True)
    cell_edge_color = optional_trait(tl.Unicode)
    cell_surface_color = optional_trait(tl.Unicode)
    cell_edge_width = tl.Float(1.5).tag(sync=True)
    show_cell_vectors = optional_trait(tl.Bool)

    # Appearance
    color_scheme = tl.CaselessStrEnum(
        values=ELEMENT_COLOR_SCHEMES, default_value="Vesta"
    ).tag(sync=True)
    background_color = optional_trait(tl.Unicode)
    background_opacity = optional_trait(tl.Float)
    # matterviz CutawaySettings {mode: "off" | "plane" | "slab", axis: 0 | 1 | 2,
    # position, thickness, whole_atoms?} slicing the structure in fractional coords
    cutaway = optional_trait(tl.Dict)

    # UI controls. gizmo: bool or a matterviz GizmoOptions dict
    gizmo = tl.Union([tl.Bool(), tl.Dict()], allow_none=True, default_value=None).tag(
        sync=True
    )
    auto_rotate = optional_trait(tl.Float)
    fullscreen_toggle = optional_trait(tl.Bool)


class PlotControlsTraits(tl.HasTraits):
    """Control-pane traits shared by matterviz plot components: two-way
    ``controls_open`` plus HTML attribute dicts for the toggle button and the pane.
    """

    controls_open = tl.Bool(default_value=False).tag(sync=True)
    controls_toggle_props = optional_trait(tl.Dict)
    controls_pane_props = optional_trait(tl.Dict)


class HierarchyPlotTraits(PlotControlsTraits):
    """Native data, coloring, label and zoom traits for hierarchy charts."""

    _chart_type: str
    data = tl.Any(allow_none=True).tag(sync=True)
    value_mode = optional_trait(
        tl.CaselessStrEnum, values=["leaf-sum", "total", "remainder"]
    )
    sort = optional_trait(
        tl.CaselessStrEnum, values=["descending", "ascending", "none"]
    )
    # fraction by which each depth level lightens its inherited color
    level_lighten = optional_trait(tl.Float)
    # group siblings below this fraction of the total into one "Other" cell (0 = off)
    min_fraction = optional_trait(tl.Float)
    # keep the N largest children per parent, bucketing the rest into "Other" (0 = all)
    max_children = optional_trait(tl.Int, min=0)
    other_label = optional_trait(tl.Unicode)  # label for the aggregated "Other" cell
    max_depth = optional_trait(tl.Int)  # levels shown below the zoom root (0 = all)
    show_labels = optional_trait(tl.Bool)
    # label content (plotly textinfo equivalent)
    label_text = optional_trait(
        tl.CaselessStrEnum,
        values=[
            "label",
            "value",
            "percent",
            "label+value",
            "label+percent",
            "label+parent-percent",
        ],
    )
    zoom_on_click = optional_trait(tl.Bool)
    show_breadcrumbs = optional_trait(tl.Bool)  # clickable ancestor pathbar when zoomed
    legend = optional_trait(tl.Dict)
    show_legend = optional_trait(tl.Bool)  # depth-1 category legend
    value_format = optional_trait(tl.Unicode)  # d3-format string for labels/tooltips
    # chart-edge padding {"t": N, "b": N, "l": N, "r": N} in pixels
    padding = optional_trait(tl.Dict)
    # formats offered in the export pane ([] hides it; None = all) + their base filename
    export_formats = optional_trait(tl.List, trait=tl.Enum(EXPORT_FORMATS))
    export_filename = optional_trait(tl.Unicode)
    fullscreen_toggle = optional_trait(tl.Bool)
    # two-way synced zoom state: id of the cell the view is rooted on
    zoom_root_id = optional_trait(tl.Any)

    def __init__(
        self, data: dict[str, Any] | list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> None:
        """Display labeled nodes with leaf values and optional children."""
        unknown = kwargs.keys() - self.trait_names() - {"version_override", "model_id"}
        if unknown:
            raise TypeError(f"Unknown {type(self).__name__} options: {sorted(unknown)}")
        super().__init__(widget_type=self._chart_type, data=data, **kwargs)

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
            value, f"{self._chart_type}.{proposal['trait'].name}"
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

    series = optional_trait(tl.List)
    x_axis = optional_trait(tl.Dict)
    x2_axis = optional_trait(tl.Dict)
    y_axis = optional_trait(tl.Dict)
    y2_axis = optional_trait(tl.Dict)
    display = optional_trait(tl.Dict)  # grid/zero-line options
    legend = optional_trait(tl.Dict)
    ref_lines = optional_trait(tl.List)
    # plot-area padding {"t": N, "b": N, "l": N, "r": N} in pixels
    padding = optional_trait(tl.Dict)
    # fraction of the data range added as padding (frontend default 0.05)
    range_padding = optional_trait(tl.Float)
    show_legend = optional_trait(tl.Bool)
    x_range = optional_trait(tl.List)
    x2_range = optional_trait(tl.List)
    y_range = optional_trait(tl.List)
    y2_range = optional_trait(tl.List)
