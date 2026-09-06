"""Shared synced display traits; widget-specific defaults stay on each widget."""

from __future__ import annotations

from typing import Any

import traitlets as tl


def optional_trait(trait_cls: Any, **kwargs: Any) -> Any:
    """Synced trait defaulting to None so the frontend component default applies."""
    return trait_cls(allow_none=True, default_value=None, **kwargs).tag(sync=True)


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
    # None defers to the frontend default. An unknown strategy name would crash
    # the renderer (matterviz looks it up in BONDING_STRATEGIES), so validate here.
    bonding_strategy = tl.CaselessStrEnum(
        values=["electroneg_ratio", "solid_angle"], allow_none=True, default_value=None
    ).tag(sync=True)

    # Cell
    cell_edge_opacity = tl.Float(0.1).tag(sync=True)
    cell_surface_opacity = tl.Float(0.05).tag(sync=True)
    cell_edge_color = optional_trait(tl.Unicode)
    cell_surface_color = optional_trait(tl.Unicode)
    cell_edge_width = tl.Float(1.5).tag(sync=True)
    show_cell_vectors = optional_trait(tl.Bool)

    # Appearance
    color_scheme = tl.Unicode("Vesta").tag(sync=True)
    background_color = optional_trait(tl.Unicode)
    background_opacity = optional_trait(tl.Float)

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
