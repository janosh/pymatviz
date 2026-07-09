"""Shared synced traitlets for structure-rendering widgets.

StructureWidget and TrajectoryWidget render the same matterviz structure viewer
and accept the same ~30 display options. This mixin holds the traits they share
so each widget only declares what is specific to it. Note: show_image_atoms is
NOT shared since its default differs between the two widgets.
"""

from __future__ import annotations

from typing import Any

import traitlets as tl


def optional_trait(trait_cls: Any, **kwargs: Any) -> Any:
    """Synced trait defaulting to None so the frontend component default applies."""
    return trait_cls(allow_none=True, default_value=None, **kwargs).tag(sync=True)


class StructureVizTraits(tl.HasTraits):
    """Display options shared by structure-rendering widgets (synced to JS)."""

    data_url = tl.Unicode(allow_none=True).tag(sync=True)

    # Atom visualization
    atom_radius = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_atoms = tl.Bool(default_value=True).tag(sync=True)
    show_bonds = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_site_labels = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_site_indices = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    same_size_atoms = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Site vectors (force, magmom, spin, etc.) -- per-key configuration
    # Keys map to site property names (e.g. "force", "magmom", "force_DFT").
    # Values are dicts with optional keys: visible (bool), color (str|null),
    # scale (float|null). Auto-populated by the frontend when omitted.
    vector_configs = tl.Dict(allow_none=True, default_value=None).tag(sync=True)
    vector_scale = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    vector_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    vector_normalize = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    vector_uniform_thickness = tl.Bool(allow_none=True, default_value=None).tag(
        sync=True
    )
    vector_origin_gap = tl.Float(allow_none=True, default_value=None).tag(sync=True)

    # Bonds
    bond_thickness = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    bond_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    # None defers to the frontend default. An unknown strategy name would crash
    # the renderer (matterviz looks it up in BONDING_STRATEGIES), so validate here.
    bonding_strategy = tl.CaselessStrEnum(
        values=["electroneg_ratio", "solid_angle"], allow_none=True, default_value=None
    ).tag(sync=True)

    # Cell
    cell_edge_opacity = tl.Float(0.1).tag(sync=True)
    cell_surface_opacity = tl.Float(0.05).tag(sync=True)
    cell_edge_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    cell_surface_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    cell_edge_width = tl.Float(1.5).tag(sync=True)
    show_cell_vectors = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Appearance
    color_scheme = tl.Unicode("Vesta").tag(sync=True)
    background_color = tl.Unicode(allow_none=True).tag(sync=True)
    background_opacity = tl.Float(allow_none=True, default_value=None).tag(sync=True)

    # UI controls
    show_gizmo = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    auto_rotate = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    fullscreen_toggle = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
