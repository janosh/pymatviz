"""Structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets.matterviz import MatterVizWidget


class StructureWidget(MatterVizWidget):
    """MatterViz widget for visualizing structures in Python notebooks.

    Examples:
        Basic usage:
        >>> from pymatviz import StructureWidget
        >>> structure_data = {...}  # Structure dictionary from pymatgen/ASE
        >>> widget = StructureWidget(structure=structure_data)
        >>> widget

        With custom visualization options:
        >>> widget = StructureWidget(
        ...     structure=structure_data,
        ...     atom_radius=0.8,
        ...     show_bonds=True,
        ...     color_scheme="Jmol",
        ...     style="border-radius: 10px; width: 100%; height: 600px;",
        ... )
    """

    structure = tl.Dict(allow_none=True).tag(sync=True)
    data_url = tl.Unicode(allow_none=True).tag(sync=True)

    # Atom visualization
    atom_radius = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_atoms = tl.Bool(default_value=True).tag(sync=True)
    show_bonds = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_site_labels = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_image_atoms = tl.Bool(default_value=True).tag(sync=True)
    show_force_vectors = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    same_size_atoms = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Force vectors
    force_vector_scale = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    force_vector_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)

    # Bonds
    bond_thickness = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    bond_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    bonding_strategy = tl.Unicode("nearest_neighbor").tag(sync=True)

    # Cell
    cell_edge_opacity = tl.Float(0.1).tag(sync=True)
    cell_surface_opacity = tl.Float(0.05).tag(sync=True)
    cell_edge_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    cell_surface_color = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    cell_line_width = tl.Float(1.5).tag(sync=True)
    show_vectors = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Appearance
    color_scheme = tl.Unicode("Vesta").tag(sync=True)
    background_color = tl.Unicode(allow_none=True).tag(sync=True)
    background_opacity = tl.Float(allow_none=True, default_value=None).tag(sync=True)

    # Styling
    style = tl.Unicode(allow_none=True).tag(sync=True)  # Custom CSS styles

    # UI controls
    show_controls = tl.Bool(default_value=True).tag(sync=True)
    show_info = tl.Bool(default_value=True).tag(sync=True)
    show_fullscreen_button = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    png_dpi = tl.Int(allow_none=True, default_value=None).tag(sync=True)

    def __init__(
        self, structure: dict[str, Any] | Any | None = None, **kwargs: Any
    ) -> None:
        """Initialize the StructureWidget.

        Args:
            structure: Structure data (dict from pymatgen/ASE .as_dict() or similar)
            **kwargs: Additional widget properties
        """
        from pymatviz.process_data import normalize_structures

        if isinstance(structure, dict):
            struct_dict = structure
        elif structure is not None:
            struct_dict = next(iter(normalize_structures(structure).values())).as_dict()
        else:
            struct_dict = None

        super().__init__(structure=struct_dict, **kwargs)
