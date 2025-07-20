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
        ... )
    """

    structure = tl.Dict(allow_none=True).tag(sync=True)
    data_url = tl.Unicode(allow_none=True).tag(sync=True)

    # Atom properties
    atom_radius = tl.Float(1.0).tag(sync=True)
    show_atoms = tl.Bool(default_value=True).tag(sync=True)
    show_bonds = tl.Bool(default_value=False).tag(sync=True)
    show_site_labels = tl.Bool(default_value=False).tag(sync=True)
    show_image_atoms = tl.Bool(default_value=True).tag(sync=True)
    show_force_vectors = tl.Bool(default_value=False).tag(sync=True)
    same_size_atoms = tl.Bool(default_value=False).tag(sync=True)
    auto_rotate = tl.Float(0.0).tag(sync=True)

    # Force vectors
    force_vector_scale = tl.Float(1.0).tag(sync=True)
    force_vector_color = tl.Unicode("#ff6b6b").tag(sync=True)

    # Bonds
    bond_thickness = tl.Float(0.1).tag(sync=True)
    bond_color = tl.Unicode("#666666").tag(sync=True)
    bonding_strategy = tl.Unicode("nearest_neighbor").tag(sync=True)

    # Cell
    cell_edge_opacity = tl.Float(0.8).tag(sync=True)
    cell_surface_opacity = tl.Float(0.1).tag(sync=True)
    cell_edge_color = tl.Unicode("#333333").tag(sync=True)
    cell_surface_color = tl.Unicode("#333333").tag(sync=True)
    cell_line_width = tl.Float(2.0).tag(sync=True)
    show_vectors = tl.Bool(default_value=True).tag(sync=True)

    # Styling
    color_scheme = tl.Unicode("Vesta").tag(sync=True)
    background_color = tl.Unicode(allow_none=True).tag(sync=True)
    background_opacity = tl.Float(0.1).tag(sync=True)

    # Dimensions
    width = tl.Int(allow_none=True).tag(sync=True)
    height = tl.Int(allow_none=True).tag(sync=True)

    # UI
    show_controls = tl.Bool(default_value=True).tag(sync=True)
    show_info = tl.Bool(default_value=True).tag(sync=True)
    show_fullscreen_button = tl.Bool(default_value=False).tag(sync=True)
    png_dpi = tl.Int(150).tag(sync=True)

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
