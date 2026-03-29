"""Structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets.matterviz import MatterVizWidget


class StructureWidget(MatterVizWidget):
    """MatterViz widget for visualizing structures in Python notebooks.

    Structure data can be provided as:
    - ``structure``: A parsed dict (from pymatgen/ASE ``.as_dict()``), or
      a pymatgen ``Structure``/ASE ``Atoms`` object (auto-converted).
    - ``structure_string``: A raw file string (CIF, POSCAR, XYZ, etc.)
      parsed on the frontend. Useful when you have the file content but
      not a parsed object. If both are provided, ``structure`` takes
      precedence.

    Examples:
        Basic usage:
        >>> from pymatviz import StructureWidget
        >>> structure_data = {...}  # Structure dictionary from pymatgen/ASE
        >>> widget = StructureWidget(structure=structure_data)

        With custom visualization options:
        >>> widget = StructureWidget(
        ...     structure=structure_data,
        ...     atom_radius=0.8,
        ...     show_bonds=True,
        ...     color_scheme="Jmol",
        ...     style="border-radius: 10px; width: 100%; height: 600px;",
        ... )

        Site vectors (force/magmom/spin) are auto-detected from site properties.
        Single vector keys use element-colored arrows; multiple keys get palette
        colors with per-key toggles, scale sliders, and origin gap control:
        >>> StructureWidget(structure=struct_with_forces)  # auto-detected
        >>> StructureWidget(structure=struct, vector_origin_gap=0.3)  # multi-method
        >>> StructureWidget(structure=struct, vector_normalize=True)  # direction only
    """

    structure = tl.Dict(allow_none=True).tag(sync=True)
    structure_string = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    data_url = tl.Unicode(allow_none=True).tag(sync=True)

    # Atom visualization
    atom_radius = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_atoms = tl.Bool(default_value=True).tag(sync=True)
    show_bonds = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_site_labels = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_site_indices = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_image_atoms = tl.Bool(default_value=True).tag(sync=True)
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
    bonding_strategy = tl.Unicode("nearest_neighbor").tag(sync=True)

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

    # Isosurface (for volumetric data: CHGCAR, ELFCAR, CUBE files)
    # Pass volumetric grid data directly instead of loading from data_url.
    # Each element is a dict matching matterviz VolumetricData:
    #   grid (3D nested list), grid_dims ([nx,ny,nz]), lattice ([[ax,ay,az],...]),
    #   origin ([ox,oy,oz]), data_range ({min,max,abs_max,mean}),
    #   periodic (bool), label (str, optional), data_order (str, optional).
    volumetric_data = tl.List(tl.Dict(), default_value=[]).tag(sync=True)
    isosurface_settings = tl.Dict(allow_none=True).tag(sync=True)

    # UI controls
    show_gizmo = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    auto_rotate = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    enable_info_pane = tl.Bool(default_value=True).tag(sync=True)
    fullscreen_toggle = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    png_dpi = tl.Int(allow_none=True, default_value=None).tag(sync=True)

    def __init__(
        self, structure: dict[str, Any] | Any | None = None, **kwargs: Any
    ) -> None:
        """Initialize the StructureWidget.

        Args:
            structure: Structure data -- a pymatgen ``Structure``, ASE
                ``Atoms``, or a pre-serialized dict. Converted to dict
                internally. Alternatively, pass ``structure_string`` as
                a keyword argument with a raw CIF/POSCAR/XYZ string.
            **kwargs: Additional widget properties (e.g.
                ``structure_string``, ``atom_radius``, ``show_bonds``).
        """
        from pymatviz.process_data import normalize_structures

        if isinstance(structure, dict):
            struct_dict = structure
        elif structure is not None:
            struct_dict = next(iter(normalize_structures(structure).values())).as_dict()
        else:
            struct_dict = None

        super().__init__(widget_type="structure", structure=struct_dict, **kwargs)
