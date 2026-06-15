"""Structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._traits import StructureVizTraits
from pymatviz.widgets.matterviz import MatterVizWidget


def structure_to_dict(structure: dict[str, Any] | Any | None) -> dict[str, Any] | None:
    """Convert a structure-like object (or dict, or None) to a widget structure dict.

    Passes dicts and ``None`` through unchanged; converts pymatgen ``Structure`` /
    ASE ``Atoms`` (and other ``normalize_structures`` inputs) via ``.as_dict()``.
    """
    if structure is None or isinstance(structure, dict):
        return structure

    from pymatviz.process_data import normalize_structures

    return next(iter(normalize_structures(structure).values())).as_dict()


class StructureWidget(StructureVizTraits, MatterVizWidget):
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

    # display options shared with TrajectoryWidget live in StructureVizTraits

    structure = tl.Dict(allow_none=True).tag(sync=True)
    structure_string = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)

    show_image_atoms = tl.Bool(default_value=True).tag(sync=True)

    # Isosurface (for volumetric data: CHGCAR, ELFCAR, CUBE files)
    # Pass volumetric grid data directly instead of loading from data_url.
    # Each element is a dict matching matterviz VolumetricData:
    #   grid (3D nested list), grid_dims ([nx,ny,nz]), lattice ([[ax,ay,az],...]),
    #   origin ([ox,oy,oz]), data_range ({min,max,abs_max,mean}),
    #   periodic (bool), label (str, optional), data_order (str, optional).
    volumetric_data = tl.List(tl.Dict(), default_value=[]).tag(sync=True)
    isosurface_settings = tl.Dict(allow_none=True).tag(sync=True)

    # UI controls
    enable_info_pane = tl.Bool(default_value=True).tag(sync=True)
    png_dpi = tl.Int(allow_none=True, default_value=None).tag(sync=True)

    # Interaction state (two-way synced with the frontend for ipywidgets linking).
    # selected_sites (clicked atoms) and hovered_site_idx (hovered atom) are
    # populated on user interaction (observe them to drive other widgets) and can
    # also be set from Python. highlighted_sites is a Python-driven highlight
    # input (e.g. from a linked plot).
    selected_sites = tl.List(tl.Int(), default_value=[]).tag(sync=True)
    highlighted_sites = tl.List(tl.Int(), default_value=[]).tag(sync=True)
    hovered_site_idx = tl.Int(allow_none=True, default_value=None).tag(sync=True)

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
        super().__init__(
            widget_type="structure", structure=structure_to_dict(structure), **kwargs
        )
