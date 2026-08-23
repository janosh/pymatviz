"""Structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import traitlets as tl

from pymatviz.widgets._traits import StructureVizTraits
from pymatviz.widgets.matterviz import MatterVizWidget


def structure_to_dict(structure: Any) -> dict[str, Any] | None:
    """Convert a structure-like object (or dict, or None) to a widget structure dict.

    Passes dicts and ``None`` through unchanged; converts pymatgen ``Structure`` /
    ASE ``Atoms`` (and other ``normalize_structures`` inputs) via ``.as_dict()``.
    """
    if structure is None or isinstance(structure, dict):
        return structure

    from pymatviz.process_data import normalize_structures

    return next(iter(normalize_structures(structure).values())).as_dict()


def volume_to_dicts(volume: Any) -> list[dict[str, Any]]:
    """Convert a volumetric dict (flat ``values`` + ``dims`` or nested ``grid``) or a
    pymatgen ``VolumetricData`` (one payload per ``data`` key, ``Chgcar`` divided by the
    cell volume to e/Å³ like matterviz's CHGCAR parser) to flat z-fastest matterviz
    payloads. Shape errors (dims, lattice, origin, periodic) are reported by the
    renderer.
    """
    if isinstance(volume, Mapping):
        nested = "grid" in volume
        arr = np.asarray(volume["grid"] if nested else volume.get("values", []), float)
        payload = {"values": arr.ravel().tolist()}
        payload["dims"] = list(arr.shape) if nested else volume.get("dims")
        for key in (
            "lattice",
            "origin",
            "periodic",
            "label",
            "source",
            "source_filename",
        ):
            if key in volume:
                payload[key] = volume[key]
        return [payload]

    data, structure = getattr(volume, "data", None), getattr(volume, "structure", None)
    if not isinstance(data, Mapping) or structure is None:
        raise TypeError(
            "volumetric_data entries must be dicts or pymatgen VolumetricData, got "
            f"{type(volume).__name__}"
        )
    # CHGCAR stores rho * V_cell; matterviz divides by the volume on parse
    scale = 1 / structure.volume if type(volume).__name__ == "Chgcar" else 1
    return [
        {
            "values": (np.asarray(grid, dtype=float) * scale).ravel().tolist(),
            "dims": list(np.shape(grid)),
            "lattice": structure.lattice.matrix.tolist(),
            "origin": [0.0, 0.0, 0.0],
            "periodic": True,
            "label": str(key),
        }
        for key, grid in data.items()
    ]


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

        Isosurfaces: pymatgen ``VolumetricData`` objects or dicts plus explicit layers
        (``isovalue`` in the volume's units, e/Å³ for CHGCAR):
        >>> layer = dict(isovalue=0.05, color="#3b82f6", opacity=0.6, visible=True)
        >>> StructureWidget(
        ...     volumetric_data=[Chgcar.from_file("CHGCAR")],
        ...     isosurface_settings={"layers": [layer], "wireframe": False, "halo": 0},
        ... )
    """

    # display options shared with TrajectoryWidget live in StructureVizTraits

    structure = tl.Dict(allow_none=True).tag(sync=True)
    structure_string = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)

    show_image_atoms = tl.Bool(default_value=True).tag(sync=True)

    # Isosurface: pymatgen VolumetricData objects or dicts (see volume_to_dicts) and
    # matterviz IsosurfaceSettings {layers: [{isovalue, color, opacity, visible,
    # show_negative, negative_color}], wireframe, halo, display_range?}
    volumetric_data = tl.List(default_value=[]).tag(sync=True)
    isosurface_settings = tl.Dict(allow_none=True).tag(sync=True)
    # Two-way synced isosurface view state (active volume, structure/slice view, plane)
    active_volume_idx = tl.Int(default_value=0).tag(sync=True)
    display_mode = tl.CaselessStrEnum(
        ["structure", "slice"], default_value="structure"
    ).tag(sync=True)
    slice_settings = tl.Dict(default_value={}).tag(sync=True)

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

    @tl.validate("volumetric_data")
    def _normalize_volumetric_data(self, proposal: dict) -> list[dict[str, Any]]:
        """Flatten every volume (dict or pymatgen VolumetricData) on assignment."""
        return [pl for vol in proposal["value"] for pl in volume_to_dicts(vol)]

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
