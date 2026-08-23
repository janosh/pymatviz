"""Structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

import math
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


def _flat_grid(values: Any, dims: Any | None = None) -> tuple[list[float], list[int]]:
    """Flatten a 3D scalar grid to row-major (z-fastest) floats plus ``[nx, ny, nz]``.

    ``values`` is either a nested ``grid[ix][iy][iz]`` array or an already flat
    sequence, in which case ``dims`` is required and must match its length.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:
        if dims is not None and list(dims) != list(arr.shape):
            raise ValueError(f"dims {list(dims)} do not match grid shape {arr.shape}")
        return arr.ravel(order="C").tolist(), list(arr.shape)
    if arr.ndim != 1:
        raise ValueError(f"volumetric grid must be 1D (flat) or 3D, got {arr.ndim}D")
    if dims is None or len(dims) != 3:
        raise ValueError(f"flat volumetric values need dims=[nx, ny, nz], got {dims=}")
    n_points = math.prod(int(dim) for dim in dims)
    if arr.size != n_points:
        raise ValueError(f"{arr.size} flat values do not fill dims {list(dims)}")
    return arr.tolist(), [int(dim) for dim in dims]


def volume_to_dicts(volume: Any) -> list[dict[str, Any]]:
    """Convert volumetric grid data to flat matterviz ``VolumetricData`` payloads.

    Accepts a pymatgen ``VolumetricData`` (``Chgcar``, ``Elfcar``, ``Locpot``, cube
    data; one payload per ``data`` key, ``Chgcar`` values divided by the cell volume
    to e/Å³ like matterviz's own CHGCAR parser) or a dict with ``lattice`` (3x3 rows
    a, b, c), ``origin`` (Cartesian xyz), ``periodic`` (bool), optional ``label``, and
    the grid as either flat ``values`` + ``dims`` or a nested ``grid[ix][iy][iz]``
    list. Always emits flat row-major ``values`` (z fastest, i.e. index
    ``(ix * ny + iy) * nz + iz``) with ``dims = [nx, ny, nz]``, which the renderer
    adopts without re-flattening. Keys it recomputes (``data_range``) or no longer
    reads (``grid``, ``grid_dims``, ``data_order``) are dropped.
    """
    if isinstance(volume, Mapping):
        lattice = np.asarray(volume.get("lattice"), dtype=float)
        origin = np.asarray(volume.get("origin"), dtype=float)
        if lattice.shape != (3, 3) or origin.shape != (3,):
            raise ValueError(
                "volumetric dict needs a 3x3 'lattice' and an xyz 'origin', got "
                f"lattice shape {lattice.shape} and origin shape {origin.shape}"
            )
        if not isinstance(periodic := volume.get("periodic"), bool):
            raise TypeError(
                f"volumetric dict needs a boolean 'periodic', got {periodic}"
            )
        if "grid" in volume:
            values, dims = _flat_grid(volume["grid"], volume.get("grid_dims"))
        elif "values" in volume:
            values, dims = _flat_grid(volume["values"], volume.get("dims"))
        else:
            raise ValueError(
                "volumetric dict needs 'values' + 'dims' or a nested 'grid'"
            )
        payload: dict[str, Any] = {
            "values": values,
            "dims": dims,
            "lattice": lattice.tolist(),
            "origin": origin.tolist(),
            "periodic": periodic,
        }
        for key in ("label", "source", "source_filename"):
            if isinstance(volume.get(key), str):
                payload[key] = volume[key]
        return [payload]

    data = getattr(volume, "data", None)
    structure = getattr(volume, "structure", None)
    if not isinstance(data, Mapping) or structure is None:
        raise TypeError(
            "volumetric_data entries must be dicts or pymatgen VolumetricData, got "
            f"{type(volume).__name__}"
        )
    # CHGCAR stores rho * V_cell; matterviz divides by the volume on parse
    scale = 1 / structure.volume if type(volume).__name__ == "Chgcar" else 1
    lattice = np.asarray(structure.lattice.matrix, dtype=float).tolist()
    return [
        {
            "values": (np.asarray(grid, dtype=float) * scale).ravel(order="C").tolist(),
            "dims": list(np.shape(grid)),
            "lattice": lattice,
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

        Isosurfaces from volumetric data (CHGCAR/ELFCAR/LOCPOT/.cube). Pass pymatgen
        ``VolumetricData`` objects or dicts via ``volumetric_data``, or a file URL via
        ``data_url`` (which auto-adds one layer on load). ``isosurface_settings`` is
        layers-only: every rendered surface is an explicit layer and the default has
        none, so pass at least one layer with ``volumetric_data``:
        >>> StructureWidget(
        ...     volumetric_data=[Chgcar.from_file("CHGCAR")],
        ...     isosurface_settings={
        ...         "layers": [
        ...             {
        ...                 "isovalue": 0.05,  # in the volume's units (e/Å³ for CHGCAR)
        ...                 "color": "#3b82f6",
        ...                 "opacity": 0.6,
        ...                 "visible": True,
        ...                 "show_negative": False,  # also draw the -isovalue surface
        ...                 "negative_color": "#ef4444",
        ...                 # optional: "volume_idx" (defaults to active_volume_idx),
        ...                 # "color_volume_idx", "colormap", "color_range"
        ...             }
        ...         ],
        ...         "wireframe": False,
        ...         "halo": 0,  # fraction of a cell to extend surfaces past the edge
        ...         # optional "display_range": [[lo, hi]] * 3 fractional repeat/clip
        ...         # bounds per lattice axis (VESTA-style)
        ...     },
        ... )
    """

    # display options shared with TrajectoryWidget live in StructureVizTraits

    structure = tl.Dict(allow_none=True).tag(sync=True)
    structure_string = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)

    show_image_atoms = tl.Bool(default_value=True).tag(sync=True)

    # Isosurface (for volumetric data: CHGCAR, ELFCAR, CUBE files)
    # Pass volumetric grid data directly instead of loading from data_url. Entries
    # are pymatgen VolumetricData objects or dicts (see volume_to_dicts); they are
    # normalized on assignment to flat matterviz VolumetricData payloads:
    #   values (flat z-fastest floats), dims ([nx,ny,nz]), lattice ([[ax,ay,az],...]),
    #   origin ([ox,oy,oz]), periodic (bool), label (str, optional).
    volumetric_data = tl.List(default_value=[]).tag(sync=True)
    # matterviz IsosurfaceSettings, layers-only (see class docstring):
    #   {layers: [{isovalue, color, opacity, visible, show_negative, negative_color,
    #              volume_idx?, color_volume_idx?, colormap?, color_range?}],
    #    wireframe: bool, halo: float, display_range?: [[lo, hi] x 3]}
    # Required alongside volumetric_data to see a surface: the frontend default has no
    # layers. Volumes loaded from data_url instead get one auto layer (20% of |max|).
    isosurface_settings = tl.Dict(allow_none=True).tag(sync=True)
    # Two-way synced isosurface view state: which volume the controls edit, the
    # structure vs cross-section slice view, and the slice plane settings.
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
    def _normalize_volumetric_data(
        self, proposal: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Flatten every volume (dict or pymatgen VolumetricData) on assignment."""
        return [
            payload for vol in proposal["value"] for payload in volume_to_dicts(vol)
        ]

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
