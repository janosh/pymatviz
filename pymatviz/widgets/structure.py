"""Structure visualization widget for Jupyter notebooks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import traitlets as tl

from pymatviz.widgets._binary import is_binary_envelope, volume_envelope
from pymatviz.widgets._normalize import normalize_plot_json, structure_to_dict
from pymatviz.widgets._schema_traits import StructureTraits
from pymatviz.widgets._traits import (
    JsonBool,
    JsonDict,
    JsonList,
    StructureVizTraits,
    from_frontend,
    index_error,
    normalized_trait,
    validator,
)
from pymatviz.widgets.matterviz import MatterVizWidget


def volume_to_envelopes(volume: Any) -> list[dict[str, Any]]:
    """Convert a volumetric dict (flat ``values`` + ``dims``, nested ``grid``, or a
    ready ``matterviz-binary-v1`` volume envelope) or a pymatgen ``VolumetricData``
    (one entry per ``data`` key, Chgcar values divided by the cell volume to e/Å³ like
    matterviz's CHGCAR parser) to binary volume envelopes (float32 grid, z fastest).
    ``id`` is what isosurface layers reference via ``volume_id``: dicts use their
    ``id``, else their ``label`` (like matterviz's file parsers), pymatgen volumes use
    their ``data`` key (e.g. ``"total"``, ``"diff"``).

    Raises:
        TypeError: If volume is neither a dict nor a pymatgen ``VolumetricData``.
            Lattice/origin/periodic errors are reported by the renderer.
        ValueError: If a dict lacks an id/label, or a flat ``values`` doesn't match
            its ``dims`` or has non-finite values.
    """
    if isinstance(volume, Mapping):
        vol_id = volume.get("id", volume.get("label"))
        if not isinstance(vol_id, str) or not vol_id.strip():
            raise ValueError(
                "volumetric_data dicts need a nonempty string 'id' or 'label' "
                f"(referenced by isosurface layers' volume_id), got {vol_id=}, "
                f"keys={sorted(volume)}"
            )
        if is_binary_envelope(volume, "volume"):  # copy: prefixing rewrites its id
            return [{**volume, "id": vol_id}]
        if "grid" in volume:
            grid = volume["grid"]
        else:
            values = np.asarray(volume.get("values", []))
            dims = np.asarray(volume.get("dims"))
            max_int = 2**53 - 1
            dims_ok = dims.shape == (3,) and dims.dtype.kind in "iuf"
            if dims_ok:  # range checks need numeric dims
                dims_ok = all(0 <= dim <= max_int and int(dim) == dim for dim in dims)
            if not dims_ok or values.size != math.prod(map(int, dims)):
                raise ValueError(
                    f"volumetric_data {vol_id!r} flat values ({values.size}) need dims "
                    f"with that many points, got {dims=}"
                )
            grid = values.reshape(tuple(map(int, dims)))
        keys = {
            "lattice",
            "origin",
            "periodic",
            "label",
            "source",
            "source_filename",
            "units",
            "signed",
        }
        fields = normalize_plot_json(
            {key: volume[key] for key in keys & volume.keys()},
            f"volumetric_data {vol_id!r}",
        )
        return [volume_envelope(grid, **fields, id=vol_id)]

    data, structure = getattr(volume, "data", None), getattr(volume, "structure", None)
    if not isinstance(data, Mapping) or structure is None:
        raise TypeError(
            "volumetric_data entries must be dicts or pymatgen VolumetricData, got "
            f"{type(volume).__name__}"
        )
    # CHGCAR stores rho * V_cell; matterviz divides by the volume on parse
    is_chgcar = type(volume).__name__ == "Chgcar"
    return [
        volume_envelope(
            np.asarray(grid, dtype=float) / structure.volume if is_chgcar else grid,
            id=str(key),
            lattice=structure.lattice.matrix.tolist(),
            origin=[0.0, 0.0, 0.0],
            periodic=True,
            label=str(key),
        )
        for key, grid in data.items()
    ]


def check_layer_volume_ids(
    isosurface_settings: Mapping[str, Any] | None, volumes: list[dict[str, Any]]
) -> None:
    """Raise if an isosurface layer lacks a ``volume_id`` or, for volumes passed from
    Python, references a ``volume_id``/``color_volume_id`` matching no volume
    (matterviz silently drops such layers). Volumes loaded from ``data_url`` are
    parsed in the browser, so their layers are only checked for a ``volume_id``.

    Raises:
        traitlets.TraitError: For a missing or dangling volume reference.
    """
    vol_ids = {volume["id"] for volume in volumes}
    for layer_idx, layer in enumerate((isosurface_settings or {}).get("layers") or []):
        if not isinstance(vol_id := layer.get("volume_id"), str) or not vol_id:
            raise tl.TraitError(
                f"isosurface_settings layer {layer_idx} needs a nonempty string "
                f"volume_id, got {vol_id=}"
            )
        for key in ("volume_id", "color_volume_id") if vol_ids else ():
            if (ref := layer.get(key)) is not None and ref not in vol_ids:
                raise tl.TraitError(
                    f"isosurface_settings layer {layer_idx} {key}={ref!r} matches no "
                    f"volumetric_data id {sorted(vol_ids)}. Swap volumes and layers "
                    "together inside widget.hold_trait_notifications()."
                )


class StructureWidget(StructureVizTraits, StructureTraits, MatterVizWidget):
    """MatterViz widget for visualizing structures in Python notebooks.

    Structure data can be provided as:
    - ``structure``: A structure dict (e.g. pymatgen ``Structure.as_dict()``) or a
      kemio/pymatgen ``Structure``, ASE ``Atoms`` or ``PhonopyAtoms`` object,
      converted on construction and on later assignment.
    - ``structure_string``: A raw file string (CIF, POSCAR, XYZ, etc.)
      parsed on the frontend. Useful when you have the file content but
      not a parsed object. If both are provided, ``structure`` takes
      precedence.

    ``volumetric_data`` entries hold ``memoryview`` buffers, so ``to_dict()`` isn't
    plain JSON. Custom bundles must support matterviz-binary-v1. Contiguous float32
    input grids are borrowed and must stay unmodified.

    Examples:
        Basic usage:
        >>> import numpy as np
        >>> import kemio
        >>> nacl = kemio.Structure(
        ...     np.eye(3) * 5.64, ["Na", "Cl"], [[0] * 3, [0.5] * 3], fractional=True
        ... )
        >>> import pymatviz as pmv
        >>> widget = pmv.structure(structure=nacl)

        With custom visualization options:
        >>> widget = pmv.structure(
        ...     structure=nacl,
        ...     atom_radius=0.8,
        ...     show_bonds="always",
        ...     color_scheme="Jmol",
        ...     style="border-radius: 10px; width: 100%; height: 600px;",
        ... )

        Site vectors (force/magmom/spin) are auto-detected from site properties.
        Single vector keys use element-colored arrows; multiple keys get palette
        colors with per-key toggles, scale sliders, and origin gap control:
        >>> forces = {
        ...     "force_DFT": [[0.1, 0, 0], [-0.1, 0, 0]],
        ...     "force_MLIP": [[0.12, 0.01, 0], [-0.12, -0.01, 0]],
        ... }
        >>> struct_with_forces = kemio.Structure(
        ...     nacl.lattice,
        ...     ["Na", "Cl"],
        ...     nacl.frac_coords,
        ...     fractional=True,
        ...     site_data=forces,
        ... )
        >>> widget = pmv.structure(structure=struct_with_forces)  # auto-detected
        >>> widget = pmv.structure(structure=struct_with_forces, vector_origin_gap=0.3)
        >>> widget = pmv.structure(structure=struct_with_forces, vector_normalize=True)

        Isosurfaces: pymatgen ``VolumetricData`` objects (e.g.
        ``Chgcar.from_file("CHGCAR")``) or dicts plus explicit layers (``isovalue`` in
        the volume's units, e/Å³ for CHGCAR) that pick their volume by ``volume_id``
        (a pymatgen ``data`` key, or a dict's ``id``/``label``):
        >>> grid = np.random.default_rng(seed=0).random((8, 8, 8))
        >>> density = {"id": "total", "grid": grid, "lattice": np.eye(3) * 5.64}
        >>> density |= {"origin": [0, 0, 0], "periodic": True}
        >>> layer = dict(isovalue=0.5, color="#3b82f6", opacity=0.6, visible=True)
        >>> layer |= dict(volume_id="total")
        >>> widget = pmv.structure(
        ...     structure=nacl,
        ...     volumetric_data=[density],
        ...     isosurface_settings={"layers": [layer], "wireframe": False, "halo": 0},
        ... )

        Several pymatgen volumes share ``data`` keys (every ``Chgcar`` has ``"total"``),
        so pass a ``{prefix: volume}`` mapping to namespace ids (and labels) as
        ``"prefix/key"``:
        >>> volumes = {"relaxed": density, "initial": density}
        >>> layer |= dict(volume_id="relaxed/total")
        >>> widget = pmv.structure(
        ...     structure=nacl,
        ...     volumetric_data=volumes,
        ...     isosurface_settings={"layers": [layer]},
        ... )
    """

    # display options shared with TrajectoryWidget live in StructureVizTraits

    structure = normalized_trait(structure_to_dict)
    show_image_atoms = JsonBool(default_value=True).tag(sync=True)
    # Cell shown: the input cell or MatterViz's (moyo) conventional/primitive standard
    # cell, which carries site properties over with vectors rotated into its frame
    cell_type = tl.CaselessStrEnum(
        ["original", "conventional", "primitive"], default_value="original"
    ).tag(sync=True)

    # Isosurface: a list of pymatgen VolumetricData objects or dicts (see
    # volume_to_envelopes), or a {prefix: volume} mapping namespacing ids as
    # "prefix/id".
    # Synced as a list of flat payloads (Any, not List, so mappings reach the
    # validator). Layers follow matterviz IsosurfaceSettings {layers: [{volume_id,
    # isovalue, color, opacity, visible, show_negative, negative_color}], wireframe,
    # halo, display_range?}
    volumetric_data = tl.Any().tag(sync=True)
    # [] means no volumes. None skipped the validator (Any ignores allow_none=False),
    # so plain assignment synced it and broke isosurface_settings checks
    volumetric_data.allow_none = False
    # "structure", "slice" (2D volume slice) or "split" (both side by side)
    display_mode = tl.CaselessStrEnum(
        ["structure", "slice", "split"], default_value="structure"
    ).tag(sync=True)
    slice_settings = JsonDict(default_value={}).tag(sync=True)

    # UI controls
    enable_info_pane = JsonBool(default_value=True).tag(sync=True)

    # Interaction state (two-way synced with the frontend for ipywidgets linking).
    # selected_sites (clicked atoms) and hovered_site_idx (hovered atom) are
    # populated on user interaction (observe them to drive other widgets) and can
    # also be set from Python. highlighted_sites is a Python-driven highlight
    # input (e.g. from a linked plot). Frontend values index the rendered sites,
    # which append image atoms after the structure's own and follow cell_type, so
    # only Python assignments are checked against the structure's sites (with
    # cell_type="original"). A new structure drops indices beyond its sites.
    selected_sites = JsonList(tl.Int(), default_value=[]).tag(sync=True)
    highlighted_sites = JsonList(tl.Int(), default_value=[]).tag(sync=True)

    def _n_own_sites(self, structure: Any) -> int | None:
        """Site count site indices set from Python address: the structure dict's,
        None while unknown (structure_string, data_url, a standardized cell_type, a
        structure object mid-batch before its validator ran).
        """
        if self.cell_type != "original" or not isinstance(structure, dict):
            return None
        return len(structure["sites"])

    @validator("selected_sites", "highlighted_sites")
    def _check_site_indices(self, proposal: dict[str, Any]) -> list[int]:
        """Reject Python-set indices that are negative or beyond the sites."""
        name, indices = proposal["trait"].name, proposal["value"]
        if from_frontend(self, name):
            return indices
        n_sites = self._n_own_sites(self.structure)
        for pos, idx in enumerate(indices):
            if idx < 0 or (n_sites is not None and idx >= n_sites):
                raise index_error(self, f"{name}[{pos}]", idx, n_sites, "sites")
        return indices

    @tl.observe("structure")
    def _drop_stale_site_indices(self, change: dict[str, Any]) -> None:
        """Drop selected/highlighted indices beyond a new structure's sites."""
        if (n_sites := self._n_own_sites(change["new"])) is None:
            return
        with self.hold_trait_notifications():
            for name in ("selected_sites", "highlighted_sites"):
                indices = getattr(self, name)
                if any(idx >= n_sites for idx in indices):
                    setattr(self, name, [idx for idx in indices if idx < n_sites])

    @tl.default("volumetric_data")
    def _default_volumetric_data(self) -> list[dict[str, Any]]:
        """Fresh empty volume list per instance."""
        return []

    @validator("volumetric_data")
    def _normalize_volumetric_data(self, proposal: dict) -> list[dict[str, Any]]:
        """Flatten every volume (dict or pymatgen VolumetricData) on assignment.

        A ``{prefix: volume}`` mapping prefixes each payload's ``id`` and ``label``
        with ``"prefix/"`` so volumes with colliding keys can be loaded together.

        Raises:
            TypeError: If the value is neither a list/tuple nor a mapping, or a
                mapping key is not a nonempty string.
            ValueError: If payload ids are not unique.
        """
        value = proposal["value"]
        if isinstance(value, Mapping):
            payloads = []
            for prefix, volume in value.items():
                if not isinstance(prefix, str) or not prefix.strip():
                    raise TypeError(
                        "volumetric_data mapping keys must be nonempty strings (id "
                        f"prefixes), got {prefix=}"
                    )
                for payload in volume_to_envelopes(volume):
                    payload["id"] = f"{prefix}/{payload['id']}"
                    if "label" in payload:
                        payload["label"] = f"{prefix}/{payload['label']}"
                    payloads.append(payload)
        elif isinstance(value, (list, tuple)):
            payloads = [
                payload for vol in value for payload in volume_to_envelopes(vol)
            ]
        else:
            raise TypeError(
                "volumetric_data must be a list of volumes or a {prefix: volume} "
                f"mapping, got {type(value).__name__}"
            )
        ids = [payload["id"] for payload in payloads]
        if dupes := sorted({vol_id for vol_id in ids if ids.count(vol_id) > 1}):
            raise ValueError(
                f"volumetric_data ids must be unique, got duplicates {dupes} in {ids}. "
                "Pass a {prefix: volume} mapping (e.g. {'relaxed': chgcar_1, "
                "'initial': chgcar_2}) to namespace ids as 'prefix/id', or give dict "
                "volumes distinct 'id's."
            )
        check_layer_volume_ids(self.isosurface_settings, payloads)
        return payloads

    @validator("isosurface_settings")
    def _validate_isosurface_settings(
        self, proposal: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Check that layers reference loaded volumes (cross-validated at the end of
        __init__/hold_trait_notifications, so trait order doesn't matter).
        """
        volumes = self.volumetric_data
        # while volumetric_data still holds the raw input (a {prefix: volume} mapping,
        # pymatgen volumes, label-only dicts), its own validator checks the layers
        # once it has normalized the volumes and their ids
        if all(is_binary_envelope(volume, "volume") for volume in volumes):
            check_layer_volume_ids(proposal["value"], volumes)
        return proposal["value"]

    _positional_traits = ("structure",)
