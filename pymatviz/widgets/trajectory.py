"""Trajectory visualization widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import traitlets as tl

from pymatviz.widgets._binary import (
    _is_number,
    force_stats,
    frame_input,
    is_binary_envelope,
    trajectory_to_envelope,
)
from pymatviz.widgets._schema_traits import TrajectoryTraits
from pymatviz.widgets._traits import StructureVizTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class TrajectoryWidget(StructureVizTraits, TrajectoryTraits, MatterVizWidget):
    """MatterViz widget for visualizing molecular dynamics and geometry optimization
    trajectories in Python notebooks.

    Accepts trajectory data directly (list of structures, dict with frames) or a
    ``data_url`` the frontend fetches and parses itself (XYZ/extXYZ, ASE .traj,
    XDATCAR, LAMMPS dump, pymatgen ``Trajectory`` JSON; ``.gz``/``.zip`` compressed).
    A binary ``trajectory`` holds ``memoryview`` buffers, so ``to_dict()`` isn't plain
    JSON (``json.dumps`` needs a ``default`` for them).

    Examples:
        Basic usage with list of structures:
        >>> import pymatviz as pmv
        >>> trajectory_data = [...]  # List of structures
        >>> widget = pmv.trajectory(trajectory=trajectory_data)
        >>> widget

        With per-frame properties (scalars are plotted; per-atom ``forces`` of shape
        (n_sites, 3) are drawn as vectors with their max/RMS norms plotted):
        >>> trajectory_with_props = [
        ...     {"structure": struct1, "energy": -1.23, "forces": forces1},
        ...     {"structure": struct2, "energy": -1.25, "forces": forces2},
        ... ]
        >>> widget = pmv.trajectory(trajectory=trajectory_with_props)

        A pymatgen ``Trajectory`` syncs as binary arrays (its ``as_dict()`` as JSON):
        >>> widget = pmv.trajectory(trajectory=Trajectory.from_file("XDATCAR"))

        With custom visualization options:
        >>> widget = pmv.trajectory(
        ...     trajectory=trajectory_data,
        ...     display_mode="structure+plot",
        ...     plot_type="distribution",
        ...     layout="horizontal",
        ...     show_controls=True,
        ...     auto_play=True,
        ...     style="height: 600px; border: 2px solid blue;",
        ... )

        With local file path (automatically detected and loaded):
        >>> widget = pmv.trajectory(data_url="path/to/trajectory.xyz")
        >>> widget = pmv.trajectory(data_url="path/to/trajectory.h5")

        LAMMPS dumps only carry integer atom types; name them with atom_type_mapping:
        >>> widget = pmv.trajectory(
        ...     data_url="path/to/dump.lammpstrj", atom_type_mapping={1: "Si", 2: "O"}
        ... )
    """

    trajectory = tl.Any(default_value=None, allow_none=True).tag(sync=True)
    current_step_idx = tl.Int(0).tag(sync=True)

    # Layout
    layout = tl.CaselessStrEnum(
        ["auto", "horizontal", "vertical"], default_value="auto"
    ).tag(sync=True)
    # Visible panes; "auto" hides visually flat plots. Two-way synced.
    display_mode = tl.CaselessStrEnum(
        ["auto", "structure+plot", "structure", "plot"], default_value="auto"
    ).tag(sync=True)
    # Plot pane content, independent of display_mode. Two-way synced.
    plot_type = tl.CaselessStrEnum(
        ["time-series", "distribution"], default_value="time-series"
    ).tag(sync=True)
    # Data inspector sidebar (frame attributes, per-frame and per-atom tables). Two-way
    # synced, so closing it in the viewer reads back as False.
    data_inspector_open = tl.Bool(default_value=True).tag(sync=True)

    step_labels = tl.Union(
        [tl.Int(), tl.List()], allow_none=True, default_value=None
    ).tag(sync=True)

    def __init__(self, trajectory: Any = None, **kwargs: Any) -> None:
        """Initialize the TrajectoryWidget.

        Args:
            trajectory: Trajectory data in one of these formats:
                - dict with 'frames' key (matterviz format)
                - pymatgen ``Trajectory`` or its ``as_dict()`` (parsed by matterviz)
                - list of structures (pymatgen Structure, ASE Atoms, PhonopyAtoms, etc.)
                - list of dicts with properties: [
                    {"structure": struct, "energy": 1.23, ...}, ...
                ]
                Per-atom forces under a frame's ``force``/``forces`` key (or an ASE
                calculator's results) move to site ``properties.force`` and add
                ``force_max``/``force_norm`` to the frame metadata, the layout
                matterviz's own parsers produce.
                Structure sequences and pymatgen Trajectories sync as a
                ``matterviz-binary-v1`` envelope when representable. Custom widget
                bundles must support this format. Six-component or 3x3 frame
                stresses travel as float64 ``frame_stress`` (order and units kept).
                Serialized trajectory dicts stay JSON.
            **kwargs: Additional widget properties
        """
        super().__init__(trajectory=trajectory, **kwargs)

    @tl.validate("trajectory")
    def _validate_trajectory(self, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize constructor data and later assignments before syncing to JS."""
        from pymatviz.widgets._normalize import normalize_plot_json

        normalized = self._normalize_trajectory(proposal["value"])
        if normalized is not None and is_binary_envelope(normalized, "trajectory"):
            return normalized
        return normalize_plot_json(normalized, "trajectory")

    def _to_structure_dict(self, structure_input: Any) -> dict[str, Any]:
        """Convert a structure-like frame to its JSON representation."""
        from pymatviz.structures import structure_dict

        return structure_dict(structure_input)

    @staticmethod
    def _attach_forces(frame: dict[str, Any], frame_idx: int) -> dict[str, Any]:
        """Move per-atom forces from frame metadata (``force`` or ``forces``) to each
        site's ``properties.force`` (MatterViz draws site vectors, not frame
        metadata ones) and record ``force_max``/``force_norm`` (max and RMS of the
        per-atom force norms, like matterviz's parsers) from the sites'
        ``force``/``forces`` vectors unless the metadata already has them. Returns
        a new frame; inputs are not mutated.

        Raises:
            ValueError: If both ``force`` and ``forces`` are given, or the forces are
                not an (n_sites, 3) array.
        """
        metadata = dict(frame.get("metadata") or {})
        sites = frame["structure"].get("sites", [])
        if "force" in metadata and "forces" in metadata:
            raise ValueError(
                f"trajectory frame {frame_idx} metadata has both 'force' and "
                "'forces'; keep one"
            )
        if (force_key := "forces" if "forces" in metadata else "force") in metadata:
            raw_forces = metadata.pop(force_key)
            forces = np.asarray(raw_forces, dtype=float)
            if forces.shape != (len(sites), 3):
                raise ValueError(
                    f"trajectory frame {frame_idx} {force_key!r} must be per-atom with "
                    f"shape (n_sites, 3) = ({len(sites)}, 3), got {forces.shape}"
                )
            sites = [
                {**site, "properties": {**site.get("properties", {}), "force": vec}}
                for site, vec in zip(sites, forces.tolist(), strict=True)
            ]
            frame = {**frame, "structure": {**frame["structure"], "sites": sites}}

        for key in ("force", "forces"):
            site_vectors = [site.get("properties", {}).get(key) for site in sites]
            if not sites or any(vec is None for vec in site_vectors):
                continue
            vectors = np.asarray(site_vectors, dtype=float)
            if vectors.shape != (len(sites), 3):
                raise ValueError(
                    f"trajectory frame {frame_idx} site property {key!r} must be one "
                    f"3-vector per site, got shape {vectors.shape}"
                )
            for name, value in force_stats(vectors).items():
                metadata.setdefault(name, float(value))
            break
        if metadata:
            return {**frame, "metadata": metadata}
        return {key: val for key, val in frame.items() if key != "metadata"}

    @staticmethod
    def _complete_structure_fields(structure: dict[str, Any]) -> dict[str, Any]:
        """Fill what matterviz's JSON reader does not default itself: a lattice
        ``matrix`` from cell parameters, ``{element, occu}`` species from bare symbols,
        site ``label`` and ``properties``. Schema errors are reported by the renderer.
        """
        completed = dict(structure)
        lattice = structure.get("lattice")
        cell_params = ("a", "b", "c", "alpha", "beta", "gamma")
        if (
            isinstance(lattice, dict)
            and "matrix" not in lattice
            and set(cell_params) <= lattice.keys()
        ):
            import matscikit as mk

            # pymatgen's orientation (c along z) for Cartesian xyz site coords
            matrix = mk.lattice_from_parameters(
                *(float(lattice[key]) for key in cell_params), c_along_z=True
            )
            completed["lattice"] = {**lattice, "matrix": matrix.tolist()}
        sites = []
        for site_idx, site in enumerate(structure.get("sites", [])):
            species = [
                {"element": sp, "occu": 1.0}
                if isinstance(sp, str)
                else {"occu": 1.0, **sp}
                for sp in site.get("species", [])
            ]
            coords = {
                key: [float(coord) for coord in site[key]]
                for key in ("abc", "xyz")
                if key in site
            }
            element = species[0].get("element", "X") if species else "X"
            defaults = {"label": f"{element}{site_idx + 1}", "properties": {}}
            sites.append({**defaults, **site, **coords, "species": species})
        return {**completed, "sites": sites}

    def _complete_trajectory_dict(
        self, trajectory_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Complete every frame of a ``{"frames": [{"structure", "step"?, ...}]}``
        dict; ``step`` (x axis of the property plots) defaults to the frame index.
        """
        frames = [
            self._attach_forces(
                {
                    **frame,
                    "step": frame.get("step", frame_idx),
                    "structure": self._complete_structure_fields(frame["structure"]),
                },
                frame_idx,
            )
            for frame_idx, frame in enumerate(trajectory_data["frames"])
        ]
        return {**trajectory_data, "frames": frames}

    def _normalize_frames(
        self, items: list[Any] | tuple[Any, ...], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert a sequence of structures or ``{"structure", **props}`` dicts to a
        matterviz trajectory dict with one frame per item.
        """
        if envelope := trajectory_to_envelope(items, metadata):
            return envelope
        frames: list[dict[str, Any]] = []
        for step_idx, item in enumerate(items):
            structure, properties = frame_input(item)
            frame = {"structure": self._to_structure_dict(structure), "step": step_idx}
            if properties:
                frame["metadata"] = properties
            frames.append(self._attach_forces(frame, step_idx))
        return {"frames": frames, "metadata": metadata}

    def _pymatgen_json(self, data: dict[str, Any]) -> dict[str, Any]:
        """Expand occupancy mappings into the renderer's per-site species lists.
        Needs the optional pymatgen dependency.
        """
        from matscikit.interop import decode_monty
        from pymatgen.core import DummySpecies, Molecule, Species, Structure
        from pymatgen.core.periodic_table import get_el_sp

        from pymatviz.utils.data import atomic_numbers

        # The renderer accepts symbols and species dicts; pymatgen also accepts
        # atomic numbers and decorated strings such as Fe2+.
        species = [
            get_el_sp(entry)
            if isinstance(entry, (str, int, float, np.number))
            and entry not in atomic_numbers
            else entry
            for entry in data.get("species", [])
        ]
        if not any(
            isinstance(entry, Mapping) and "element" not in entry for entry in species
        ):
            return data | {"species": species}
        # Rebuild only the known species fields when raw JSON mixes decorated
        # elements with occupancy dictionaries. Metadata stays plain data.
        for idx, entry in enumerate(species):
            if isinstance(entry, dict) and "element" in entry:
                atom = entry["element"]
                if entry.get("oxidation_state") is not None:
                    cls = (
                        DummySpecies
                        if entry.get("@class") == "DummySpecies"
                        else Species
                    )
                    atom = cls.from_dict(entry)
                species[idx] = {atom: entry.get("occu", 1)}
            elif isinstance(entry, dict):
                species[idx] = {
                    key: val for key, val in entry.items() if not key.startswith("@")
                }
        coords = np.asarray(decode_monty(data["coords"]), dtype=float)
        if data.get("coords_are_displacement"):
            coords = np.cumsum(coords, axis=0) + decode_monty(data["base_positions"])
        lattice = decode_monty(data.get("lattice"))
        cells = (
            None if lattice is None else np.broadcast_to(lattice, (len(coords), 3, 3))
        )
        site_props = decode_monty(data.get("site_properties")) or {}
        if isinstance(site_props, list) and len(site_props) == 1:
            (site_props,) = site_props  # The renderer broadcasts a single property map.
        frame_props = decode_monty(data.get("frame_properties")) or [{}] * len(coords)
        frames = []
        for idx, (positions, props) in enumerate(zip(coords, frame_props, strict=True)):
            per_site = site_props[idx] if isinstance(site_props, list) else site_props
            if cells is None:
                structure = Molecule(
                    species,
                    positions,
                    charge=data.get("charge") or 0,
                    spin_multiplicity=data.get("spin_multiplicity"),
                    site_properties=per_site,
                )
            else:
                structure = Structure(
                    cells[idx],
                    species,
                    positions,
                    charge=data.get("charge"),
                    site_properties=per_site,
                )
            frame = {"structure": structure.as_dict(), "step": idx, "metadata": props}
            frames.append(self._attach_forces(frame, idx))
        result: dict[str, Any] = {"frames": frames}
        if (step := data.get("time_step")) is not None:
            result["time_step"] = {"value": step, "unit": "fs"}
        return result

    def _normalize_trajectory(self, trajectory: Any) -> dict[str, Any] | None:
        """Convert trajectory to a ``matterviz-binary-v1`` envelope when the binary
        format can hold it (fixed species and order, numeric per-frame scalars,
        per-atom 3-vectors), else to matterviz's JSON trajectory format, which also
        carries disordered sites, varying composition, string or tensor metadata.
        """
        if trajectory is None or is_binary_envelope(trajectory, "trajectory"):
            return trajectory

        from matscikit.interop import has_class, is_structure_like

        is_pymatgen = has_class(trajectory, "pymatgen.core.trajectory.Trajectory")
        if is_pymatgen or (
            isinstance(trajectory, dict) and trajectory.get("@class") == "Trajectory"
        ):
            step = trajectory.time_step if is_pymatgen else trajectory.get("time_step")
            if step is not None and (
                not _is_number(step) or not 0 < float(step) < np.inf
            ):
                raise ValueError(
                    f"time_step value must be finite and > 0, got {step!r}"
                )
            if is_pymatgen:
                if envelope := trajectory_to_envelope(trajectory):
                    return envelope
                trajectory = trajectory.as_dict()
            return self._pymatgen_json(trajectory)

        if isinstance(trajectory, dict):
            frames_data = trajectory.get("frames")
            if (
                isinstance(frames_data, list)
                and frames_data
                and all(is_structure_like(frame) for frame in frames_data)
            ):
                input_metadata = trajectory.get("metadata")
                metadata = input_metadata if isinstance(input_metadata, dict) else {}
                return self._normalize_frames(frames_data, metadata)
            return self._complete_trajectory_dict(trajectory)

        if is_structure_like(trajectory):
            trajectory = [trajectory]
        if isinstance(trajectory, (list, tuple)):
            return self._normalize_frames(trajectory, {})

        raise TypeError(
            f"Unsupported trajectory type: {type(trajectory)}. "
            "Expected list of structures, single structure, or trajectory dict."
        )
