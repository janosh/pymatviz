"""Trajectory visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import traitlets as tl

from pymatviz.structure.helpers import add_vacuum_if_needed
from pymatviz.widgets._traits import StructureVizTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class TrajectoryWidget(StructureVizTraits, MatterVizWidget):
    """MatterViz widget for visualizing molecular dynamics and geometry optimization
    trajectories in Python notebooks.

    Accepts trajectory data directly (list of structures, dict with frames) or a
    ``data_url`` the frontend fetches and parses itself (XYZ/extXYZ, ASE .traj,
    XDATCAR, LAMMPS dump, pymatgen ``Trajectory`` JSON; ``.gz``/``.zip`` compressed).

    Examples:
        Basic usage with list of structures:
        >>> from pymatviz import TrajectoryWidget
        >>> trajectory_data = [...]  # List of structures
        >>> widget = TrajectoryWidget(trajectory=trajectory_data)
        >>> widget

        With per-frame properties (scalars are plotted; per-atom ``forces`` of shape
        (n_sites, 3) are drawn as vectors with their max/RMS norms plotted):
        >>> trajectory_with_props = [
        ...     {"structure": struct1, "energy": -1.23, "forces": forces1},
        ...     {"structure": struct2, "energy": -1.25, "forces": forces2},
        ... ]
        >>> widget = TrajectoryWidget(trajectory=trajectory_with_props)

        A pymatgen ``Trajectory`` (or its ``as_dict()``) is parsed by the frontend:
        >>> widget = TrajectoryWidget(trajectory=Trajectory.from_file("XDATCAR"))

        With custom visualization options:
        >>> widget = TrajectoryWidget(
        ...     trajectory=trajectory_data,
        ...     display_mode="structure+plot",
        ...     plot_type="distribution",
        ...     layout="horizontal",
        ...     show_controls=True,
        ...     auto_play=True,
        ...     style="height: 600px; border: 2px solid blue;",
        ... )

        With local file path (automatically detected and loaded):
        >>> widget = TrajectoryWidget(data_url="path/to/trajectory.xyz")
        >>> widget = TrajectoryWidget(data_url="path/to/trajectory.h5")

        LAMMPS dumps only carry integer atom types; name them with atom_type_mapping:
        >>> widget = TrajectoryWidget(
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
    # Property shown in the distribution plot (None = highest-priority property)
    distribution_property = optional_trait(tl.Unicode)
    auto_play = optional_trait(tl.Bool)
    # {atom type: element} for LAMMPS dumps loaded from data_url, e.g. {1: "Si", 2: "O"}
    atom_type_mapping = optional_trait(tl.Dict)

    show_image_atoms = optional_trait(tl.Bool)

    # Plot
    step_labels = tl.Union(
        [tl.Int(), tl.List()], allow_none=True, default_value=None
    ).tag(sync=True)
    property_labels = tl.Dict(allow_none=True).tag(sync=True)

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
            **kwargs: Additional widget properties
        """
        super().__init__(widget_type="trajectory", trajectory=trajectory, **kwargs)

    @tl.validate("trajectory")
    def _validate_trajectory(self, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize constructor data and later assignments before syncing to JS."""
        from pymatviz.widgets._normalize import normalize_plot_json

        return normalize_plot_json(
            self._normalize_trajectory(proposal["value"]), "trajectory"
        )

    def _to_structure_dict(self, structure_input: Any) -> tuple[dict[str, Any], Any]:
        """Convert structure-like input to dict and metadata source object."""
        from pymatviz.process_data import normalize_structures

        if hasattr(structure_input, "as_dict"):
            return structure_input.as_dict(), structure_input

        # Handle ASE Atoms-like objects that may not define a full cell.
        structure_obj = add_vacuum_if_needed(structure_input)
        normalized_structures = normalize_structures(structure_obj)
        if len(normalized_structures) != 1:
            raise ValueError(
                f"Expected exactly one structure per frame, got "
                f"{len(normalized_structures)}"
            )
        structure_dict = next(iter(normalized_structures.values())).as_dict()
        # metadata from the original: Atoms.copy() in add_vacuum_if_needed drops calc
        return structure_dict, structure_input

    def _extract_object_metadata(self, structure_input: Any) -> dict[str, Any]:
        """Extract metadata from structure-like object properties/info, plus ASE
        calculator results (energy, forces, stress, ...) which pymatgen's
        ``AseAtomsAdaptor`` drops.
        """
        metadata = dict(
            getattr(structure_input, "properties", None)
            or getattr(structure_input, "info", None)
            or {}
        )
        calc = getattr(structure_input, "calc", None)
        return metadata | dict(getattr(calc, "results", None) or {})

    @staticmethod
    def _attach_forces(frame: dict[str, Any], frame_idx: int) -> dict[str, Any]:
        """Move per-atom forces from frame metadata (``force`` or ``forces``) to each
        site's ``properties.force`` (matterviz 0.8 ignores vectors in frame metadata)
        and record ``force_max``/``force_norm`` (max and RMS of the per-atom force
        norms, like matterviz's parsers) from the sites' ``force``/``forces`` vectors
        unless the metadata already has them. Returns a new frame; inputs are not
        mutated.

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
            norms = np.linalg.norm(vectors, axis=1)
            metadata.setdefault("force_max", float(norms.max()))
            metadata.setdefault("force_norm", float(np.sqrt(np.mean(norms**2))))
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
            from pymatgen.core.lattice import Lattice

            matrix = Lattice.from_parameters(
                *(float(lattice[key]) for key in cell_params)
            )
            completed["lattice"] = {**lattice, "matrix": matrix.matrix.tolist()}
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

    def _normalize_frames(self, items: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        """Convert a sequence of structures or ``{"structure", **props}`` dicts to a
        matterviz trajectory dict with one frame per item.
        """
        frames: list[dict[str, Any]] = []
        for step_idx, item in enumerate(items):
            if isinstance(item, dict):
                structure = item.get("structure", item)
                properties = {
                    key: val for key, val in item.items() if key != "structure"
                }
            else:
                structure, properties = item, {}

            structure_dict, metadata_source = self._to_structure_dict(structure)
            frame: dict[str, Any] = {"structure": structure_dict, "step": step_idx}
            if metadata := properties or self._extract_object_metadata(metadata_source):
                frame["metadata"] = metadata
            frames.append(self._attach_forces(frame, step_idx))
        return {"frames": frames, "metadata": {}}

    def _normalize_trajectory(self, trajectory: Any) -> dict[str, Any] | None:
        """Convert trajectory to matterviz format."""
        if trajectory is None:
            return None

        from pymatgen.core.trajectory import Trajectory

        from pymatviz.process_data import is_structure_like

        if isinstance(trajectory, Trajectory):
            trajectory = trajectory.as_dict()
        if isinstance(trajectory, dict) and trajectory.get("@class") == "Trajectory":
            # matterviz parses pymatgen Trajectory dicts itself, moving
            # frame_properties forces to sites and their stats into frame metadata
            return trajectory

        if isinstance(trajectory, dict):
            frames_data = trajectory.get("frames")
            if (
                isinstance(frames_data, list)
                and frames_data
                and all(is_structure_like(frame) for frame in frames_data)
            ):
                normalized_trajectory = self._normalize_frames(frames_data)
                if isinstance(input_metadata := trajectory.get("metadata"), dict):
                    normalized_trajectory["metadata"] = input_metadata
                return normalized_trajectory
            return self._complete_trajectory_dict(trajectory)

        if isinstance(trajectory, (list, tuple)):
            return self._normalize_frames(trajectory)

        if hasattr(trajectory, "as_dict") or hasattr(
            trajectory, "get_chemical_symbols"
        ):
            return self._normalize_frames([trajectory])

        raise TypeError(
            f"Unsupported trajectory type: {type(trajectory)}. "
            "Expected list of structures, single structure, or trajectory dict."
        )
