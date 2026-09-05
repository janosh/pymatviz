"""Trajectory visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.structure.helpers import add_vacuum_if_needed
from pymatviz.widgets._traits import StructureVizTraits
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

        With properties in dict format:
        >>> trajectory_with_props = [
        ...     {"structure": struct1, "energy": -1.23, "force": [0.1, 0.2, 0.3]},
        ...     {"structure": struct2, "energy": -1.25, "force": [0.05, 0.15, 0.25]},
        ... ]
        >>> widget = TrajectoryWidget(trajectory=trajectory_with_props)

        With custom visualization options:
        >>> widget = TrajectoryWidget(
        ...     trajectory=trajectory_data,
        ...     display_mode="structure+scatter",
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

    # display options shared with StructureWidget live in StructureVizTraits

    trajectory = tl.Dict(allow_none=True).tag(sync=True)
    current_step_idx = tl.Int(0).tag(sync=True)

    # Layout
    layout = tl.CaselessStrEnum(
        ["auto", "horizontal", "vertical"], default_value="auto"
    ).tag(sync=True)
    display_mode = tl.CaselessStrEnum(
        [
            "structure+scatter",
            "structure",
            "scatter",
            "histogram",
            "structure+histogram",
        ],
        default_value="structure+scatter",
    ).tag(sync=True)
    auto_play = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    # {atom type: element} for LAMMPS dumps loaded from data_url, e.g. {1: "Si", 2: "O"}
    atom_type_mapping = tl.Dict(allow_none=True, default_value=None).tag(sync=True)

    show_image_atoms = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Plot
    step_labels = tl.Union(
        [tl.Int(), tl.List()], allow_none=True, default_value=None
    ).tag(sync=True)
    property_labels = tl.Dict(allow_none=True).tag(sync=True)

    def __init__(
        self, trajectory: dict[str, Any] | list[Any] | Any | None = None, **kwargs: Any
    ) -> None:
        """Initialize the TrajectoryWidget.

        Args:
            trajectory: Trajectory data in one of these formats:
                - dict with 'frames' key (matterviz format)
                - list of structures (pymatgen Structure, ASE Atoms, PhonopyAtoms, etc.)
                - list of dicts with properties: [
                    {"structure": struct, "energy": 1.23, ...}, ...
                ]
            **kwargs: Additional widget properties
        """
        if trajectory is not None:  # Convert trajectory objects if needed
            trajectory = self._normalize_trajectory(trajectory)

        super().__init__(widget_type="trajectory", trajectory=trajectory, **kwargs)

    def _to_structure_dict(self, structure_input: Any) -> tuple[dict[str, Any], Any]:
        """Convert structure-like input to dict and metadata source object."""
        from pymatviz.process_data import normalize_structures

        structure_obj = structure_input
        if hasattr(structure_obj, "as_dict"):
            return structure_obj.as_dict(), structure_obj

        # Handle ASE Atoms-like objects that may not define a full cell.
        structure_obj = add_vacuum_if_needed(structure_obj)
        normalized_structures = normalize_structures(structure_obj)
        if len(normalized_structures) != 1:
            raise ValueError(
                f"Expected exactly one structure per frame, got "
                f"{len(normalized_structures)}"
            )
        structure_dict = next(iter(normalized_structures.values())).as_dict()
        return structure_dict, structure_obj

    def _extract_object_metadata(self, structure_input: Any) -> dict[str, Any]:
        """Extract metadata from structure-like object properties/info."""
        if hasattr(structure_input, "properties") and structure_input.properties:
            return dict(structure_input.properties)
        if hasattr(structure_input, "info") and structure_input.info:
            return dict(structure_input.info)
        return {}

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

            matrix = Lattice.from_parameters(*(float(lattice[k]) for k in cell_params))
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
                k: [float(c) for c in site[k]] for k in ("abc", "xyz") if k in site
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
            {
                **frame,
                "step": frame.get("step", frame_idx),
                "structure": self._complete_structure_fields(frame["structure"]),
            }
            for frame_idx, frame in enumerate(trajectory_data["frames"])
        ]
        return {**trajectory_data, "frames": frames}

    def _normalize_trajectory(self, trajectory: Any) -> dict[str, Any] | None:
        """Convert trajectory to matterviz format."""
        if trajectory is None:
            return None

        from pymatviz.process_data import is_structure_like
        from pymatviz.widgets._normalize import normalize_plot_json

        if isinstance(trajectory, dict):
            frames_data = trajectory.get("frames")
            if (
                isinstance(frames_data, list)
                and frames_data
                and all(is_structure_like(frame) for frame in frames_data)
            ):
                normalized_trajectory = self._normalize_trajectory(frames_data)
                if isinstance(normalized_trajectory, dict):
                    input_metadata = trajectory.get("metadata")
                    if isinstance(input_metadata, dict):
                        normalized_trajectory["metadata"] = input_metadata
                return normalize_plot_json(normalized_trajectory, "trajectory")
            return normalize_plot_json(
                self._complete_trajectory_dict(trajectory), "trajectory"
            )

        if isinstance(trajectory, (list, tuple)):
            frames: list[dict[str, Any]] = []
            for step_idx, item in enumerate(trajectory):
                if isinstance(item, dict):
                    structure = item.get("structure", item)
                    properties = {k: v for k, v in item.items() if k != "structure"}
                else:
                    structure = item
                    properties = {}

                structure_dict, metadata_source = self._to_structure_dict(structure)
                frame: dict[str, Any] = {"structure": structure_dict, "step": step_idx}

                metadata = properties or self._extract_object_metadata(metadata_source)
                if metadata:
                    # convert numpy arrays/scalars to JSON-safe primitives so the
                    # frontend receives numeric arrays, not stringified reprs
                    frame["metadata"] = normalize_plot_json(
                        metadata, "trajectory.frame.metadata"
                    )

                frames.append(frame)

            return {"frames": frames, "metadata": {}}

        if hasattr(trajectory, "as_dict") or hasattr(
            trajectory, "get_chemical_symbols"
        ):
            structure_dict, metadata_source = self._to_structure_dict(trajectory)
            frame: dict[str, Any] = {"structure": structure_dict, "step": 0}
            metadata = self._extract_object_metadata(metadata_source)
            if metadata:
                frame["metadata"] = normalize_plot_json(
                    metadata, "trajectory.frame.metadata"
                )
            return {"frames": [frame], "metadata": {}}

        raise TypeError(
            f"Unsupported trajectory type: {type(trajectory)}. "
            "Expected list of structures, single structure, or trajectory dict."
        )
