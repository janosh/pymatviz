"""Trajectory visualization widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import traitlets as tl

from pymatviz.structure.helpers import add_vacuum_if_needed
from pymatviz.widgets.matterviz import MatterVizWidget


class TrajectoryWidget(MatterVizWidget):
    """MatterViz widget for visualizing molecular dynamics and geometry optimization
    trajectories in Python notebooks.

    The widget supports multiple input formats:
    - Direct trajectory data (list of structures, dict with frames)
    - Local file paths to trajectory files (automatically detected and loaded)
    - Remote file URLs to trajectory files (automatically detected and loaded)

    Supported file formats:
    - XYZ files (.xyz, .xyz.gz, .xyz.bz2, .xyz.xz, .extxyz, .extxyz.gz, ...)
    - ASE ULM binary trajectory files (.traj)
    - flame HDF5 files (.h5, .hdf5)
    - NumPy compressed arrays (.npz)
    - Pickle files (.pkl)
    - Generic data files (.dat)
    - ZIP archives (.zip) containing trajectory files

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
    """

    trajectory = tl.Dict(allow_none=True).tag(sync=True)
    data_url = tl.Unicode(allow_none=True).tag(sync=True)
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
    fullscreen_toggle = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    auto_play = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Structure visualization
    atom_radius = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_atoms = tl.Bool(default_value=True).tag(sync=True)
    show_bonds = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_site_labels = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_image_atoms = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    show_force_vectors = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    same_size_atoms = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    color_scheme = tl.Unicode("Vesta").tag(sync=True)

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
    cell_edge_width = tl.Float(1.5).tag(sync=True)
    show_cell_vectors = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Appearance
    background_color = tl.Unicode(allow_none=True).tag(sync=True)
    background_opacity = tl.Float(allow_none=True, default_value=None).tag(sync=True)

    # Plot
    step_labels = tl.Union(
        [tl.Int(), tl.List()], allow_none=True, default_value=None
    ).tag(sync=True)
    property_labels = tl.Dict(allow_none=True).tag(sync=True)
    units = tl.Dict(allow_none=True).tag(sync=True)

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

    @staticmethod
    def _validate_species_list(species_data: Any, location: str) -> None:
        """Validate that species data is a non-empty list."""
        if not isinstance(species_data, list) or not species_data:
            raise ValueError(
                "Trajectory frame site key 'species' must be a non-empty list. "
                f"{location}: {species_data}."
            )

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

    def _complete_structure_fields(
        self, structure_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Fill derived fields from minimal valid trajectory structure data."""
        from pymatgen.core.lattice import Lattice

        completed_structure = dict(structure_data)
        lattice_data = completed_structure["lattice"]

        has_lattice_matrix = "matrix" in lattice_data
        has_cell_params = all(
            key in lattice_data for key in ("a", "b", "c", "alpha", "beta", "gamma")
        )

        if has_lattice_matrix:
            lattice_obj = Lattice(lattice_data["matrix"])
        elif has_cell_params:
            lattice_obj = Lattice.from_parameters(
                a=float(lattice_data["a"]),
                b=float(lattice_data["b"]),
                c=float(lattice_data["c"]),
                alpha=float(lattice_data["alpha"]),
                beta=float(lattice_data["beta"]),
                gamma=float(lattice_data["gamma"]),
            )
            lattice_data["matrix"] = lattice_obj.matrix.tolist()
        else:
            raise ValueError(
                "Trajectory frame structure lattice must provide either 'matrix' or "
                "all of ['a', 'b', 'c', 'alpha', 'beta', 'gamma']."
            )

        lattice_data.setdefault("pbc", [True, True, True])
        lattice_data["a"] = float(lattice_obj.a)
        lattice_data["b"] = float(lattice_obj.b)
        lattice_data["c"] = float(lattice_obj.c)
        lattice_data["alpha"] = float(lattice_obj.alpha)
        lattice_data["beta"] = float(lattice_obj.beta)
        lattice_data["gamma"] = float(lattice_obj.gamma)
        lattice_data["volume"] = float(lattice_obj.volume)

        completed_sites: list[dict[str, Any]] = []
        for site_idx, site_data in enumerate(completed_structure["sites"]):
            site_dict = dict(site_data)
            species_data = site_dict["species"]
            self._validate_species_list(species_data, f"Site index {site_idx}, value")
            site_dict["species"] = [
                (
                    {**species, "occu": 1.0}
                    if isinstance(species, Mapping) and "occu" not in species
                    else species
                )
                for species in species_data
            ]

            has_abc = "abc" in site_dict
            has_xyz = "xyz" in site_dict
            if not has_abc and not has_xyz:
                site_keys = sorted(str(key) for key in site_dict)
                raise ValueError(
                    "Trajectory frame site needs coordinate key 'abc' or 'xyz'. "
                    f"Site index: {site_idx}, keys: {site_keys}."
                )

            if has_abc:
                abc_coords = [float(coord) for coord in site_dict["abc"]]
                site_dict["abc"] = abc_coords
            if has_xyz:
                xyz_coords = [float(coord) for coord in site_dict["xyz"]]
                site_dict["xyz"] = xyz_coords

            if has_abc and not has_xyz:
                site_dict["xyz"] = lattice_obj.get_cartesian_coords(
                    site_dict["abc"]
                ).tolist()
            elif has_xyz and not has_abc:
                site_dict["abc"] = lattice_obj.get_fractional_coords(
                    site_dict["xyz"]
                ).tolist()

            default_species = site_dict["species"][0]
            default_element = (
                str(default_species.get("element", "X"))
                if isinstance(default_species, Mapping)
                else str(default_species)
            )
            site_dict.setdefault("label", f"{default_element}{site_idx + 1}")
            site_dict.setdefault("properties", {})
            completed_sites.append(site_dict)

        completed_structure["lattice"] = lattice_data
        completed_structure["sites"] = completed_sites
        return completed_structure

    def _complete_trajectory_dict(
        self, trajectory_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Fill all frames with derived fields required by the widget renderer."""
        completed_trajectory = dict(trajectory_data)
        completed_frames: list[dict[str, Any]] = []
        for frame_data in trajectory_data["frames"]:
            frame_dict = dict(frame_data)
            frame_dict["structure"] = self._complete_structure_fields(
                frame_dict["structure"]
            )
            completed_frames.append(frame_dict)
        completed_trajectory["frames"] = completed_frames
        if "metadata" in trajectory_data:
            completed_trajectory["metadata"] = trajectory_data["metadata"]
        return completed_trajectory

    def _validate_trajectory_dict(self, trajectory_data: dict[str, Any]) -> None:
        """Validate trajectory-dict schema and raise helpful errors.

        Expected top-level schema:
            {"frames": [frame0, frame1, ...], "metadata": {...}}
        Expected frame schema:
            {"structure": <structure-dict>, ...}
        Expected structure schema:
            {"lattice": {"matrix": ...}, "sites": [...]}
        """
        if "frames" not in trajectory_data:
            available_keys = sorted(str(key) for key in trajectory_data)
            raise ValueError(
                "Trajectory dict is missing required key 'frames'. "
                f"Expected keys include ['frames', 'metadata']; got {available_keys}."
            )

        frames_data = trajectory_data["frames"]
        if not isinstance(frames_data, list):
            raise TypeError(
                "Trajectory dict key 'frames' must be a list. "
                f"Got type: {type(frames_data)}."
            )
        if not frames_data:
            raise ValueError(
                "Trajectory dict 'frames' is empty. Provide at least one frame."
            )

        first_frame = frames_data[0]
        if not isinstance(first_frame, dict):
            raise TypeError(
                "Trajectory frame must be a dict with at least a 'structure' key. "
                f"Got first frame type: {type(first_frame)}."
            )
        if "structure" not in first_frame:
            frame_keys = sorted(str(key) for key in first_frame)
            raise ValueError(
                "Trajectory frame is missing required key 'structure'. "
                f"Frame keys: {frame_keys}."
            )

        structure_data = first_frame["structure"]
        if not isinstance(structure_data, dict):
            raise TypeError(
                "Trajectory frame 'structure' must be a dict. "
                f"Got type: {type(structure_data)}."
            )

        if "sites" not in structure_data or not isinstance(
            structure_data["sites"], list
        ):
            raise ValueError(
                "Trajectory frame structure must include list-valued key 'sites'."
            )
        if not structure_data["sites"]:
            raise ValueError(
                "Trajectory frame structure has empty 'sites'. "
                "At least one site is required."
            )

        if "lattice" not in structure_data or not isinstance(
            structure_data["lattice"], dict
        ):
            raise ValueError("Trajectory frame structure must include 'lattice'.")
        lattice_data = structure_data["lattice"]
        has_lattice_matrix = "matrix" in lattice_data
        has_cell_params = all(
            key in lattice_data for key in ("a", "b", "c", "alpha", "beta", "gamma")
        )
        if not has_lattice_matrix and not has_cell_params:
            raise ValueError(
                "Trajectory frame structure lattice must provide either 'matrix' or "
                "all of ['a', 'b', 'c', 'alpha', 'beta', 'gamma']."
            )
        first_site = structure_data["sites"][0]
        if not isinstance(first_site, dict):
            raise TypeError(
                "Trajectory frame site entries must be dicts. "
                f"Got type: {type(first_site)}."
            )
        if "species" not in first_site:
            site_keys = sorted(str(key) for key in first_site)
            raise ValueError(
                "Trajectory frame site is missing required key 'species'. "
                f"Site keys: {site_keys}."
            )
        species_data = first_site["species"]
        self._validate_species_list(species_data, "Got value")
        if "abc" not in first_site and "xyz" not in first_site:
            site_keys = sorted(str(key) for key in first_site)
            raise ValueError(
                "Trajectory frame site needs coordinate key 'abc' or 'xyz'. "
                f"Site keys: {site_keys}."
            )

    def _normalize_trajectory(self, trajectory: Any) -> dict[str, Any] | None:
        """Convert trajectory to matterviz format."""
        if trajectory is None:
            return None

        from pymatviz.process_data import is_structure_like

        # Check if already in trajectory-dict format and validate schema.
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
                return normalized_trajectory
            self._validate_trajectory_dict(trajectory)
            return self._complete_trajectory_dict(trajectory)

        # Handle list/sequence of structures or dicts with properties
        if isinstance(trajectory, (list, tuple)):
            frames: list[dict[str, Any]] = []
            for step_idx, item in enumerate(trajectory):
                # Handle dict format with properties like {"structure": struct,
                # "energy": 1.23, ...}
                if isinstance(item, dict):
                    # Extract structure from dict
                    structure = item.get("structure", item)

                    # Extract properties (everything except 'structure')
                    properties = {k: v for k, v in item.items() if k != "structure"}
                else:
                    # Handle direct structure objects (backward compatibility)
                    structure = item
                    properties = {}

                structure_dict, metadata_source = self._to_structure_dict(structure)

                # Create trajectory frame
                frame = {"structure": structure_dict, "step": step_idx}

                # Add properties from dict format
                if properties:
                    frame["metadata"] = properties
                else:
                    metadata = self._extract_object_metadata(metadata_source)
                    if metadata:
                        frame["metadata"] = metadata

                frames.append(frame)

            return {"frames": frames, "metadata": {}}

        # Handle single structure (convert to single-frame trajectory)
        if hasattr(trajectory, "as_dict") or hasattr(
            trajectory, "get_chemical_symbols"
        ):
            structure_dict, metadata_source = self._to_structure_dict(trajectory)

            frame = {"structure": structure_dict, "step": 0}

            # Add metadata if available
            metadata = self._extract_object_metadata(metadata_source)

            if metadata:
                frame["metadata"] = metadata

            return {"frames": [frame], "metadata": {}}

        raise TypeError(
            f"Unsupported trajectory type: {type(trajectory)}. "
            "Expected list of structures, single structure, or trajectory dict."
        )
