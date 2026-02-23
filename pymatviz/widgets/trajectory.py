"""Trajectory visualization widget for Jupyter notebooks."""

from __future__ import annotations

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

        if (
            "lattice" not in structure_data
            or not isinstance(structure_data["lattice"], dict)
            or "matrix" not in structure_data["lattice"]
        ):
            raise ValueError(
                "Trajectory frame structure must include 'lattice.matrix' for "
                "periodic structure rendering."
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
        if "abc" not in first_site and "xyz" not in first_site:
            site_keys = sorted(str(key) for key in first_site)
            raise ValueError(
                "Trajectory frame site needs coordinate key 'abc' or 'xyz'. "
                f"Site keys: {site_keys}."
            )
        if "label" not in first_site:
            raise ValueError(
                "Trajectory frame site is missing key 'label'. Provide a per-site "
                "label like 'Si1' for stable widget rendering."
            )
        if "properties" not in first_site or not isinstance(
            first_site["properties"], dict
        ):
            raise ValueError(
                "Trajectory frame site needs dict key 'properties' "
                "(use {} if no properties are present)."
            )

        missing_lattice_keys = [
            key
            for key in ("a", "b", "c", "alpha", "beta", "gamma")
            if key not in structure_data["lattice"]
        ]
        if missing_lattice_keys:
            raise ValueError(
                "Trajectory frame structure lattice is missing derived cell keys "
                f"{missing_lattice_keys}. Include keys "
                "['a', 'b', 'c', 'alpha', 'beta', 'gamma'] for robust rendering."
            )

    def _normalize_trajectory(self, trajectory: Any) -> dict[str, Any] | None:
        """Convert trajectory to matterviz format."""
        if trajectory is None:
            return None

        def is_structure_like(struct_or_atoms: Any) -> bool:
            """Check whether object looks like a structure/atoms instance."""
            return hasattr(struct_or_atoms, "as_dict") or hasattr(
                struct_or_atoms, "get_chemical_symbols"
            )

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
            return trajectory

        # Handle list/sequence of structures or dicts with properties
        if isinstance(trajectory, (list, tuple)):
            from pymatviz.process_data import normalize_structures

            frames = []
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

                # Convert structure to pymatgen Structure then to dict
                struct_i = structure
                if hasattr(struct_i, "as_dict"):
                    # Already a pymatgen object
                    struct_dict = struct_i.as_dict()
                else:  # Handle ASE Atoms (that might not have a cell)
                    struct_i = add_vacuum_if_needed(struct_i)

                    # Use normalize_structures to handle conversion
                    normalized = normalize_structures(struct_i)
                    if len(normalized) != 1:
                        raise ValueError(
                            f"Expected exactly one structure per frame, got "
                            f"{len(normalized)}"
                        )
                    struct_dict = next(iter(normalized.values())).as_dict()

                # Create trajectory frame
                frame = {"structure": struct_dict, "step": step_idx}

                # Add properties from dict format
                if properties:
                    frame["metadata"] = properties
                else:
                    metadata = {}  # Add metadata if structure has properties/info
                    if hasattr(struct_i, "properties") and struct_i.properties:
                        metadata.update(struct_i.properties)
                    elif hasattr(struct_i, "info") and struct_i.info:
                        metadata.update(struct_i.info)

                    if metadata:
                        frame["metadata"] = metadata

                frames.append(frame)

            return {"frames": frames, "metadata": {}}

        # Handle single structure (convert to single-frame trajectory)
        if hasattr(trajectory, "as_dict") or hasattr(
            trajectory, "get_chemical_symbols"
        ):
            from pymatviz.process_data import normalize_structures

            if hasattr(trajectory, "as_dict"):
                struct_dict = trajectory.as_dict()
            else:  # Handle ASE Atoms (that might not have a cell)
                trajectory = add_vacuum_if_needed(trajectory)

                normalized = normalize_structures(trajectory)
                if len(normalized) != 1:
                    raise ValueError(
                        f"Expected exactly one structure per frame, got "
                        f"{len(normalized)}"
                    )
                struct_dict = next(iter(normalized.values())).as_dict()

            frame = {"structure": struct_dict, "step": 0}

            # Add metadata if available
            metadata = {}
            if hasattr(trajectory, "properties") and trajectory.properties:
                metadata.update(trajectory.properties)
            elif hasattr(trajectory, "info") and trajectory.info:
                metadata.update(trajectory.info)

            if metadata:
                frame["metadata"] = metadata

            return {"frames": [frame], "metadata": {}}

        raise TypeError(
            f"Unsupported trajectory type: {type(trajectory)}. "
            "Expected list of structures, single structure, or trajectory dict."
        )
