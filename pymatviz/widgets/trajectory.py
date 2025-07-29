"""Trajectory visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any, Literal

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
    - torch-sim HDF5 files (.h5, .hdf5)
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
    layout: Literal["auto", "horizontal", "vertical"] = tl.Unicode("auto").tag(
        sync=True
    )
    display_mode: Literal[
        "structure+scatter", "structure", "scatter", "histogram", "structure+histogram"
    ] = tl.Unicode("structure+scatter").tag(sync=True)
    show_controls = tl.Bool(default_value=True).tag(sync=True)
    show_fullscreen_button = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    auto_play = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Styling
    style = tl.Unicode(allow_none=True).tag(sync=True)  # Custom CSS styles

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
    cell_line_width = tl.Float(1.5).tag(sync=True)
    show_vectors = tl.Bool(allow_none=True, default_value=None).tag(sync=True)

    # Appearance
    background_color = tl.Unicode(allow_none=True).tag(sync=True)
    background_opacity = tl.Float(allow_none=True, default_value=None).tag(sync=True)

    # UI controls
    show_info = tl.Bool(default_value=True).tag(sync=True)
    png_dpi = tl.Int(allow_none=True, default_value=None).tag(sync=True)

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

        super().__init__(trajectory=trajectory, **kwargs)

    def _normalize_trajectory(self, trajectory: Any) -> dict[str, Any] | None:
        """Convert trajectory to matterviz format."""
        if trajectory is None:
            return None

        # Check if already in correct format (dict with 'frames' key)
        if isinstance(trajectory, dict) and "frames" in trajectory:
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
