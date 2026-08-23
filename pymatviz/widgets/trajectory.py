"""Trajectory visualization widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import traitlets as tl

from pymatviz.structure.helpers import add_vacuum_if_needed
from pymatviz.widgets._traits import StructureVizTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class TrajectoryWidget(StructureVizTraits, MatterVizWidget):
    """MatterViz widget for visualizing molecular dynamics and geometry optimization
    trajectories in Python notebooks.

    The widget supports multiple input formats:
    - Direct trajectory data (list of structures, dict with frames), sent as JSON
    - ``data_url``: a URL the frontend fetches, decompresses and parses itself

    File formats the frontend parses from ``data_url`` (optionally ``.gz`` or
    ``.zip`` compressed; ``.bz2``/``.xz`` cannot be inflated in the browser):
    - XYZ / extended XYZ (.xyz, .extxyz) with one or more frames
    - ASE ULM binary trajectory files (.traj)
    - VASP XDATCAR and LAMMPS dump files
    - pymatgen ``Trajectory`` JSON or ``{"frames": [...]}`` JSON
    HDF5 files are not parsed in the widget bundle: load them in Python and pass
    the frames instead.

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

        LAMMPS dumps without an element column only carry integer atom types; name
        them with ``atom_type_mapping`` (unmapped types fall back to atomic number
        = type, with a warning in the browser console):
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
    # {atom type: element symbol} for LAMMPS dumps parsed from data_url, e.g.
    # {"1": "Si", "2": "O"} (int keys are fine: JSON stringifies them). Forwarded into
    # TrajectoryFileViewer.loading_options by the anywidget bridge.
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
        """Fill the fields matterviz's JSON reader does not default itself.

        The renderer rebuilds every lattice from its ``matrix`` (scalar parameters,
        volume, a fully periodic ``pbc`` when missing) and derives the missing one of
        ``abc``/``xyz`` per site, so only a ``matrix`` built from cell parameters,
        species objects/occupancies, site labels and site properties are filled here.
        """
        completed_structure = dict(structure_data)
        # copy so assignments below don't mutate the caller's dict
        lattice_data = dict(completed_structure["lattice"])

        if "matrix" not in lattice_data:
            if not all(
                key in lattice_data for key in ("a", "b", "c", "alpha", "beta", "gamma")
            ):
                raise ValueError(
                    "Trajectory frame structure lattice must provide either 'matrix' "
                    "or all of ['a', 'b', 'c', 'alpha', 'beta', 'gamma']."
                )
            from pymatgen.core.lattice import Lattice

            lattice_data["matrix"] = Lattice.from_parameters(
                *(
                    float(lattice_data[key])
                    for key in ("a", "b", "c", "alpha", "beta", "gamma")
                )
            ).matrix.tolist()

        completed_sites: list[dict[str, Any]] = []
        for site_idx, site_data in enumerate(completed_structure["sites"]):
            site_dict = dict(site_data)
            species_data = site_dict["species"]
            self._validate_species_list(species_data, f"Site index {site_idx}, value")
            # the renderer reads species as {element, occu} objects: promote bare
            # symbols and default the occupancy
            site_dict["species"] = [
                {"element": species, "occu": 1.0}
                if isinstance(species, str)
                else {"occu": 1.0, **species}
                for species in species_data
            ]

            if "abc" not in site_dict and "xyz" not in site_dict:
                site_keys = sorted(str(key) for key in site_dict)
                raise ValueError(
                    "Trajectory frame site needs coordinate key 'abc' or 'xyz'. "
                    f"Site index: {site_idx}, keys: {site_keys}."
                )
            for coord_key in ("abc", "xyz"):
                if coord_key in site_dict:
                    site_dict[coord_key] = [
                        float(coord) for coord in site_dict[coord_key]
                    ]

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
        """Fill all frames with the fields the widget renderer requires.

        ``step`` defaults to the frame index: the renderer rejects frames without a
        finite step (it is the x-axis of the property plots).
        """
        completed_trajectory = dict(trajectory_data)
        completed_frames: list[dict[str, Any]] = []
        for frame_idx, frame_data in enumerate(trajectory_data["frames"]):
            frame_dict = dict(frame_data)
            frame_dict.setdefault("step", frame_idx)
            frame_dict["structure"] = self._complete_structure_fields(
                frame_dict["structure"]
            )
            completed_frames.append(frame_dict)
        completed_trajectory["frames"] = completed_frames
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
        self._validate_species_list(species_data, "First site (index 0), value")
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

        from pymatviz.widgets._normalize import normalize_plot_json

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
