"""Trajectory visualization widget for Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import traitlets as tl

from pymatviz.structures import structure_dict
from pymatviz.widgets._binary import (
    BINARY_FORMAT,
    force_stats,
    frame_input,
    frame_item,
    is_binary_envelope,
    trajectory_to_envelope,
)
from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets._schema_traits import TrajectoryTraits
from pymatviz.widgets._traits import (
    JsonBool,
    JsonInt,
    StructureVizTraits,
    checked_position,
    clamped_position,
    normalized_trait,
    optional_trait,
    validator,
)
from pymatviz.widgets.matterviz import MatterVizWidget


def _is_number(value: Any) -> bool:
    """Recognize real numeric scalars, including 0-d arrays and excluding booleans."""
    if isinstance(value, int):
        return not isinstance(value, bool)
    return (
        isinstance(value, (float, np.number, np.ndarray))
        and np.ndim(value) == 0
        and np.asarray(value).dtype.kind in "iuf"
    )


def n_frames(trajectory: Any) -> int | None:
    """Frame count of a ``trajectory`` trait value: a binary envelope, a ``{"frames":
    [...]}`` dict or (mid-batch, before its validator ran) a sequence of frames. None
    when Python can't tell without converting it (None for a ``data_url`` trajectory,
    raw pymatgen Trajectories, iterators, which counting would consume).
    """
    if isinstance(trajectory, Mapping):
        if trajectory.get("format") == BINARY_FORMAT:
            positions = (trajectory.get("arrays") or {}).get("positions") or {}
            return (positions.get("shape") or [None])[0]  # the renderer reports gaps
        frames = trajectory.get("frames")
        return len(frames) if isinstance(frames, list | tuple) else None
    return len(trajectory) if isinstance(trajectory, list | tuple) else None


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


def _complete_structure(structure: dict[str, Any]) -> dict[str, Any]:
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
        import kemio

        # pymatgen's orientation (c along z) for Cartesian xyz site coords
        matrix = kemio.lattice_from_parameters(
            *(float(lattice[key]) for key in cell_params), c_along_z=True
        )
        completed["lattice"] = {**lattice, "matrix": matrix.tolist()}
    sites = []
    for site_idx, site in enumerate(structure.get("sites", [])):
        species = [
            {"element": specie, "occu": 1.0}
            if isinstance(specie, str)
            else {"occu": 1.0, **specie}
            for specie in site.get("species", [])
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


def _complete_frames(trajectory_data: dict[str, Any]) -> dict[str, Any]:
    """Complete every frame of a ``{"frames": [{"structure", "step"?, ...}]}``
    dict (structures as dicts or objects); ``step`` (x axis of the property
    plots) defaults to the frame index.

    Raises:
        TypeError: If there is no ``frames`` list.
        ValueError: If a frame is not a dict with a ``structure`` key.
    """
    if not isinstance(input_frames := trajectory_data.get("frames"), list | tuple):
        raise TypeError(
            f"trajectory dicts need a 'frames' list, got keys={list(trajectory_data)}"
        )
    frames = []
    for frame_idx, frame in enumerate(input_frames):
        if not isinstance(frame, dict) or "structure" not in frame:
            keys = list(frame) if isinstance(frame, dict) else type(frame).__name__
            raise ValueError(
                f"trajectory frame {frame_idx} must be a structure or a dict with "
                f"a 'structure' key, got {keys}"
            )
        structure = frame["structure"]
        if not isinstance(structure, dict):
            structure = structure_dict(structure)
        completed = {
            **frame,
            "step": frame.get("step", frame_idx),
            "structure": _complete_structure(structure),
        }
        frames.append(_attach_forces(completed, frame_idx))
    return {**trajectory_data, "frames": frames}


def _frames_trajectory(
    items: list[Any] | tuple[Any, ...], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Convert a sequence of structures or ``{"structure", **props}`` dicts to a
    matterviz trajectory dict with one frame per item.
    """
    if envelope := trajectory_to_envelope(items, metadata):
        return envelope
    frames: list[dict[str, Any]] = []
    for step_idx, item in enumerate(items):
        structure, properties = frame_input(item)
        frame = {
            "structure": structure_dict(structure),
            "step": properties.get("step", step_idx),
        }
        if properties:
            frame["metadata"] = properties
        frames.append(_attach_forces(frame, step_idx))
    return {"frames": frames, "metadata": metadata}


def _trajectory_payload(trajectory: Any) -> dict[str, Any]:
    """Synced form of a trajectory: a ``matterviz-binary-v1`` envelope when the binary
    format can hold it (fixed species and order, numeric per-frame scalars,
    per-atom 3-vectors), else matterviz's JSON trajectory format, which also
    carries disordered sites, varying composition, string or tensor metadata.
    """
    from kemio.interop import has_class, is_structure_like, to_trajectory

    if is_binary_envelope(trajectory, "trajectory"):
        return trajectory
    is_pymatgen = has_class(trajectory, "pymatgen.core.trajectory.Trajectory")
    is_dict = isinstance(trajectory, dict)
    if is_pymatgen or (is_dict and trajectory.get("@class") == "Trajectory"):
        step = trajectory.time_step if is_pymatgen else trajectory.get("time_step")
        if step is not None and not (_is_number(step) and 0 < float(step) < np.inf):
            raise ValueError(f"time_step value must be finite and > 0, got {step!r}")
        # kemio reads a pymatgen Trajectory or its as_dict() (no pymatgen needed)
        items = [
            frame_item(frame, drop=("time_step",))
            for frame in to_trajectory(trajectory)
        ]
        result = _frames_trajectory(items, {})
        if step is not None:
            result["time_step"] = {"value": float(step), "unit": "fs"}
    elif is_dict:
        frames = trajectory.get("frames")
        if isinstance(frames, list) and frames and all(map(is_structure_like, frames)):
            metadata = trajectory.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            result = _frames_trajectory(frames, metadata)
        else:
            result = _complete_frames(trajectory)
    elif isinstance(trajectory, list | tuple):
        result = _frames_trajectory(trajectory, {})
    elif is_structure_like(trajectory):
        result = _frames_trajectory([trajectory], {})
    else:
        raise TypeError(
            f"Unsupported trajectory type: {type(trajectory)}. "
            "Expected list of structures, single structure, or trajectory dict."
        )
    if is_binary_envelope(result, "trajectory"):
        return result
    return normalize_plot_json(result, "trajectory")


class TrajectoryWidget(StructureVizTraits, TrajectoryTraits, MatterVizWidget):
    """MatterViz widget for visualizing molecular dynamics and geometry optimization
    trajectories in Python notebooks.

    Accepts trajectory data directly (list of structures, dict with frames) or a
    ``data_url`` the frontend fetches and parses itself (XYZ/extXYZ, ASE .traj,
    XDATCAR, LAMMPS dump, pymatgen ``Trajectory`` JSON; ``.gz``/``.zip`` compressed).
    A binary ``trajectory`` holds ``memoryview`` buffers, so ``to_dict()`` isn't plain
    JSON (``json.dumps`` needs a ``default`` for them).

    ``trajectory`` is a dict with ``frames`` (structures as dicts or objects), a
    pymatgen ``Trajectory`` or its ``as_dict()`` (read by kemio), a list of
    structures (kemio, pymatgen, ASE, PhonopyAtoms) or kemio ``io.Frame``s, or a
    list of ``{"structure": struct, "step": 10, "energy": 1.23, ...}`` dicts
    (``step``, else the frame index, is the plots' x axis; structure info and
    ASE calculator results add to the properties). Per-atom forces under a
    frame's ``force``/``forces`` key (or an ASE calculator's results) move to
    site ``properties.force`` and add ``force_max``/``force_norm`` to the frame
    metadata, as matterviz's own parsers do. Six-component or 3x3 frame
    stresses travel as float64 ``frame_stress`` (order and units kept).
    Structure sequences and pymatgen Trajectories sync as a
    ``matterviz-binary-v1`` envelope when representable (custom bundles must
    support it), serialized trajectory dicts as JSON.

    Examples:
        Basic usage with a list of structures (here NaCl under strain):
        >>> import numpy as np
        >>> import kemio
        >>> import pymatviz as pmv
        >>> species, coords = ["Na", "Cl"], [[0] * 3, [0.5] * 3]
        >>> trajectory_data = [
        ...     kemio.Structure(np.eye(3) * a_len, species, coords, fractional=True)
        ...     for a_len in (5.5, 5.6, 5.7)
        ... ]
        >>> widget = pmv.trajectory(trajectory=trajectory_data)
        >>> widget

        With per-frame properties (scalars are plotted; per-atom ``forces`` of shape
        (n_sites, 3) are drawn as vectors with their max/RMS norms plotted):
        >>> forces = np.array([[0.1, 0, 0], [-0.1, 0, 0]])
        >>> trajectory_with_props = [
        ...     {"structure": trajectory_data[0], "energy": -1.23, "forces": forces},
        ...     {"structure": trajectory_data[1], "energy": -1.25, "forces": -forces},
        ... ]
        >>> widget = pmv.trajectory(trajectory=trajectory_with_props)

        A pymatgen ``Trajectory`` (e.g. ``Trajectory.from_file("XDATCAR")``) syncs as
        binary arrays, its ``as_dict()`` as JSON.

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

    trajectory = normalized_trait(_trajectory_payload)  # on every assignment
    # Two-way synced displayed frame, in range of the frames (replacing the trajectory
    # with a shorter one clamps it to the last frame)
    current_step_idx = JsonInt(0).tag(sync=True)

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
    data_inspector_open = JsonBool(default_value=True).tag(sync=True)

    step_labels = optional_trait(tl.Union, trait_types=[tl.Int(), tl.List()])

    _positional_traits = ("trajectory",)

    @validator("current_step_idx")
    def _check_step_idx(self, proposal: dict[str, Any]) -> int:
        """Reject steps beyond the frames (clamping frontend ones)."""
        n_steps = n_frames(self.trajectory)
        return checked_position(
            self, "current_step_idx", proposal["value"], n_steps, "frames"
        )

    @tl.observe("trajectory")
    def _clamp_step_idx(self, change: dict[str, Any]) -> None:
        """Move the step to the last frame of a shorter new trajectory."""
        self.current_step_idx = clamped_position(
            self.current_step_idx, n_frames(change["new"])
        )
