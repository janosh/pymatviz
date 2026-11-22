"""Encode matterviz-binary-v1 envelopes with little-endian anywidget comm buffers."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

from pymatviz.widgets._normalize import normalize_plot_json


BINARY_FORMAT = "matterviz-binary-v1"
TRAJECTORY_RESERVED = {"positions", "numbers", "lattice", "steps"}
# kemio Structure info keys describing the structure, not frame metadata
STRUCTURE_INFO_KEYS = ("properties", "spin_multiplicity")
MAX_ATOMIC_NUMBER = 118


def is_binary_envelope(value: Any, kind: str) -> bool:
    """Recognize tagged envelopes and reject a mismatched kind.

    Raises:
        TypeError: If the envelope belongs to a different widget.
    """
    if not isinstance(value, Mapping) or value.get("format") != BINARY_FORMAT:
        return False
    if (got := value.get("kind")) != kind:
        raise TypeError(f"Binary envelope kind must be {kind!r}, got {got!r}")
    return True


def binary_array(
    values: Any,
    dtype: str,
    *,
    name: str = "array",
    shape: tuple[int | None, ...] | None = None,
    finite: bool = True,
) -> dict[str, Any]:
    """Borrow or cast a numeric array into a contiguous little-endian wire buffer.

    Raises:
        ValueError: For invalid shape, dtype, or a non-finite wire value.
    """
    arr = np.asarray(values)
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric array, got {arr.dtype}")
    if shape is not None:
        # strict=False is safe: a length mismatch already fails the ndim check
        size_pairs = zip(shape, arr.shape, strict=False)
        wrong_size = any(want not in (None, got) for want, got in size_pairs)
        if arr.ndim != len(shape) or wrong_size:
            expected = ", ".join("*" if size is None else str(size) for size in shape)
            raise ValueError(
                f"{name} must be a numeric array of shape [{expected}], got {arr.shape}"
            )
    with np.errstate(over="ignore"):
        wire = np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<"))
    if finite and wire.dtype.kind == "f" and not (valid := np.isfinite(wire)).all():
        idx = valid.argmin()
        if np.isfinite(arr.flat[idx]):
            raise ValueError(f"{name} beyond {dtype} range at flat index {idx}")
        raise ValueError(f"{name} has non-finite value at flat index {idx}")
    return {
        "dtype": dtype,
        "shape": list(wire.shape),
        "data": memoryview(wire.ravel()).cast("B"),
    }


# Series with at least this many points sync as binary plot-series envelopes: as JSON
# a 1e6-point scatter is ~41 MB and ~0.3 s of Python-side list building; binary 16 MB
BINARY_SERIES_MIN_POINTS = 10_000
# numeric columns sent as arrays (x and y must be finite, color/size may hold NaN gaps)
_SERIES_COLUMNS = {"x": True, "y": True, "color_values": False, "size_values": False}


def plot_series_envelope(
    entry: Mapping[str, Any], name: str, keys: tuple[str, ...] = ("x", "y")
) -> dict[str, Any] | None:
    """Binary plot-series envelope for a large numeric series, else None (stays JSON).

    The required ``keys`` (``x``/``y``, ``x``/``y``/``z`` for 3D scatter, or the raw
    ``y`` samples of box plots, all finite), numeric 1D columns in ``_SERIES_COLUMNS``
    and the columns of a numeric per-point ``metadata`` DataFrame (as
    ``metadata.<column>``) become float64 arrays, every other field stays JSON. A
    series whose ``keys`` aren't all numeric, or with fewer than
    ``BINARY_SERIES_MIN_POINTS`` points, returns None.

    Raises:
        ValueError: If column lengths differ or required columns hold non-finite
            values.
    """
    import pandas as pd

    from pymatviz.widgets._normalize import check_metadata_frame, metadata_records

    finite = _SERIES_COLUMNS | dict.fromkeys(keys, True)
    columns = {
        key: arr
        for key in finite
        if key in entry
        and (arr := np.asarray(entry[key])).dtype.kind in "iuf"
        and arr.ndim == 1
    }
    if not set(keys) <= columns.keys():
        return None
    if (n_points := len(columns[keys[0]])) < BINARY_SERIES_MIN_POINTS:
        return None
    # a numeric per-point metadata DataFrame travels as metadata.<column> arrays
    if isinstance(meta := entry.get("metadata"), pd.DataFrame):
        check_metadata_frame(meta, n_points, name)
        if all(dtype.kind in "iuf" for dtype in meta.dtypes):
            columns |= {f"metadata.{col}": meta[col].to_numpy() for col in meta}
            finite |= dict.fromkeys(columns.keys() - finite.keys(), False)
            entry = {key: val for key, val in entry.items() if key != "metadata"}
        else:
            records = metadata_records(meta, n_points, name)
            entry = {**entry, "metadata": records}
    lengths = {key: len(arr) for key, arr in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"{name} columns must share one length, got {lengths}")
    arrays = {
        key: binary_array(arr, "float64", name=f"{name}.{key}", finite=finite[key])
        for key, arr in columns.items()
    }
    fields = normalize_plot_json(
        {key: val for key, val in entry.items() if key not in arrays}, name
    )
    return {"format": BINARY_FORMAT, "kind": "plot-series", "arrays": arrays, **fields}


def series_column(series_item: Mapping[str, Any], key: str) -> Any:
    """A plot series' ``key`` column: a read-only float array decoded from a binary
    plot-series envelope, else the JSON value (None if absent).
    """
    arrays = series_item.get("arrays") or {}
    if is_binary_envelope(series_item, "plot-series") and key in arrays:
        arr = arrays[key]
        dtype = np.dtype(arr["dtype"]).newbyteorder("<")
        return np.frombuffer(arr["data"], dtype=dtype)
    return series_item.get(key)


def volume_envelope(values: Any, **fields: Any) -> dict[str, Any]:
    """Volume envelope for a finite [nx, ny, nz] grid (C order, z fastest).

    The grid is sent as float32: half the wire size of float64, ~7 significant digits
    (far below isosurface rendering resolution), widened to float64 once in JS.

    Raises:
        ValueError: If the grid is not 3D or finite, or overflows float32.
    """
    name = f"volume {fields.get('id')!r} values"
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":  # Numeric strings and boolean masks.
        values = values.astype(float)
    grid = binary_array(values, "float32", name=name, shape=(None, None, None))
    return {
        "format": BINARY_FORMAT,
        "kind": "volume",
        "arrays": {"values": grid},
        **fields,
    }


def _is_number(value: Any) -> bool:
    """Recognize real numeric scalars, including 0-d arrays and excluding booleans."""
    if isinstance(value, int):
        return not isinstance(value, bool)
    return (
        isinstance(value, (float, np.number, np.ndarray))
        and np.ndim(value) == 0
        and np.asarray(value).dtype.kind in "iuf"
    )


def object_metadata(structure: Any) -> dict[str, Any]:
    """Metadata of a structure-like object: a kemio Structure's info (extxyz frame
    keys, ``info["properties"]`` merged in) or ASE info, plus ASE calculator results
    (energy, forces, stress, ...) which pymatgen's ``AseAtomsAdaptor`` drops.
    """
    import kemio

    if isinstance(structure, kemio.Structure):
        info = structure.info
        frame_info = {
            key: val for key, val in info.items() if key not in STRUCTURE_INFO_KEYS
        }
        return frame_info | dict(info.get("properties") or {})
    metadata = getattr(structure, "properties", None) or getattr(structure, "info", {})
    calc = getattr(structure, "calc", None)
    return {**(metadata or {}), **(getattr(calc, "results", None) or {})}


def frame_item(frame: Any, drop: Collection[str] = ()) -> dict[str, Any]:
    """``{"structure", **metadata, **results}`` of a kemio ``io.Frame``, its
    structure's info (minus ``drop`` keys) moved to the item on a copy.
    """
    metadata = object_metadata(frame.structure)
    structure = frame.structure.replace(  # frame metadata goes to the item
        info={
            key: val
            for key, val in frame.structure.info.items()
            if key == "spin_multiplicity"
        }
    )
    results = {
        key: val
        for key in ("energy", "free_energy", "forces", "stress")
        if (val := getattr(frame, key)) is not None
    }
    kept = {key: val for key, val in metadata.items() if key not in drop}
    return {"structure": structure, **kept, **results}


def frame_input(item: Any) -> tuple[Any, Mapping[str, Any]]:
    """Unwrap a frame into its structure and properties: the object's metadata
    overridden by a ``{"structure", **props}`` dict's explicit ones.
    """
    from kemio.io import Frame

    if isinstance(item, Frame):
        item = frame_item(item)
    if isinstance(item, Mapping) and "structure" in item:
        structure = item["structure"]
        props = {key: val for key, val in item.items() if key != "structure"}
        return structure, {**object_metadata(structure), **props}
    return item, object_metadata(item)


def force_stats(forces: np.ndarray) -> dict[str, Any]:
    """Max and RMS per-atom force magnitude for one frame or a stack of frames."""
    norms = np.linalg.norm(forces.astype(float, copy=False), axis=-1)
    return {
        "force_max": norms.max(axis=-1),
        "force_norm": np.sqrt(np.mean(norms**2, axis=-1)),
    }


# Cells below this volume (Å³) are degenerate: such trajectories go to JSON
MIN_CELL_VOLUME = 1e-6
# site data binary channels can't hold, which JSON structures keep
_JSON_SITE_DATA = frozenset({"labels", "oxidation_states"})


def _exact(values: np.ndarray) -> bool:
    """Whether values are real numbers float64 holds exactly (no bools, strings,
    objects or integers beyond 2^53).
    """
    if values.dtype.kind == "f":
        return values.dtype.itemsize <= 8
    if values.dtype.kind not in "iu":
        return False
    return not values.size or int(np.abs(values).max()) <= 2**53


def _channels(traj: Any) -> dict[str, Any] | None:
    """Per-atom vectors (site data, forces as ``force``), per-frame scalars (info,
    energies; ``step`` as matterviz's ``steps`` x axis, NaN where a frame lacks one)
    and ``frame_stress`` of kemio ``TrajectoryArrays``, or None if any needs JSON.
    """
    n_frames, n_sites = traj.positions.shape[:2]
    vectors, scalars = dict(traj.site_data), dict(traj.info)
    # per-atom forces: the results', a site data column's or a frame property's
    force_sources = [
        val
        for source in (vectors, scalars)
        for key in ("force", "forces")
        if (val := source.pop(key, None)) is not None
    ]
    if traj.forces is not None:
        force_sources.append(traj.forces)
    if len(force_sources) > 1:
        return None
    if force_sources:
        vectors["force"] = force_sources[0]
    scalars |= {
        key: val
        for key in ("energy", "free_energy")
        if (val := getattr(traj, key)) is not None
    }
    if "steps" in scalars or (vectors.keys() & scalars.keys()):
        return None
    scalars = {"steps" if key == "step" else key: val for key, val in scalars.items()}
    stress = traj.stress
    per_frame = [*vectors.values(), *([stress] if stress is not None else [])]
    reserved = TRAJECTORY_RESERVED - {"steps"}
    reserved_clash = (vectors.keys() | scalars.keys()) & reserved
    stress_clash = stress is not None and "frame_stress" in scalars
    stats_clash = vectors.keys() & {"force_max", "force_norm"}
    if reserved_clash or stress_clash or stats_clash:
        return None

    vectors_ok = {val.shape for val in vectors.values()} <= {(n_frames, n_sites, 3)}
    scalars_ok = {val.shape for val in scalars.values()} <= {(n_frames,)}
    exact_values = all(_exact(val) for val in [*vectors.values(), *scalars.values()])
    if not (vectors_ok and scalars_ok and exact_values):
        return None

    # isfinite and the NaN-frame check below need the exact numeric dtypes checked above
    steps_finite = "steps" not in scalars or np.isfinite(scalars["steps"]).all()
    stress_ok = stress is None or _exact(stress)
    if not (steps_finite and stress_ok):
        return None

    # a frame without a vector or stress (all NaN) needs JSON, while non-finite
    # values within a frame raise in binary_array
    if any(np.isnan(val.reshape(n_frames, -1)).all(axis=1).any() for val in per_frame):
        return None
    if (force := vectors.get("force")) is not None:
        for name, computed in force_stats(force).items():
            given = scalars.get(name)
            scalars[name] = (
                computed
                if given is None
                else np.where(np.isnan(given), computed, given)
            )
    arrays = {
        name: binary_array(values, "float32", name=name)
        for name, values in vectors.items()
    }
    for name, values in scalars.items():
        arrays[name] = binary_array(values, "float64", name=name, finite=False)
    if stress is not None:
        arrays["frame_stress"] = binary_array(
            stress.reshape(n_frames, -1), "float64", name="stress"
        )
    return arrays


def trajectory_to_envelope(
    trajectory: Any,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Binary envelope of a trajectory (anything kemio's ``to_trajectory`` takes)
    from ``kemio.interop.trajectory_arrays``, or None for data only JSON holds:
    frames that don't stack, net charges, labels, oxidation states, a non-default
    spin, degenerate cells or quantities without a binary channel.
    """
    from kemio.errors import NotStackableError
    from kemio.interop import trajectory_arrays

    try:
        traj = trajectory_arrays(trajectory)
    except NotStackableError:
        return None
    numbers = traj.numbers.astype(int)
    traj = traj._replace(info=dict(traj.info))
    traj.info.pop("time_step", None)  # the widget's time_step, not a frame series
    spin = traj.info.pop("spin_multiplicity", None)
    cells = traj.cells
    # matterviz renders Z in 1..118 (a DummySpecies' Z is a hash)
    z_max = MAX_ATOMIC_NUMBER
    bad_numbers = not len(numbers) or not 1 <= numbers.min() <= numbers.max() <= z_max
    json_only_sites = traj.charge.any() or traj.site_data.keys() & _JSON_SITE_DATA
    wrong_spin = spin is not None and np.any(spin != 1 + numbers.sum() % 2)
    if bad_numbers or json_only_sites or wrong_spin:
        return None

    if cells is not None and (np.abs(np.linalg.det(cells)) < MIN_CELL_VOLUME).any():
        return None
    if (channels := _channels(traj)) is None:
        return None
    n_frames = len(traj.positions)
    with np.errstate(over="ignore"):  # The wire check below rejects overflow.
        positions = traj.positions.astype("<f4")
    arrays = {
        "positions": binary_array(
            positions, "float32", name="positions", shape=(n_frames, len(numbers), 3)
        ),
        "numbers": binary_array(numbers, "uint8", name="numbers"),
    }
    fields: dict[str, Any] = {}
    if cells is not None:
        fields["pbc"] = list(traj.pbc)
        if (cells == cells[0]).all():
            cells = cells[0]
        shape = (3, 3) if cells.ndim == 2 else (n_frames, 3, 3)
        arrays["lattice"] = binary_array(cells, "float64", name="lattice", shape=shape)
    arrays.update(channels)
    if metadata:
        fields["metadata"] = normalize_plot_json(dict(metadata), "trajectory metadata")
    return {"format": BINARY_FORMAT, "kind": "trajectory", "arrays": arrays, **fields}
