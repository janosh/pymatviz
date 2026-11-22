"""Encode matterviz-binary-v1 envelopes with little-endian anywidget comm buffers."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np

from pymatviz.widgets._normalize import normalize_plot_json


BINARY_FORMAT = "matterviz-binary-v1"
TRAJECTORY_RESERVED = {"positions", "numbers", "lattice", "steps"}
# matscikit Structure info keys describing the structure, not frame metadata
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
    if shape is not None and (
        arr.ndim != len(shape)
        or any(
            want not in (None, got) for want, got in zip(shape, arr.shape, strict=True)
        )
    ):
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
    if is_binary_envelope(series_item, "plot-series") and key in (
        arrays := series_item.get("arrays") or {}
    ):
        arr = arrays[key]
        return np.frombuffer(
            arr["data"], dtype=np.dtype(arr["dtype"]).newbyteorder("<")
        )
    return series_item.get(key)


def _numbers_ok(numbers: np.ndarray) -> bool:
    """Whether atomic numbers are integers in 1..118, what matterviz can render (a
    DummySpecies' Z is a hash).
    """
    return numbers.dtype.kind in "iu" and bool(
        np.all((numbers >= 1) & (numbers <= MAX_ATOMIC_NUMBER))
    )


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
    """Metadata of a structure-like object: a matscikit Structure's info (extxyz frame
    keys, ``info["properties"]`` merged in) or ASE info, plus ASE calculator results
    (energy, forces, stress, ...) which pymatgen's ``AseAtomsAdaptor`` drops.
    """
    import matscikit as mk

    if isinstance(structure, mk.Structure):
        info = structure.info
        return {
            key: val for key, val in info.items() if key not in STRUCTURE_INFO_KEYS
        } | dict(info.get("properties") or {})
    metadata = getattr(structure, "properties", None) or getattr(structure, "info", {})
    calc = getattr(structure, "calc", None)
    return {**(metadata or {}), **(getattr(calc, "results", None) or {})}


# Cells below this volume (Å³) are degenerate: such frames go to JSON
MIN_CELL_VOLUME = 1e-6


class _Frame(NamedTuple):
    """Geometry and property sources for one structure or ASE frame."""

    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray | None
    pbc: tuple[bool, ...] | None
    site_props: Mapping[str, Any]
    props: Mapping[str, Any]


def frame_item(frame: Any, drop: Collection[str] = ()) -> dict[str, Any]:
    """``{"structure", **metadata, **results}`` of a matscikit ``io.Frame``, its
    structure's info (minus ``drop`` keys) moved to the item on a copy.
    """
    structure = frame.structure.replace()
    metadata = object_metadata(structure)
    structure.info = {  # frame metadata goes to the item, not the structure
        key: val for key, val in structure.info.items() if key == "spin_multiplicity"
    }
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
    from matscikit.io import Frame

    if isinstance(item, Frame):
        item = frame_item(item)
    if isinstance(item, Mapping) and "structure" in item:
        structure = item["structure"]
        props = {key: val for key, val in item.items() if key != "structure"}
        return structure, {**object_metadata(structure), **props}
    return item, object_metadata(item)


def _frame(item: Any) -> _Frame | None:
    """Extract geometry, or select JSON for richer structures."""
    import matscikit as mk
    from matscikit.interop import has_class, is_ase_atoms, to_structure

    structure, props = frame_input(item)
    if isinstance(item, Mapping) and len(item) > 1 and object_metadata(structure):
        return None  # JSON keeps frame metadata separate from structure properties.
    if has_class(structure, "pymatgen.core.structure.SiteCollection"):
        structure = to_structure(structure)  # site properties become site data
    if isinstance(structure, mk.Structure):
        # disordered sites, labels, oxidation states, a net charge or a non-default
        # spin need JSON; other site data travel as per-atom vectors
        if not structure.is_ordered:
            return None
        numbers = structure.numbers.astype(int)
        spin = structure.info.get("spin_multiplicity")
        site_data = dict(structure.site_data)
        if (
            site_data.keys() & {"labels", "oxidation_states"}
            or structure.charge
            or (spin is not None and spin != 1 + int(numbers.sum()) % 2)
        ):
            return None
        periodic = any(structure.pbc)
        if periodic and abs(np.linalg.det(structure.lattice)) < MIN_CELL_VOLUME:
            return None
        return _Frame(
            numbers,
            structure.positions,
            structure.lattice if periodic else None,
            structure.pbc if periodic else None,
            site_data,
            props,
        )
    if not is_ase_atoms(structure) or structure.constraints:
        return None
    cell, pbc = np.asarray(structure.cell), tuple(structure.pbc.tolist())
    if not cell.any() and not any(pbc):
        cell, pbc = None, None
    elif structure.cell.volume < MIN_CELL_VOLUME:
        return None
    site_props = {
        key: val
        for key, val in structure.arrays.items()
        if key not in ("numbers", "positions", "momenta")
    }
    if "initial_magmoms" in site_props:  # The adaptor maps these to site magmom.
        return None
    if "momenta" in structure.arrays:  # velocities, as the JSON path sends them
        site_props["velocities"] = structure.get_velocities()
    return _Frame(structure.numbers, structure.positions, cell, pbc, site_props, props)


def _channels(
    frames: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    n_frames: int,
    n_atoms: int,
) -> dict[str, Any] | None:
    """Encode site vectors, frame scalars (NaN if missing), and optional stress."""
    vectors: dict[str, list[np.ndarray]] = {}
    scalars: dict[str, np.ndarray] = {}
    stress: list[np.ndarray] = []
    for idx, (site_props, props) in enumerate(frames):
        seen: set[str] = set()
        for key, value in (*site_props.items(), *props.items()):
            # user steps are the reserved steps array matterviz plots against
            name = {"forces": "force", "step": "steps"}.get(key, key)
            if name in seen or key in TRAJECTORY_RESERVED:
                return None
            seen.add(name)
            frame_property = name != "force" and key not in site_props
            if frame_property and _is_number(value):
                if name in vectors or (
                    np.asarray(value).dtype.kind != "f" and abs(int(value)) > 2**53
                ):
                    return None
                if name not in scalars:
                    scalars[name] = np.full(n_frames, np.nan)
                scalars[name][idx] = value
                continue
            if frame_property and name != "stress":
                return None
            try:
                array = np.asarray(value)
            except ValueError:  # Ragged properties can still travel as JSON.
                return None
            if array.dtype.kind not in "iuf" or name in scalars:
                return None
            if frame_property:
                if (
                    array.shape not in ((6,), (3, 3))
                    or (stress and array.size != stress[0].size)
                    or array.dtype.itemsize > 8
                    or (
                        array.dtype.kind in "iu"
                        and (int(array.max()) > 2**53 or int(array.min()) < -(2**53))
                    )
                ):
                    return None
                stress.append(array.ravel())
            elif array.shape == (n_atoms, 3):
                vectors.setdefault(name, []).append(array)
            else:
                return None
    if (
        any(len(series) != n_frames for series in vectors.values())
        or ("steps" in scalars and np.isnan(scalars["steps"]).any())
        or (
            stress
            and (
                len(stress) != n_frames
                or "frame_stress" in scalars
                or "frame_stress" in vectors
            )
        )
    ):
        return None
    stacked = {name: np.stack(series) for name, series in vectors.items()}
    if (force := stacked.get("force")) is not None:
        if vectors.keys() & {"force_max", "force_norm"}:
            return None
        for name, computed in force_stats(force).items():
            given = scalars.get(name)
            scalars[name] = (
                computed
                if given is None
                else np.where(np.isnan(given), computed, given)
            )
    arrays = {
        name: binary_array(values, "float32", name=name)
        for name, values in stacked.items()
    }
    for name, values in scalars.items():
        arrays[name] = binary_array(values, "float64", name=name, finite=False)
    if stress:
        arrays["frame_stress"] = binary_array(stress, "float64", name="stress")
    return arrays


def force_stats(forces: np.ndarray) -> dict[str, Any]:
    """Max and RMS per-atom force magnitude for one frame or a stack of frames."""
    norms = np.linalg.norm(forces.astype(float, copy=False), axis=-1)
    return {
        "force_max": norms.max(axis=-1),
        "force_norm": np.sqrt(np.mean(norms**2, axis=-1)),
    }


def trajectory_to_envelope(
    trajectory: Any,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Encode supported trajectories, or return None for JSON-only data."""
    fields: dict[str, Any] = {}
    if metadata:
        fields["metadata"] = normalize_plot_json(dict(metadata), "trajectory metadata")

    frames: list[_Frame] = []
    for item in trajectory:
        frame = _frame(item)
        if frame is None:
            return None
        if frames:
            first = frames[0]
            if (
                not np.array_equal(frame.numbers, first.numbers)
                or frame.pbc != first.pbc
                or (frame.cell is None) != (first.cell is None)
            ):
                return None
        elif not len(frame.numbers) or not _numbers_ok(frame.numbers):
            return None
        frames.append(frame)
    if not frames:
        return None
    numbers = frames[0].numbers
    if frames[0].pbc is not None:
        fields["pbc"] = list(frames[0].pbc)
    channels = _channels(
        ((frame.site_props, frame.props) for frame in frames),
        len(frames),
        len(numbers),
    )
    if channels is None:
        return None
    with np.errstate(over="ignore"):  # The wire check below rejects overflow.
        positions = np.stack([frame.positions for frame in frames], dtype="<f4")
    cell = (
        np.array([frame.cell for frame in frames])
        if frames[0].cell is not None
        else None
    )
    arrays = {
        "positions": binary_array(
            positions,
            "float32",
            name="positions",
            shape=(len(positions), len(numbers), 3),
        ),
        "numbers": binary_array(numbers, "uint8", name="numbers"),
    }
    if cell is not None:
        if cell.ndim == 3 and (cell == cell[0]).all():
            cell = cell[0]
        shape = (3, 3) if cell.ndim == 2 else (len(positions), 3, 3)
        arrays["lattice"] = binary_array(cell, "float64", name="lattice", shape=shape)
    arrays.update(channels)
    return {"format": BINARY_FORMAT, "kind": "trajectory", "arrays": arrays, **fields}
