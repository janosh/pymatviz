"""Encoders for matterviz's ``matterviz-binary-v1`` anywidget transfer.

Heavy arrays travel as raw little-endian buffers (ipywidgets extracts every nested
``memoryview`` of a synced trait into a binary comm buffer, which anywidget hands to
JS as a ``DataView``) inside a small JSON envelope::

    {
        "format": "matterviz-binary-v1",
        "kind": "trajectory" | "volume",
        "arrays": {name: {"dtype", "shape", "data"}},
        **fields,
    }

The encoders check their inputs (shapes, finite values, atomic numbers) the way
matterviz's JS decoders (``src/lib/io/binary-arrays.ts``,
``src/lib/trajectory/binary.ts``, ``src/lib/isosurface/binary.ts``) do, so bad data
raises on assignment in Python instead of in JS. Ready envelopes (e.g. a widget's own
synced value) are passed through after checking only their kind.
Wire format: matterviz ``extensions/anywidget/readme.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


BINARY_FORMAT = "matterviz-binary-v1"
TRAJECTORY_RESERVED = ("positions", "numbers", "lattice", "steps")
MAX_ATOMIC_NUMBER = 118


def binary_array(values: Any, dtype: str) -> dict[str, Any]:
    """Encode an array as a ``matterviz-binary-v1`` entry (C order, little-endian).

    ``data`` is a byte ``memoryview`` of the contiguous array, so no copy is made when
    ``values`` already is a C-contiguous little-endian array of ``dtype``.
    """
    arr = np.ascontiguousarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
    return {"dtype": dtype, "shape": list(arr.shape), "data": memoryview(arr).cast("B")}


def is_binary_envelope(value: Any, kind: str) -> bool:
    """Whether ``value`` is tagged as a binary envelope (the tag alone decides, like
    matterviz's ``is_binary_envelope``: a broken envelope never falls back to JSON).

    Raises:
        TypeError: For an envelope of another kind (e.g. a volume as a trajectory).
    """
    if not isinstance(value, Mapping) or value.get("format") != BINARY_FORMAT:
        return False
    if (got := value.get("kind")) != kind:
        raise TypeError(f"Binary envelope kind must be {kind!r}, got {got!r}")
    return True


def _checked(
    name: str, values: Any, shape: tuple[int | None, ...], *, finite: bool = True
) -> np.ndarray:
    """``values`` as a numeric array after checking its shape (None matches any size)
    and, if ``finite``, that all values are finite.

    Raises:
        ValueError: For a non-numeric dtype, a wrong shape or a non-finite value.
    """
    arr = np.asarray(values)
    if (
        arr.dtype.kind not in "iuf"
        or arr.ndim != len(shape)
        or any(
            want not in (None, got) for want, got in zip(shape, arr.shape, strict=True)
        )
    ):
        want_str = ", ".join("*" if size is None else str(size) for size in shape)
        raise ValueError(
            f"{name} must be a numeric array of shape [{want_str}], got {arr.dtype} "
            f"{list(arr.shape)}"
        )
    if finite and (bad := np.flatnonzero(~np.isfinite(arr))).size:
        raise ValueError(f"{name} has non-finite value at flat index {bad[0]}")
    return arr


def _float32(name: str, arr: np.ndarray) -> np.ndarray:
    """Finite float ``arr`` cast to little-endian float32 (no copy if it already is).

    Raises:
        ValueError: If a value overflows float32 to inf.
    """
    with np.errstate(over="ignore"):
        arr32 = arr.astype("<f4", copy=False)
    if not np.isfinite(arr32).all():
        raise ValueError(
            f"{name} beyond float32 range (max |value| = {np.abs(arr).max():.3g})"
        )
    return arr32


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
    grid = _checked(name, np.asarray(values, dtype=float), (None, None, None))
    grid32 = _float32(name, grid)
    return {
        "format": BINARY_FORMAT,
        "kind": "volume",
        "arrays": {"values": binary_array(grid32, "float32")},
        **fields,
    }


def trajectory_envelope(
    positions: Any,
    numbers: Any,
    *,
    lattice: Any = None,
    pbc: Any = None,
    steps: Any = None,
    metadata: Mapping[str, Any] | None = None,
    time_step: Mapping[str, Any] | None = None,
    vectors: Mapping[str, Any] | None = None,
    scalars: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Trajectory envelope from flat arrays.

    C-contiguous little-endian inputs of the wire dtype are borrowed, not copied (like
    matterviz's zero-copy decode), so mutating them afterwards is undefined behavior.

    Dtypes: positions and per-atom ``vectors`` as float32 (Cartesian Å: float32's
    relative precision 6e-8 is < 1e-5 Å even for 100 Å unwrapped coordinates, and
    halves the wire size vs float64; JS widens each frame exactly on read), atomic
    numbers as uint8, lattice and per-frame ``scalars`` as float64 (tiny, and energies
    need the digits), steps as int32 when they fit, else float64.

    Args:
        positions: [n_frames, n_atoms, 3] Cartesian coordinates in Å.
        numbers: [n_atoms] atomic numbers, fixed atom order.
        lattice: [3, 3] fixed or [n_frames, 3, 3] per-frame cell (rows a, b, c).
        pbc: 3 bools (needs a lattice).
        steps: [n_frames] step labels (default 0..n_frames-1 in JS).
        metadata: Trajectory-level metadata (numpy values become JSON primitives).
        time_step: ``{"value": float > 0, "unit": "fs" | "ps" | "ns"}``.
        vectors: {name: [n_frames, n_atoms, 3]} per-atom channels (e.g. force).
        scalars: {name: [n_frames]} per-frame scalars (e.g. energy; NaN = missing).

    Raises:
        ValueError: For arrays matterviz would reject (wrong shapes, non-finite
            values (also after the float32 cast), atomic numbers outside 1..118, no
            frames or atoms, a time_step value not finite and > 0) or channel
            names that clash with each other or the reserved arrays.
    """
    from pymatviz.widgets._normalize import normalize_plot_json

    positions = _checked("positions", positions, (None, None, 3))
    n_frames, n_atoms = positions.shape[:2]
    if n_frames < 1 or n_atoms < 1:
        raise ValueError(
            f"trajectory needs at least 1 frame and 1 atom, got {n_frames=}, {n_atoms=}"
        )
    numbers = _checked("numbers", numbers, (n_atoms,))
    if not _numbers_ok(numbers):  # before the uint8 cast wraps e.g. 300 to 44
        raise ValueError(
            f"atomic numbers must be integers in 1..{MAX_ATOMIC_NUMBER}, got "
            f"{np.unique(numbers).tolist()}"
        )
    vectors, scalars = dict(vectors or {}), dict(scalars or {})
    if clash := sorted(
        (vectors.keys() & scalars.keys())
        | ((vectors.keys() | scalars.keys()) & set(TRAJECTORY_RESERVED))
    ):
        raise ValueError(
            f"channels {clash} are both vectors and scalars or reserved arrays "
            f"{TRAJECTORY_RESERVED}"
        )
    arrays = {
        "positions": binary_array(_float32("positions", positions), "float32"),
        "numbers": binary_array(numbers, "uint8"),
    }
    if lattice is not None:
        cell_shape = (3, 3) if np.ndim(lattice) == 2 else (n_frames, 3, 3)
        arrays["lattice"] = binary_array(
            _checked("lattice", lattice, cell_shape), "float64"
        )
    if steps is not None:
        steps = _checked("steps", steps, (n_frames,))
        fits_int32 = steps.dtype.kind in "iu" and np.all(np.abs(steps) < 2**31)
        arrays["steps"] = binary_array(steps, "int32" if fits_int32 else "float64")
    for name, vals in vectors.items():
        vec = _checked(name, vals, (n_frames, n_atoms, 3))
        arrays[name] = binary_array(_float32(name, vec), "float32")
    for name, vals in scalars.items():  # NaN = missing
        arrays[name] = binary_array(
            _checked(name, vals, (n_frames,), finite=False), "float64"
        )
    payload: dict[str, Any] = {
        "format": BINARY_FORMAT,
        "kind": "trajectory",
        "arrays": arrays,
    }
    if pbc is not None:
        payload["pbc"] = [bool(flag) for flag in pbc]
    if metadata:
        payload["metadata"] = normalize_plot_json(dict(metadata), "trajectory metadata")
    if time_step is not None:
        dt_val = time_step.get("value")
        if not (_is_number(dt_val) and 0 < np.asarray(dt_val, dtype=float) < np.inf):
            raise ValueError(f"time_step value must be finite and > 0, got {dt_val!r}")
        payload["time_step"] = normalize_plot_json(dict(time_step), "time_step")
    return payload


# === Trajectory input conversion ===
# Every source (pymatgen structures and Trajectory, ASE Atoms, frame dicts) is
# normalized to per-frame records {numbers, positions, cell, pbc, vectors, scalars};
# a None record means the frame needs the JSON format.


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a real (non-bool) numeric scalar, incl. 0-d arrays."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    return (
        isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "iuf"
    )


def _vec3(value: Any, n_atoms: int) -> np.ndarray | None:
    """``value`` as a numeric (n_atoms, 3) array, or None if it isn't one."""
    arr = np.asarray(value)
    return arr if arr.shape == (n_atoms, 3) and arr.dtype.kind in "iuf" else None


def object_metadata(structure: Any) -> dict[str, Any]:
    """Metadata of a structure-like object: its properties/info, plus ASE calculator
    results (energy, forces, stress, ...) which pymatgen's ``AseAtomsAdaptor`` drops.
    """
    metadata = dict(
        getattr(structure, "properties", None) or getattr(structure, "info", None) or {}
    )
    calc = getattr(structure, "calc", None)
    return metadata | dict(getattr(calc, "results", None) or {})


def _pmg_numbers(
    species: Sequence[Any], charge: float | None, spin: float | None
) -> np.ndarray | None:
    """Atomic numbers of pymatgen species, or None if binary would drop information:
    oxidation states, a net charge or a non-default spin multiplicity.
    """
    if any(getattr(sp, "oxi_state", None) is not None for sp in species):
        return None
    numbers = np.array([getattr(sp, "Z", 0) for sp in species])
    if (charge or 0) != 0 or spin not in (None, 1 + int(numbers.sum()) % 2):
        return None
    return numbers


def _record(
    numbers: np.ndarray | None,
    positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: tuple[bool, ...] | None,
    site_props: Mapping[str, Any],
    props: Mapping[str, Any],
) -> dict[str, Any] | None:
    """One frame's record, or None if it has no or invalid atoms, site properties
    other than per-atom 3-vectors, or props other than numeric scalars and (n_atoms,
    3) ``force``/``forces`` (strings, stress tensors, ...: JSON frame metadata only).
    """
    if numbers is None or len(numbers) == 0 or not _numbers_ok(numbers):
        return None
    n_atoms = len(numbers)
    vectors: dict[str, np.ndarray] = {}
    for key, vals in site_props.items():
        if (arr := _vec3(vals, n_atoms)) is None:
            return None
        channel = "force" if key == "forces" else key
        if channel in vectors:  # both force and forces: neither may silently win
            return None
        vectors[channel] = arr
    scalars: dict[str, float] = {}
    for key, val in props.items():
        if key in ("force", "forces"):
            if (arr := _vec3(val, n_atoms)) is None or "force" in vectors:
                return None
            vectors["force"] = arr
        elif _is_number(val):
            if np.asarray(val).dtype.kind != "f" and abs(int(val)) > 2**53:
                return None  # float64 would round it: JSON keeps it exact
            scalars[key] = float(val)
        else:
            return None
    return {
        "numbers": numbers,
        "positions": positions,
        "cell": cell,
        "pbc": pbc,
        "vectors": vectors,
        "scalars": scalars,
    }


def _frame_record(item: Any) -> dict[str, Any] | None:
    """Record of a pymatgen SiteCollection or ASE Atoms, optionally in a
    ``{"structure": ..., **props}`` dict, or None if binary can't hold it (also
    disordered sites, custom labels and cells matterviz must pad).
    """
    from pymatgen.core import SiteCollection, Structure

    structure, props = item, {}
    if isinstance(item, Mapping):
        structure = item.get("structure", item)
        props = {key: val for key, val in item.items() if key != "structure"}
    if isinstance(structure, SiteCollection):
        if not structure.is_ordered or any(
            site.label != site.species_string for site in structure
        ):
            return None
        spin = getattr(structure, "spin_multiplicity", None)  # Molecules only
        numbers = _pmg_numbers(structure.species, structure.charge, spin)
        is_periodic = isinstance(structure, Structure)
        return _record(
            numbers,
            structure.cart_coords,
            structure.lattice.matrix if is_periodic else None,
            tuple(structure.lattice.pbc) if is_periodic else None,
            structure.site_properties,
            props or object_metadata(structure),
        )
    if not (hasattr(structure, "get_positions") and hasattr(structure, "cell")):
        return None
    cell = np.asarray(structure.cell[:], dtype=float)
    pbc = tuple(bool(flag) for flag in structure.pbc)
    if not cell.any() and not any(pbc):
        cell, pbc = None, None  # molecule: matterviz draws it without a cell
    elif abs(np.linalg.det(cell)) < 1e-8:
        return None  # degenerate cell: JSON path pads it with vacuum
    site_props = {
        key: val
        for key, val in structure.arrays.items()
        if key not in ("numbers", "positions")
    }
    return _record(
        np.asarray(structure.numbers),
        structure.get_positions(),
        cell,
        pbc,
        site_props,
        props or object_metadata(structure),
    )


def _pmg_trajectory_records(traj: Any) -> list[dict[str, Any] | None]:
    """Records of a pymatgen ``Trajectory`` from its coordinate arrays (not
    ``list(traj)``, whose Structures wrap sites into the cell and drop crossings), or
    ``[None]`` if it has site properties or a frame_properties count not matching its
    frames. The trajectory is not mutated.
    """
    from pymatgen.core.periodic_table import get_el_sp

    site_props = traj.site_properties
    if any([site_props] if isinstance(site_props, Mapping) else site_props or []):
        return [None]
    species = [get_el_sp(sp) for sp in traj.species]
    numbers = _pmg_numbers(species, traj.charge, traj.spin_multiplicity)
    coords = np.asarray(traj.coords, dtype=float)
    if traj.coords_are_displacement:
        coords = np.asarray(traj.base_positions) + np.cumsum(coords, axis=0)
    n_frames = len(coords)
    frame_props = traj.frame_properties or [{}] * n_frames
    if len(frame_props) != n_frames:
        return [None]
    if traj.lattice is None:
        cells, positions = [None] * n_frames, coords
    else:
        lattice = np.asarray(traj.lattice, dtype=float)
        cells, positions = np.broadcast_to(lattice, (n_frames, 3, 3)), coords @ lattice
    return [
        _record(numbers, pos, cell, None, {}, props)
        for pos, cell, props in zip(positions, cells, frame_props, strict=True)
    ]


def trajectory_to_envelope(
    trajectory: Any, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Binary envelope for a pymatgen ``Trajectory`` or a sequence of frames
    (pymatgen structures, ASE Atoms or ``{"structure": ..., **props}`` dicts), or None
    if the binary format can't hold it: besides per-frame limits (see ``_record``),
    atoms must keep their species and order, and pbc, cell presence and per-atom
    channels must be the same in every frame.
    """
    from pymatgen.core.trajectory import Trajectory

    is_pmg = isinstance(trajectory, Trajectory)
    maybe_records = (
        _pmg_trajectory_records(trajectory)
        if is_pmg
        else [_frame_record(item) for item in trajectory]
    )
    records = [rec for rec in maybe_records if rec is not None]
    if not records or len(records) != len(maybe_records):
        return None
    first = records[0]
    vector_keys = list(first["vectors"])
    for rec in records:
        if (
            not np.array_equal(rec["numbers"], first["numbers"])
            or rec["pbc"] != first["pbc"]
            or (rec["cell"] is None) != (first["cell"] is None)
            or set(rec["vectors"]) != set(vector_keys)
            or rec["scalars"].keys() & set(vector_keys)
            or {*rec["scalars"], *vector_keys} & set(TRAJECTORY_RESERVED)
        ):
            return None
    cells = None
    if first["cell"] is not None:
        cells = np.array([rec["cell"] for rec in records])
        cells = cells[0] if (cells == cells[0]).all() else cells
    vectors = {
        key: np.array([rec["vectors"][key] for rec in records]) for key in vector_keys
    }
    # NaN where a frame lacks a key (matterviz skips non-finite values in plot rows)
    scalar_keys = dict.fromkeys(key for rec in records for key in rec["scalars"])
    scalars = {
        key: np.array([rec["scalars"].get(key, np.nan) for rec in records])
        for key in scalar_keys
    }
    if (force := vectors.get("force")) is not None:
        # max and RMS of per-atom force norms (like matterviz's parsers and the JSON
        # path) where a frame doesn't supply them
        norms = np.linalg.norm(force, axis=2)
        for key, computed in (
            ("force_max", norms.max(axis=1)),
            ("force_norm", np.sqrt(np.mean(norms**2, axis=1))),
        ):
            given = scalars.get(key)
            scalars[key] = (
                computed
                if given is None
                else np.where(np.isnan(given), computed, given)
            )
    time_step = trajectory.time_step if is_pmg else None
    return trajectory_envelope(
        np.array([rec["positions"] for rec in records]),
        first["numbers"],
        lattice=cells,
        pbc=first["pbc"],
        metadata=metadata,
        time_step=None if time_step is None else {"value": time_step, "unit": "fs"},
        vectors=vectors,
        scalars=scalars,
    )
