"""Tests for the matterviz-binary-v1 trajectory and volume encoders."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest

from pymatviz import StructureWidget, TrajectoryWidget
from pymatviz.widgets._binary import (
    BINARY_FORMAT,
    binary_array,
    trajectory_envelope,
    trajectory_to_envelope,
    volume_envelope,
)
from pymatviz.widgets._headless import (
    _BUFFER_KEY,
    _REVIVE_BUFFERS_JS,
    _widget_data_json,
)
from tests.widgets.conftest import decode_envelope


_RNG = np.random.default_rng(seed=0)
_POSITIONS = _RNG.random((4, 3, 3)) * 10
_NUMBERS = [8, 1, 1]
_CELL = np.eye(3) * 10


def _envelope(**kwargs: Any) -> dict[str, Any]:
    """A valid 4-frame x 3-atom envelope with per-frame cells and channels."""
    defaults: dict[str, Any] = {
        "lattice": np.array([_CELL * (1 + 0.01 * idx) for idx in range(4)]),
        "pbc": [True, True, False],
        "steps": np.arange(0, 40, 10),
        "metadata": {"source": "md"},
        "time_step": {"value": 0.5, "unit": "fs"},
        "vectors": {"force": _POSITIONS / 10},
        "scalars": {"energy": [-1.0, -1.5, np.nan, -2.0]},
        "positions": _POSITIONS,
        "numbers": _NUMBERS,
    }
    return trajectory_envelope(**defaults | kwargs)


@pytest.mark.parametrize(
    ("steps", "steps_dtype"),
    [(np.arange(0, 40, 10), "int32"), (np.array([0, 10, 20, 2**40]), "float64")],
)
def test_trajectory_envelope_round_trip(steps: np.ndarray, steps_dtype: str) -> None:
    """Wire bytes decode back to the inputs with the documented dtypes and shapes;
    steps beyond int32 fall to float64 (exact below 2^53).
    """
    payload = _envelope(steps=steps)
    assert payload["format"] == BINARY_FORMAT
    assert payload["kind"] == "trajectory"
    assert {key: payload[key] for key in ("pbc", "metadata", "time_step")} == {
        "pbc": [True, True, False],
        "metadata": {"source": "md"},
        "time_step": {"value": 0.5, "unit": "fs"},
    }
    dtypes = {name: spec["dtype"] for name, spec in payload["arrays"].items()}
    assert dtypes == {
        "positions": "float32",
        "numbers": "uint8",
        "lattice": "float64",
        "steps": steps_dtype,
        "force": "float32",
        "energy": "float64",
    }
    arrays = decode_envelope(payload)
    assert {name: arr.shape for name, arr in arrays.items()} == {
        "positions": (4, 3, 3),
        "numbers": (3,),
        "lattice": (4, 3, 3),
        "steps": (4,),
        "force": (4, 3, 3),
        "energy": (4,),
    }
    # float32: relative error <= 2^-24 (half an ulp) of the float64 input
    np.testing.assert_allclose(arrays["positions"], _POSITIONS, rtol=6e-8, atol=0)
    np.testing.assert_array_equal(arrays["positions"], _POSITIONS.astype(np.float32))
    assert arrays["numbers"].tolist() == _NUMBERS
    assert arrays["steps"].tolist() == steps.tolist()
    np.testing.assert_array_equal(arrays["energy"], [-1.0, -1.5, np.nan, -2.0])
    # every buffer holds exactly prod(shape) * itemsize bytes
    for spec in payload["arrays"].values():
        n_bytes = np.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize
        assert memoryview(spec["data"]).nbytes == n_bytes


@pytest.mark.parametrize(
    ("values", "dtype", "copies"),
    [
        (np.zeros((2, 3), dtype="<f4"), "float32", False),  # zero-copy view
        (np.zeros((2, 3)), "float32", True),  # cast
        (np.zeros((3, 2), dtype="<f4").T, "float32", True),  # not C-contiguous
        (np.arange(3, dtype=">i4"), "int32", True),  # big-endian
    ],
)
def test_binary_array_copies_only_when_needed(
    values: np.ndarray, dtype: str, copies: bool
) -> None:
    """binary_array views C-contiguous little-endian input in place, else copies."""
    spec = binary_array(values, dtype)
    assert spec["shape"] == list(values.shape)
    view = np.frombuffer(spec["data"], dtype=np.dtype(dtype).newbyteorder("<"))
    assert np.shares_memory(view, values) is not copies
    np.testing.assert_array_equal(view.reshape(values.shape), values)


_NAN_POSITIONS = _POSITIONS.copy()
_NAN_POSITIONS[1, 2, 0] = np.nan


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            {"positions": _NAN_POSITIONS},
            "positions has non-finite value at flat index 15",
        ),
        ({"positions": _POSITIONS[..., :2]}, r"positions .*\[\*, \*, 3\]"),
        ({"positions": _POSITIONS * 1e38}, "positions beyond float32 range"),
        ({"vectors": {"force": _POSITIONS * 1e38}}, "force beyond float32 range"),
        *[
            ({"time_step": {"value": val, "unit": "fs"}}, "finite and > 0")
            for val in (-1, 0, np.nan, np.inf, None)
        ],
        ({"positions": _POSITIONS[:0]}, "at least 1 frame"),
        ({"positions": _POSITIONS.astype(str)}, "numeric array"),
        ({"numbers": [8, 1]}, r"numbers .*shape \[3\]"),
        ({"numbers": [8.0, 1, 1]}, r"integers in 1..118, got \[1.0, 8.0\]"),
        ({"numbers": [300, 0, 1]}, r"integers in 1..118, got \[0, 1, 300\]"),
        ({"lattice": np.eye(2)}, r"lattice .*\[3, 3\]"),
        ({"steps": np.arange(3)}, r"steps .*\[4\]"),
        ({"vectors": {"force": _POSITIONS[:, :2]}}, r"force .*\[4, 3, 3\]"),
        ({"scalars": {"energy": np.zeros(5)}}, r"energy .*\[4\]"),
        ({"scalars": {"steps": np.zeros(4)}}, r"channels \['steps'\]"),
        ({"vectors": {"energy": _POSITIONS}}, r"channels \['energy'\]"),
    ],
)
def test_trajectory_envelope_rejects_bad_inputs(
    override: dict[str, Any], match: str
) -> None:
    """Arrays matterviz's trajectory_from_binary would reject raise in Python, atomic
    numbers before the uint8 cast wraps them, and clashing channel names.
    """
    with pytest.raises(ValueError, match=match):
        _envelope(**override)


def test_envelope_kind_mismatch_raises() -> None:
    """A ready envelope of the wrong kind is rejected on assignment."""
    volume = volume_envelope(np.zeros((1, 1, 1)), id="v")
    with pytest.raises(TypeError, match="kind must be 'trajectory', got 'volume'"):
        TrajectoryWidget(trajectory=volume)
    with pytest.raises(TypeError, match="kind must be 'volume', got 'trajectory'"):
        StructureWidget(volumetric_data=[_envelope() | {"id": "v"}])


def _fallback_cases() -> list[Any]:
    """Frame lists (and pymatgen Trajectories) the binary format can't hold."""
    from ase import Atoms
    from pymatgen.core import DummySpecies, Lattice, Molecule, Species, Structure
    from pymatgen.core.trajectory import Trajectory

    lattice = Lattice.cubic(3)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    fe2 = Structure(lattice, ["Fe", "Fe"], coords)
    dummy = Structure(lattice, [DummySpecies("X"), "Si"], coords)
    forces = [{"forces": np.zeros((2, 3))}, {"forces": np.zeros((3, 3))}]
    zeros = np.zeros((2, 3))
    with_force = Structure(
        lattice, ["Fe", "Fe"], coords, site_properties={"force": zeros}
    )
    h2_pos = [[0, 0, 0], [0, 0, 0.7]]
    h2_bulk = Atoms("H2", h2_pos, cell=np.eye(3) * 5, pbc=True)
    return [
        [fe2, Structure(lattice, ["Fe", "Co"], coords)],  # varying species
        [{"structure": fe2, "label": "relaxed"}],  # string frame property
        [Structure(lattice, [{"Fe": 0.5, "Co": 0.5}], [[0, 0, 0]])],  # disordered
        [dummy, dummy],
        Trajectory.from_structures([dummy, dummy]),
        [Structure(lattice, [], [])] * 2,  # zero atoms
        [Structure(lattice, [Species("Fe", 2), Species("O", -2)], coords)],
        Trajectory.from_structures(
            [Structure(lattice, [Species("Fe", 2), Species("O", -2)], coords)] * 2
        ),
        [Structure(lattice, ["Fe", "Fe"], coords, labels=["Fe1", "Fe2"])],
        [Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.7]], charge=1, spin_multiplicity=2)],
        [Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.7]], spin_multiplicity=3)],
        Trajectory.from_structures([fe2, fe2], frame_properties=forces),  # ragged
        [Structure(lattice, ["Fe", "Fe"], coords, charge=1)],
        # singular cell (2D slab without c): the JSON path pads it with vacuum
        [Atoms("H2", h2_pos, cell=[5, 5, 0], pbc=[1, 1, 0])] * 2,
        [with_force.copy(site_properties={"forces": zeros})],  # force and forces
        [{"structure": fe2, "n_evals": 2**53 + 1}],  # float64 would round it
        [h2_bulk, Atoms("H2", h2_pos, cell=np.eye(3) * 5, pbc=[1, 1, 0])],  # pbc
        [Atoms("H2", h2_pos), h2_bulk],  # cell in only some frames
        [with_force, fe2],  # vector keys differ between frames
        # scalar/vector clash
        [{"structure": with_force.copy(site_properties={"vel": zeros}), "vel": 1.0}],
        [{"structure": fe2, "steps": 1.0}],  # reserved name
    ]


@pytest.mark.parametrize("frames", _fallback_cases())
def test_unsupported_frames_fall_back_to_json(frames: Any) -> None:
    """Frames the binary format can't hold (or would silently strip, like oxidation
    states, labels, molecule charge/spin) sync as JSON instead of raising.
    """
    traj = TrajectoryWidget(trajectory=frames).trajectory
    assert "format" not in traj
    assert "frames" in traj or traj.get("@class") == "Trajectory"
    assert trajectory_to_envelope(frames) is None


def test_binary_trajectory_describe_and_numpy_metadata() -> None:
    """describe() reads binary envelopes; numpy metadata/time_step become JSON-safe."""
    from pymatgen.core import Lattice, Structure

    struct = Structure(Lattice.cubic(3), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    facts = TrajectoryWidget(trajectory=[struct] * 3).describe()
    assert {key: facts[key] for key in ("n_frames", "n_sites", "formula")} == {
        "n_frames": 3,
        "n_sites": 2,
        "formula": "Fe2",
    }
    metadata = {"T": np.float64(300), "arr": np.arange(2)}
    traj = TrajectoryWidget(
        trajectory={"frames": [struct, struct], "metadata": metadata}
    ).trajectory
    assert traj["format"] == BINARY_FORMAT
    assert json.dumps(traj["metadata"]) == '{"T": 300.0, "arr": [0, 1]}'
    time_step = {"value": np.float32(0.5), "unit": "fs"}
    payload = trajectory_envelope(_POSITIONS, _NUMBERS, time_step=time_step)
    assert json.dumps(payload["time_step"]) == '{"value": 0.5, "unit": "fs"}'
    no_positions = {"format": BINARY_FORMAT, "kind": "trajectory", "arrays": {}}
    with pytest.raises(ValueError, match=r"lacks arrays \['numbers', 'positions'\]"):
        TrajectoryWidget(trajectory=no_positions).describe()


@pytest.mark.parametrize("displacement", [False, True])
def test_pymatgen_trajectory_per_frame_lattice(displacement: bool) -> None:
    """Displacement coords are summed back to unwrapped positions in each frame's
    cell, per-frame lattices ship as [n_frames, 3, 3], frame_properties become
    scalar and force channels.
    """
    from pymatgen.core.trajectory import Trajectory

    frac = np.array([[[0.1, 0.2, 0.95]], [[0.15, 0.2, 1.02]], [[0.2, 0.25, 1.1]]])
    lattices = np.array([np.eye(3) * (4 + 0.1 * idx) for idx in range(3)])
    forces = [[[0, 0, 3.0]], [[0, 4.0, 0]], [[0, 0, 0.0]]]
    frame_props = [
        {"energy": -1.0 - idx, "forces": force} for idx, force in enumerate(forces)
    ]
    traj = Trajectory(
        ["Si"],
        frac,
        lattice=lattices,
        constant_lattice=False,
        frame_properties=frame_props,
    )
    if displacement:
        traj.to_displacements()
    arrays = decode_envelope(TrajectoryWidget(trajectory=traj).trajectory)
    np.testing.assert_array_equal(arrays["lattice"], lattices)
    assert arrays["energy"].tolist() == [-1.0, -2.0, -3.0]
    np.testing.assert_array_equal(arrays["force"], forces)
    assert arrays["force_max"].tolist() == arrays["force_norm"].tolist() == [3, 4, 0]
    expected = np.einsum("fai,fij->faj", frac, lattices).astype(np.float32)
    # f32 cast of identical f64 products; displacement cumsum adds <= 3 f64 ulps
    np.testing.assert_allclose(arrays["positions"], expected, rtol=1e-7, atol=0)


def test_volume_envelope_round_trip() -> None:
    """A volume encodes its grid as float32 [nx, ny, nz] beside the JSON fields."""
    grid = _RNG.random((2, 3, 4))
    fields = {"id": "chg", "lattice": _CELL.tolist(), "origin": [0, 0, 0]}
    payload = volume_envelope(grid, **fields, periodic=True)
    assert payload["arrays"]["values"]["dtype"] == "float32"
    values = decode_envelope(payload)["values"]
    np.testing.assert_array_equal(values, grid.astype(np.float32))
    assert values[1, 2, 3] == np.float32(grid.ravel()[(1 * 3 + 2) * 4 + 3])
    with pytest.raises(ValueError, match=r"values must be .* \[\*, \*, \*\]"):
        volume_envelope(grid[0], **fields, periodic=True)
    np_vol = {"id": "v", "grid": grid, "lattice": _CELL, "origin": np.zeros(3)}
    np_payload = StructureWidget(volumetric_data=[np_vol | {"periodic": True}])
    np_fields = {
        key: val
        for key, val in np_payload.volumetric_data[0].items()
        if key != "arrays"
    }
    assert json.loads(json.dumps(np_fields))["origin"] == [0.0, 0.0, 0.0]
    ready = payload | {"extra": "x"}  # ready-made envelopes pass through as-is
    assert StructureWidget(volumetric_data=[ready]).volumetric_data[0]["extra"] == "x"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_headless_json_embeds_buffers_as_base64() -> None:
    """to_html/to_img embed binary buffers as base64 markers revived to DataViews."""
    payload = trajectory_envelope(_POSITIONS[:1], _NUMBERS)
    embedded = json.loads(_widget_data_json({"trajectory": payload}))["trajectory"]
    marker = embedded["arrays"]["positions"]["data"]
    assert set(marker) == {_BUFFER_KEY}
    decoded = base64.b64decode(marker[_BUFFER_KEY])
    assert decoded == bytes(payload["arrays"]["positions"]["data"])
    assert _REVIVE_BUFFERS_JS in TrajectoryWidget(trajectory=payload).to_html()
    # run the revive snippet in node: buffers become DataViews synchronously
    script = f"""const widget_data = {_widget_data_json({"trajectory": payload})};
{_REVIVE_BUFFERS_JS}
const {{ data }} = widget_data.trajectory.arrays.positions;
const floats = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
console.log(JSON.stringify([data instanceof DataView, Array.from(floats)]));"""
    result = subprocess.run(  # noqa: S603
        ["node", "-e", script],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    is_view, floats = json.loads(result.stdout)
    assert is_view
    np.testing.assert_array_equal(floats, _POSITIONS[:1].astype(np.float32).ravel())
