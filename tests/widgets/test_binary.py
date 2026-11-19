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
from pymatviz.widgets._binary import BINARY_FORMAT, binary_array, volume_envelope
from pymatviz.widgets._headless import _REVIVE_BUFFERS_JS, _widget_data_json
from tests.widgets.conftest import decode_envelope


_POSITIONS = np.random.default_rng(seed=0).random((4, 3, 3)) * 10
_NUMBERS = [8, 1, 1]
_CELL = np.eye(3) * 10


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


@pytest.mark.parametrize("key", ["positions", "force", "lattice"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 1e39])
def test_invalid_frame_arrays(key: str, value: float) -> None:
    """Non-finite data and float32 overflow fail before a frame syncs."""
    from ase import Atoms

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    if key == "positions":
        atoms.positions[0, 0] = value
    elif key == "force":
        atoms.info["force"] = [[value, 0, 0]]
    else:
        atoms.cell[0, 0] = value
        if np.isfinite(value):  # Lattice uses float64.
            arrays = decode_envelope(TrajectoryWidget(trajectory=atoms).trajectory)
            assert arrays["lattice"][0, 0] == value
            return
    with pytest.raises(
        ValueError, match=f"{key} (has non-finite value|beyond float32 range)"
    ):
        TrajectoryWidget(trajectory=atoms)


def test_envelope_kind_mismatch_raises() -> None:
    """A ready envelope of the wrong kind is rejected on assignment."""
    volume = volume_envelope(np.zeros((1, 1, 1)), id="v")
    with pytest.raises(TypeError, match="kind must be 'trajectory', got 'volume'"):
        TrajectoryWidget(trajectory=volume)
    with pytest.raises(TypeError, match="kind must be 'volume', got 'trajectory'"):
        StructureWidget(volumetric_data=[volume | {"kind": "trajectory"}])


@pytest.mark.parametrize(
    ("widget_type", "kwargs"),
    [
        (
            TrajectoryWidget,
            {"trajectory": {"format": BINARY_FORMAT, "kind": "trajectory"}},
        ),
        (StructureWidget, {"volumetric_data": [{"id": "mask", "grid": [[[True]]]}]}),
    ],
)
def test_pinned_bundle_needs_decoder(
    widget_type: type, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incompatible pinned bundle raises before any bytes sync."""
    from pymatviz.widgets import matterviz

    monkeypatch.setattr(matterviz, "fetch_widget_asset", lambda *_args: "old bundle")
    with pytest.raises(ValueError, match=r"0\.8\.0.*lacks matterviz-binary-v1"):
        widget_type(version_override="0.8.0", **kwargs)


def _fallback_cases() -> list[Any]:
    """Frame lists (and pymatgen Trajectories) the binary format can't hold."""
    from ase import Atoms
    from ase.constraints import FixAtoms
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
        [fe2.copy(site_properties={"aux": [[0, 1, 2], [3, 4]]})],
        [with_force.copy(site_properties={"force_max": zeros})],
        [Atoms("H2", h2_pos, cell=[5, 5, 5], constraint=FixAtoms(indices=[0]))],
        [Atoms("H2", h2_pos, cell=[5, 5, 5], magmoms=[[0, 0, 1], [0, 0, 2]])],
        [
            {
                "structure": Structure(
                    lattice, ["Fe", "Fe"], coords, properties={"phase": "alpha"}
                ),
                "energy": -1,
            }
        ],
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
        [Atoms("H2", h2_pos, cell=[0.009] * 3, pbc=True)],
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


@pytest.mark.parametrize("source", ["object", "dict", "json"])
def test_disordered_trajectory_frames(source: str) -> None:
    """Mixed sites retain occupancies, unwrapped coordinates and properties."""
    from monty.json import MontyEncoder
    from pymatgen.core import Composition, Species
    from pymatgen.core.trajectory import Trajectory

    frac = np.array([[[0.9, 0, 0], [0, 0, 0]], [[1.1, 0, 0], [0, 0, 0]]])
    traj = Trajectory(
        [Composition({"Fe": 0.5, "Co": 0.5}), Species("Fe", 3)],
        frac,
        lattice=_CELL,
        time_step=2,
        site_properties={"moment": [1, 2]},
        frame_properties=[{}, {"energy": -1}],
    )
    traj.to_displacements()
    value = {
        "object": traj,
        "dict": traj.as_dict(),
        "json": json.loads(json.dumps(traj, cls=MontyEncoder)),
    }[source]
    payload = TrajectoryWidget(trajectory=value).trajectory
    assert traj.coords_are_displacement
    assert payload["time_step"] == {"value": 2, "unit": "fs"}
    assert payload["frames"][1]["metadata"] == {"energy": -1}
    for frame, pos in zip(payload["frames"], frac[:, 0], strict=True):
        site, iron = frame["structure"]["sites"]
        assert iron["species"][0]["oxidation_state"] == 3
        assert site["species"] == [
            {"element": "Fe", "occu": 0.5},
            {"element": "Co", "occu": 0.5},
        ]
        assert site["properties"] == {"moment": 1}
        np.testing.assert_array_equal(site["abc"], pos)
        np.testing.assert_array_equal(site["xyz"], pos @ _CELL)


def test_binary_trajectory_describe_and_numpy_metadata() -> None:
    """describe() reads binary envelopes; numpy metadata/time_step become JSON-safe."""
    from pymatgen.core import IStructure, Lattice

    struct = IStructure(Lattice.cubic(3), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
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
    assert traj["pbc"] == [True] * 3
    np.testing.assert_array_equal(
        decode_envelope(traj)["lattice"], struct.lattice.matrix
    )
    assert json.dumps(traj["metadata"]) == '{"T": 300.0, "arr": [0, 1]}'
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
    forces = np.array([[[0, 0, 300]], [[0, 4, 0]], [[0, 0, 0]]], dtype=np.float16)
    frame_props = [
        {"energy": -1.0 - idx, "forces": force} for idx, force in enumerate(forces)
    ]
    traj = Trajectory(
        ["Si"],
        frac,
        lattice=lattices,
        constant_lattice=False,
        frame_properties=frame_props,
        time_step=np.float64(0.5),
    )
    if displacement:
        traj.to_displacements()
    payload = TrajectoryWidget(trajectory=traj).trajectory
    assert json.loads(json.dumps(payload["time_step"])) == {"value": 0.5, "unit": "fs"}
    arrays = decode_envelope(payload)
    np.testing.assert_array_equal(arrays["lattice"], lattices)
    assert arrays["energy"].tolist() == [-1.0, -2.0, -3.0]
    np.testing.assert_array_equal(arrays["force"], forces)
    assert arrays["force_max"].tolist() == arrays["force_norm"].tolist() == [300, 4, 0]
    expected = np.einsum("fai,fij->faj", frac, lattices).astype(np.float32)
    np.testing.assert_array_equal(arrays["positions"], expected)
    traj.frame_properties = [None, {"energy": -2.0}, {}]
    np.testing.assert_array_equal(
        decode_envelope(TrajectoryWidget(trajectory=traj).trajectory)["energy"],
        [np.nan, -2.0, np.nan],
    )
    traj.time_step, traj.frame_properties = 0, [{"label": "MD"}] * len(frac)
    with pytest.raises(ValueError, match="time_step value"):
        TrajectoryWidget(trajectory=traj)


def test_volume_envelope_round_trip() -> None:
    """A volume encodes its grid as float32 [nx, ny, nz] beside the JSON fields."""
    grid = np.arange(24, dtype="<f4").reshape(2, 3, 4)
    volume = {"id": "v", "grid": grid, "lattice": _CELL, "origin": np.zeros(3)}
    widget = StructureWidget(volumetric_data=[volume])
    (payload,) = widget.volumetric_data
    values = decode_envelope(payload)["values"]
    assert np.shares_memory(grid, values)
    np.testing.assert_array_equal(values, grid)
    assert json.loads(json.dumps(payload["origin"])) == [0, 0, 0]
    with pytest.raises(ValueError, match=r"values must be .* \[\*, \*, \*\]"):
        widget.volumetric_data = [volume | {"grid": grid[0]}]
    widget.volumetric_data = [payload | {"extra": "x"}]
    assert widget.volumetric_data[0]["extra"] == "x"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_headless_json_embeds_buffers_as_base64() -> None:
    """Headless exports revive only binary array data as DataViews."""
    from ase import Atoms

    payload = TrajectoryWidget(
        trajectory=Atoms(numbers=_NUMBERS, positions=_POSITIONS[0])
    ).trajectory
    marker = {"__matterviz_buffer_b64__": "ordinary metadata"}
    volume = volume_envelope(_POSITIONS, id="grid")
    data_json = _widget_data_json(
        {"trajectory": payload, "volumetric_data": [volume], "metadata": marker}
    )
    encoded = json.loads(data_json)["trajectory"]["arrays"]["positions"]["data"]
    assert base64.b64decode(encoded) == bytes(payload["arrays"]["positions"]["data"])
    assert _REVIVE_BUFFERS_JS in TrajectoryWidget(trajectory=payload).to_html()
    script = f"""const widget_data = {data_json};
{_REVIVE_BUFFERS_JS}
const {{ data }} = widget_data.trajectory.arrays.positions;
console.log(JSON.stringify([data instanceof DataView,
  widget_data.volumetric_data[0].arrays.values.data instanceof DataView,
  widget_data.metadata, Array.from(new Float32Array(data.buffer))]));"""
    result = subprocess.run(  # noqa: S603
        ["node", "-e", script],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    traj_view, vol_view, metadata, floats = json.loads(result.stdout)
    assert traj_view
    assert vol_view
    assert metadata == marker
    np.testing.assert_array_equal(floats, _POSITIONS[:1].astype(np.float32).ravel())
