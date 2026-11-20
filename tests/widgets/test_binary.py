"""Tests for the matterviz-binary-v1 trajectory, volume and plot-series encoders."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest
import traitlets as tl

from pymatviz import (
    BinnedScatterPlotWidget,
    BoxPlotWidget,
    ScatterPlotWidget,
    StructureWidget,
    TrajectoryWidget,
)
from pymatviz.widgets._binary import (
    BINARY_FORMAT,
    BINARY_SERIES_MIN_POINTS,
    binary_array,
    volume_envelope,
)
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


_N_BIG = BINARY_SERIES_MIN_POINTS


@pytest.mark.parametrize("widget_cls", [ScatterPlotWidget, BinnedScatterPlotWidget])
def test_large_plot_series_sync_as_binary(widget_cls: type) -> None:
    """Large numeric series sync as plot-series envelopes, others as JSON."""
    x_vals = np.arange(_N_BIG, dtype=np.int32)
    colors = np.full(_N_BIG, np.nan)
    big = {"x": x_vals, "y": x_vals * 0.5, "color_values": colors, "label": "big"}
    small = {"x": [1, 2], "y": [3.0, 4.0]}
    categorical = {"x": ["a"] * _N_BIG, "y": np.ones(_N_BIG)}
    widget = widget_cls(series=[big, small, categorical])
    envelope, small_out, cat_out = widget.series
    assert envelope["format"] == BINARY_FORMAT
    assert envelope["kind"] == "plot-series"
    assert envelope["label"] == "big"
    arrays = decode_envelope(envelope)
    assert set(arrays) == {"x", "y", "color_values"}
    assert all(arr.dtype == np.float64 for arr in arrays.values())
    np.testing.assert_array_equal(arrays["y"], x_vals * 0.5)
    assert np.isnan(arrays["color_values"]).all()  # gaps allowed off x/y
    assert small_out == small
    assert cat_out["x"][0] == "a"
    widget.series = [envelope]  # ready envelopes pass through on reassignment
    assert widget.series[0] is envelope
    widget.series = [small]
    assert widget.series == [small]


def test_large_box_samples_sync_as_binary() -> None:
    """Box/raincloud samples (y only) of large series sync as binary arrays."""
    samples = np.random.default_rng(seed=0).normal(size=_N_BIG)
    (envelope, small) = BoxPlotWidget(
        series=[{"y": samples, "label": "a"}, {"y": [1, 2]}]
    ).series
    assert envelope["kind"] == "plot-series"
    assert envelope["label"] == "a"
    np.testing.assert_array_equal(decode_envelope(envelope)["y"], samples)
    assert small == {"y": [1, 2]}


def test_plot_series_metadata_frame() -> None:
    """A per-point metadata DataFrame becomes metadata.<col> arrays on binary series
    and per-point dicts (non-finite as None) on JSON series.
    """
    import pandas as pd

    xs = np.arange(_N_BIG, dtype=float)
    meta = pd.DataFrame({"threshold": np.r_[np.inf, xs[1:]], "f1": np.nan})
    (envelope,) = ScatterPlotWidget(
        series=[{"x": xs, "y": xs, "metadata": meta}]
    ).series
    assert "metadata" not in envelope
    arrays = decode_envelope(envelope)
    np.testing.assert_array_equal(arrays["metadata.threshold"], meta["threshold"])
    assert np.isnan(arrays["metadata.f1"]).all()

    small = {"x": [1, 2], "y": [3, 4]}
    frame = pd.DataFrame({"threshold": [np.inf, 0.5], "name": ["a", None]})
    (json_series,) = ScatterPlotWidget(series=[small | {"metadata": frame}]).series
    assert json_series["metadata"] == [
        {"threshold": None, "name": "a"},
        {"threshold": 0.5, "name": None},
    ]
    # non-numeric columns keep a large series' metadata as JSON records
    labels = pd.DataFrame({"name": ["a"] * _N_BIG})
    (mixed,) = ScatterPlotWidget(series=[{"x": xs, "y": xs, "metadata": labels}]).series
    assert mixed["metadata"][0] == {"name": "a"}
    assert "x" in mixed["arrays"]
    dupes = pd.DataFrame([[1.0, 2.0]] * _N_BIG, columns=pd.Index(["a", "a"]))
    for points, frame, match in [
        (small, pd.DataFrame({"a": [1, 2, 3]}), r"one row per point \(2\), got 3"),
        ({"x": xs, "y": xs}, pd.DataFrame({"a": [1.0]}), r"\(10000\), got 1"),
        ({"x": xs, "y": xs}, dupes, r"uniquely named columns, got \['a', 'a'\]"),
        (small, pd.DataFrame(index=range(2)), r"uniquely named columns, got \[\]"),
    ]:
        with pytest.raises(tl.TraitError, match=match):
            ScatterPlotWidget(series=[points | {"metadata": frame}])


@pytest.mark.parametrize(
    ("series", "match"),
    [
        (
            {"x": np.arange(_N_BIG), "y": np.arange(_N_BIG + 1)},
            r"share one length, got \{'x': 10000, 'y': 10001\}",
        ),
        (
            {"x": np.arange(_N_BIG), "y": np.r_[np.nan, np.ones(_N_BIG - 1)]},
            r"series\[0\]\.y has non-finite value",
        ),
    ],
)
def test_invalid_large_plot_series(series: dict[str, Any], match: str) -> None:
    """Large series with ragged columns or non-finite x/y raise on assignment."""
    with pytest.raises(tl.TraitError, match=match):
        ScatterPlotWidget(series=[series])


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
        site_properties={"moment": [np.array(1), np.array(2)]},
        frame_properties=[{}, {"energy": -1, "forces": np.full((2, 3), [0, 3, 4])}],
    )
    traj.to_displacements()
    traj.charge = 2  # Explicit charge differs from the sum of oxidation states (3).
    serialized = traj.as_dict()
    value = {
        "object": traj,
        "dict": serialized,
        "json": json.loads(
            json.dumps(
                serialized
                | {
                    "coords": traj.coords,
                    "lattice": _CELL,
                    "site_properties": [traj.site_properties],
                },
                cls=MontyEncoder,
            )
        ),
    }[source]
    with pytest.warns(UserWarning, match="sum of oxidation states"):
        widget = TrajectoryWidget(trajectory=value)
    assert widget.describe() == {
        "widget_type": "trajectory",
        "n_frames": 2,
        "n_sites": 2,
        "elements": ["Co", "Fe"],
        "formula": "Co0.5Fe1.5",
    }
    payload = widget.trajectory
    assert traj.coords_are_displacement
    assert payload["time_step"] == {"value": 2, "unit": "fs"}
    assert payload["frames"][1]["metadata"] == {
        "energy": -1,
        "force_max": 5,
        "force_norm": 5,
    }
    assert payload["frames"][1]["structure"]["sites"][0]["properties"]["force"] == [
        0,
        3,
        4,
    ]
    for frame, pos in zip(payload["frames"], frac[:, 0], strict=True):
        assert frame["structure"]["charge"] == 2
        site, iron = frame["structure"]["sites"]
        assert iron["species"][0]["oxidation_state"] == 3
        assert site["species"] == [
            {"element": "Fe", "occu": 0.5},
            {"element": "Co", "occu": 0.5},
        ]
        assert site["properties"]["moment"] == 1
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


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf, True, "2"])
def test_invalid_pymatgen_time_step(step: Any) -> None:
    """Live and serialized trajectories reject invalid time steps before encoding."""
    from pymatgen.core.trajectory import Trajectory

    traj = Trajectory(["Si"], np.zeros((1, 1, 3)), lattice=_CELL, time_step=step)
    for value in (traj, traj.as_dict()):
        with pytest.raises(ValueError, match="time_step value must be finite and > 0"):
            TrajectoryWidget(trajectory=value)


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


def _run_node(script: str) -> str:
    """Run a script in node via stdin (a large series' base64 overflows argv)."""
    cmd = ["node", "-"]
    return subprocess.run(  # noqa: S603
        cmd, input=script, capture_output=True, text=True, check=True
    ).stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_headless_json_embeds_buffers_as_base64() -> None:
    """Headless exports revive only binary array data as DataViews."""
    from ase import Atoms

    payload = TrajectoryWidget(
        trajectory=Atoms(numbers=_NUMBERS, positions=_POSITIONS[0])
    ).trajectory
    marker = {"__matterviz_buffer_b64__": "ordinary metadata"}
    volume = volume_envelope(_POSITIONS, id="grid")
    xs = np.arange(_N_BIG)
    (series,) = ScatterPlotWidget(series=[{"x": xs, "y": xs}]).series
    data_json = _widget_data_json(
        {
            "trajectory": payload,
            "volumetric_data": [volume],
            "series": [series, {"x": [1], "y": [2]}],
            "metadata": marker,
        }
    )
    encoded = json.loads(data_json)["trajectory"]["arrays"]["positions"]["data"]
    assert base64.b64decode(encoded) == bytes(payload["arrays"]["positions"]["data"])
    assert _REVIVE_BUFFERS_JS in TrajectoryWidget(trajectory=payload).to_html()
    script = f"""const widget_data = {data_json};
{_REVIVE_BUFFERS_JS}
const {{ data }} = widget_data.trajectory.arrays.positions;
console.log(JSON.stringify([data instanceof DataView,
  widget_data.volumetric_data[0].arrays.values.data instanceof DataView,
  widget_data.series[0].arrays.y.data instanceof DataView,
  widget_data.metadata, Array.from(new Float32Array(data.buffer))]));"""
    traj_view, vol_view, series_view, metadata, floats = json.loads(_run_node(script))
    assert traj_view
    assert vol_view
    assert series_view
    assert metadata == marker
    np.testing.assert_array_equal(floats, _POSITIONS[:1].astype(np.float32).ravel())
    # unset traits sync as null and must not break the revive loop
    nulls = {"trajectory": None, "volumetric_data": None, "series": None}
    null_script = f"const widget_data = {_widget_data_json(nulls)};\n"
    null_script += f"{_REVIVE_BUFFERS_JS}console.log('ok')"
    assert _run_node(null_script).strip() == "ok"
