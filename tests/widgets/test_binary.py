"""Tests for the matterviz-binary-v1 trajectory, volume and plot-series encoders."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from typing import Any

import kemio
import numpy as np
import pytest
import traitlets as tl

from pymatviz.widgets import (
    BinnedScatterPlotWidget,
    BoxPlotWidget,
    ScatterPlot3DWidget,
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
from tests.conftest import cubic_structure
from tests.widgets.conftest import decode_envelope


_FE2 = cubic_structure(3, ["Fe"] * 2, [[0, 0, 0], [0.5] * 3])


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


@pytest.mark.parametrize(
    ("key", "value"),
    [("force", np.nan), ("force", np.inf), ("force", 1e39), ("positions", 1e39)],
)
def test_invalid_frame_arrays(key: str, value: float) -> None:
    """Non-finite data and float32 overflow fail before a frame syncs (kemio
    rejects non-finite positions and lattices itself).
    """
    positions = [[value if key == "positions" else 0, 0, 0]]
    properties = {"force": [[value, 0, 0]]} if key == "force" else {}
    struct = kemio.Structure(
        np.eye(3) * 2, ["H"], positions, info={"properties": properties}
    )
    with pytest.raises(
        ValueError, match=f"{key} (has non-finite value|beyond float32 range)"
    ):
        TrajectoryWidget(trajectory=struct)


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


def test_large_scatter_plot_3d_series_sync_as_binary() -> None:
    """Large 3D series sync as binary envelopes with a finite z column."""
    xs = np.arange(_N_BIG, dtype=float)
    widget = ScatterPlot3DWidget(
        series=[{"x": xs, "y": -xs, "z": 2 * xs, "label": "big"}]
    )
    (envelope,) = widget.series
    assert envelope["kind"] == "plot-series"
    assert envelope["label"] == "big"
    arrays = decode_envelope(envelope)
    assert set(arrays) == {"x", "y", "z"}
    np.testing.assert_array_equal(arrays["z"], 2 * xs)
    assert widget.describe()["z_range"] == [0, 2 * (_N_BIG - 1)]
    widget.series = [envelope]  # ready envelopes pass through
    assert widget.series[0] is envelope
    with pytest.raises(tl.TraitError, match=r"series\[0\]\.z has non-finite"):
        widget.series = [{"x": xs, "y": xs, "z": np.r_[np.nan, xs[1:]]}]
    assert widget.series[0] is envelope


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
    frame = pd.DataFrame(
        {
            "threshold": [np.inf, 0.5],
            "name": ["a", None],
            "measurement": pd.Series([-np.inf, "unavailable"], dtype=object),
        }
    )
    (json_series,) = ScatterPlotWidget(series=[small | {"metadata": frame}]).series
    assert json_series["metadata"] == [
        {"threshold": None, "name": "a", "measurement": None},
        {"threshold": 0.5, "name": None, "measurement": "unavailable"},
    ]
    # a non-numeric column keeps all of a large series' metadata as JSON records
    labels = pd.DataFrame({"name": ["a"] * _N_BIG, "score": xs})
    (mixed,) = ScatterPlotWidget(series=[{"x": xs, "y": xs, "metadata": labels}]).series
    first_records = [{"name": "a", "score": 0}, {"name": "a", "score": 1}]
    assert mixed["metadata"][:2] == first_records
    assert set(mixed["arrays"]) == {"x", "y"}
    assert "x" in mixed["arrays"]
    dupes = pd.DataFrame([[1.0, 2.0]] * _N_BIG, columns=pd.Index(["a", "a"]))
    for points, frame, match in [
        (small, pd.DataFrame({"a": [1, 2, 3]}), r"one row per point \(2\), got 3"),
        ({"x": xs, "y": xs}, pd.DataFrame({"a": [1.0]}), r"\(10000\), got 1"),
        ({"x": xs, "y": xs}, dupes, r"uniquely named columns, got \['a', 'a'\]"),
        # 1 and "1" would both sync as metadata.1
        ({"x": xs, "y": xs}, pd.DataFrame({1: xs, "1": xs}), r"got \[1, '1'\]"),
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
    """Frame lists the binary format can't hold."""
    coords = [[0, 0, 0], [0.5] * 3]
    fe2_phase = kemio.Structure(
        _FE2.lattice, ["Fe"] * 2, _FE2.positions, info={"properties": {"phase": "a"}}
    )
    h2_pos = [[0, 0, 0], [0, 0, 0.7]]
    h2_bulk = kemio.Structure(np.eye(3) * 5, ["H", "H"], h2_pos)
    h2_mol = kemio.Structure(None, ["H", "H"], h2_pos, pbc=(False,) * 3)
    return [
        [_FE2, cubic_structure(3, ["Fe", "Co"], coords)],  # varying species
        [{"structure": _FE2, "label": "relaxed"}],  # string frame property
        [cubic_structure(3, [{"Fe": 0.5, "Co": 0.5}], coords[0])],  # disordered
        [{"structure": fe2_phase, "energy": -1}],
        [cubic_structure(3, [], np.zeros((0, 3)))] * 2,  # zero atoms
        # oxidation states, site labels and properties, molecule charge, tiny cell
        [cubic_structure(3, ["Fe2+", "O2-"], coords)],
        [_FE2.replace(site_data={"labels": ["Fe1", "Fe2"]})],
        [_FE2.replace(site_data={"magmoms": np.array([1.0, -1.0])})],
        [kemio.Structure(None, ["H", "H"], h2_pos, pbc=(False,) * 3, charge=1)],
        [kemio.Structure(np.eye(3) * 0.009, ["H", "H"], h2_pos)],  # 7e-7 A^3 cell
        [{"structure": _FE2, "n_evals": 2**53 + 1}],  # float64 would round it
        [h2_bulk, h2_bulk.replace(pbc=(True, True, False))],  # pbc
        [h2_mol, h2_bulk],  # cell in only some frames
        [{"structure": _FE2, "force": np.zeros((2, 3))}, _FE2],  # vector keys differ
        [{"structure": _FE2, "steps": 1.0}],  # reserved name
    ]


@pytest.mark.parametrize("frames", _fallback_cases())
def test_unsupported_frames_fall_back_to_json(frames: Any) -> None:
    """Frames the binary format can't hold (or would silently strip, like oxidation
    states, labels, molecule charge/spin) sync as JSON instead of raising.
    """
    traj = TrajectoryWidget(trajectory=frames).trajectory
    assert "format" not in traj
    assert "frames" in traj or traj.get("@class") == "Trajectory"


def test_binary_trajectory_wire_dtypes_and_info_channels() -> None:
    """Atomic numbers sync as uint8, positions as float32, scalars as float64; a
    frame's time_step info (the widget's time axis unit) is no per-frame series.
    """
    struct = _FE2.replace(info={"time_step": 2.0, "temp": 3.0})
    traj = TrajectoryWidget(trajectory=[struct, struct]).trajectory
    dtypes = {name: spec["dtype"] for name, spec in traj["arrays"].items()}
    assert dtypes == {
        "positions": "float32",
        "numbers": "uint8",
        "lattice": "float64",
        "temp": "float64",
    }


def test_binary_force_stats_keep_given_values() -> None:
    """Per-frame force_max/force_norm given in metadata win; frames without them
    get the max and RMS of their per-atom force norms.
    """
    forces = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])  # norms 5 and 0
    frames = [
        {"structure": _FE2, "forces": forces, "force_max": 9.0, "force_norm": 8.0},
        {"structure": _FE2, "forces": 2 * forces},
    ]
    arrays = decode_envelope(TrajectoryWidget(trajectory=frames).trajectory)
    assert arrays["force_max"].tolist() == [9.0, 10.0]
    np.testing.assert_allclose(
        arrays["force_norm"], [8.0, 10 / np.sqrt(2)], rtol=1e-15, atol=0
    )
    np.testing.assert_array_equal(arrays["force"], [forces, 2 * forces])


def test_binary_trajectory_describe_and_numpy_metadata() -> None:
    """describe() reads binary envelopes; numpy metadata/time_step become JSON-safe."""
    facts = TrajectoryWidget(trajectory=[_FE2] * 3).describe()
    assert {key: facts[key] for key in ("n_frames", "n_sites", "formula")} == {
        "n_frames": 3,
        "n_sites": 2,
        "formula": "Fe2",
    }
    metadata = {"T": np.float64(300), "arr": np.arange(2)}
    traj = TrajectoryWidget(
        trajectory={"frames": [_FE2, _FE2], "metadata": metadata}
    ).trajectory
    assert traj["pbc"] == [True] * 3
    np.testing.assert_array_equal(decode_envelope(traj)["lattice"], _FE2.lattice)
    assert json.dumps(traj["metadata"]) == '{"T": 300.0, "arr": [0, 1]}'
    no_positions = {"format": BINARY_FORMAT, "kind": "trajectory", "arrays": {}}
    with pytest.raises(ValueError, match=r"lacks arrays \['numbers', 'positions'\]"):
        TrajectoryWidget(trajectory=no_positions).describe()


def test_volume_envelope_round_trip() -> None:
    """A volume encodes its grid as float32 [nx, ny, nz] beside the JSON fields."""
    grid = np.arange(24, dtype="<f4").reshape(2, 3, 4)
    volume = {"id": "v", "grid": grid, "lattice": np.eye(3), "origin": np.zeros(3)}
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
    positions = np.random.default_rng(seed=0).random((4, 3, 3)) * 10
    molecule = kemio.Structure(None, [8, 1, 1], positions[0], pbc=(False,) * 3)
    payload = TrajectoryWidget(trajectory=molecule).trajectory
    marker = {"__matterviz_buffer_b64__": "ordinary metadata"}
    volume = volume_envelope(positions, id="grid")
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
    assert [traj_view, vol_view, series_view, metadata] == [True, True, True, marker]
    np.testing.assert_array_equal(floats, positions[0].astype(np.float32).ravel())
    # unset traits sync as null and must not break the revive loop
    nulls = {"trajectory": None, "volumetric_data": None, "series": None}
    null_script = f"const widget_data = {_widget_data_json(nulls)};\n"
    null_script += f"{_REVIVE_BUFFERS_JS}console.log('ok')"
    assert _run_node(null_script).strip() == "ok"
