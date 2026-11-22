"""Tests for TrajectoryWidget rendering and notebook integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import kemio
import numpy as np
import pytest
import traitlets as tl
from kemio.interop import structure_to_pymatgen_dict
from kemio.io import Frame

from pymatviz.notebook import create_widget
from pymatviz.widgets import TrajectoryWidget
from tests.conftest import cubic_structure
from tests.widgets.conftest import decode_envelope


if TYPE_CHECKING:
    from typing import Any


def _fe2_frames(n_frames: int) -> list[kemio.Structure]:
    """Cubic Fe2 structures whose lattice grows by 0.1 A per frame."""
    return [
        cubic_structure(3.0 + idx * 0.1, ["Fe"] * 2, [[0, 0, 0], [0.5] * 3])
        for idx in range(n_frames)
    ]


# Defined here, not in tests/widgets/conftest.py: pytest 9.1 binds conftest fixtures
# to the first Dir node collected for their directory, so they go missing when CLI
# args revisit tests/widgets after another directory, e.g.
# `pytest tests/widgets/a.py tests/b.py tests/widgets/c.py`.
@pytest.fixture
def multi_frame_trajectory(fe3co4_disordered: kemio.Structure) -> dict[str, Any]:
    """Five-frame trajectory dict of the same disordered structure."""
    frames = [
        {
            "structure": structure_to_pymatgen_dict(fe3co4_disordered),
            "step": idx,
            "metadata": {"step": idx},
        }
        for idx in range(5)
    ]
    return {"frames": frames}


@pytest.mark.parametrize("input_format", ["dict", "list", "tuple", "structure"])
def test_widget_trajectory_updates(
    multi_frame_trajectory: dict[str, Any],
    fe3co4_disordered: kemio.Structure,
    input_format: str,
) -> None:
    """Widget must handle trajectory updates correctly."""
    widget = TrajectoryWidget()
    assert widget.trajectory is None
    assert widget.current_step_idx == 0

    widget.trajectory = multi_frame_trajectory
    assert widget.trajectory == json.loads(json.dumps(multi_frame_trajectory))

    widget.current_step_idx = 2
    # Test trajectory update (step doesn't reset automatically)
    updated_structure = fe3co4_disordered.copy()
    updated_structure.info = {"properties": {"energy": -1.5}}
    new_trajectory = {
        "frames": [updated_structure, updated_structure],
        "metadata": {"temperature": np.int64(300)},
    }
    widget.trajectory = {
        "dict": new_trajectory,
        "list": new_trajectory["frames"],
        "tuple": tuple(new_trajectory["frames"]),
        "structure": updated_structure,
    }[input_format]
    expected_structure = json.loads(
        json.dumps(structure_to_pymatgen_dict(updated_structure))
    )
    assert widget.trajectory == {
        "frames": [
            {
                "structure": expected_structure,
                "step": step_idx,
                "metadata": {"energy": -1.5},
            }
            for step_idx in range(1 if input_format == "structure" else 2)
        ],
        "metadata": {"temperature": 300} if input_format == "dict" else {},
    }
    json.dumps(widget.get_state()["trajectory"])
    assert new_trajectory["frames"][0] is updated_structure
    assert isinstance(new_trajectory["metadata"]["temperature"], np.int64)
    assert widget.current_step_idx == 2  # Remains unchanged
    widget.trajectory = None
    assert widget.trajectory is None
    assert widget.current_step_idx == 2
    with pytest.raises(TypeError, match="Unsupported trajectory type"):
        widget.trajectory = "invalid trajectory"
    assert widget.trajectory is None


def test_widget_complete_lifecycle(
    multi_frame_trajectory: dict[str, Any], fe3co4_disordered: kemio.Structure
) -> None:
    """Synced trait defaults, non-default construction and state restoration."""
    widget = TrajectoryWidget(trajectory=multi_frame_trajectory)
    assert widget.widget_type == "trajectory"
    synced = (
        "trajectory",
        "current_step_idx",
        "layout",
        "display_mode",
        "plot_type",
        "distribution_property",
        "show_controls",
    )
    traits = widget.class_traits()
    assert all(traits[name].metadata.get("sync") is True for name in synced)
    # show_controls=None keeps the component default (controls pinned for trajectories)
    defaults = dict(
        current_step_idx=0,
        layout="auto",
        display_mode="auto",
        plot_type="time-series",
        distribution_property=None,
        show_controls=None,
        data_inspector_open=True,
        atom_type_mapping=None,
    )
    assert {key: getattr(widget, key) for key in defaults} == defaults

    widget = TrajectoryWidget(
        trajectory=multi_frame_trajectory,
        style="width: 800px; height: 600px",
        show_controls=False,
        layout="horizontal",
        display_mode="structure",
        plot_type="distribution",
        distribution_property="energy",
    )

    assert widget.trajectory == json.loads(json.dumps(multi_frame_trajectory))
    assert widget.current_step_idx == 0
    assert widget.style == "width: 800px; height: 600px"
    assert widget.show_controls is False
    assert widget.layout == "horizontal"
    assert widget.display_mode == "structure"
    assert widget.plot_type == "distribution"
    assert widget.distribution_property == "energy"

    widget.current_step_idx = 2
    widget.set_trait("trajectory", {"frames": [fe3co4_disordered] * 10})
    expected_structure = json.loads(
        json.dumps(structure_to_pymatgen_dict(fe3co4_disordered))
    )
    frame_structs = [frame["structure"] for frame in widget.trajectory["frames"]]
    assert frame_structs == [expected_structure] * 10

    state = {
        "trajectory": widget.trajectory,
        "current_step_idx": widget.current_step_idx,
        "style": widget.style,
        "show_controls": widget.show_controls,
        "layout": widget.layout,
        "display_mode": widget.display_mode,
        "plot_type": widget.plot_type,
        "distribution_property": widget.distribution_property,
    }

    restored_widget = TrajectoryWidget(**state)

    for key, value in state.items():
        assert getattr(restored_widget, key) == value

    # atom_type_mapping (LAMMPS type -> element) syncs with JSON string keys,
    # rejects non-dicts
    widget = TrajectoryWidget(data_url="https://example.com/dump.lammpstrj")
    widget.atom_type_mapping = {1: "Si", np.int64(2): "O"}
    assert widget.to_dict()["atom_type_mapping"] == {"1": "Si", "2": "O"}
    with pytest.raises(tl.TraitError, match="'atom_type_mapping' trait"):
        TrajectoryWidget(atom_type_mapping=["Si", "O"])


@pytest.mark.parametrize(
    ("trajectory_input", "expected_properties", "is_binary"),
    [
        ([{"structure": "struct1", "energy": -1.23}], {"energy": -1.23}, True),
        # Arbitrary tensor metadata stays JSON.
        (
            [
                {
                    "structure": "struct1",
                    "other_tensor": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                    "temp": 300,
                    "step": 5,
                }
            ],
            {
                "other_tensor": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                "temp": 300,
                "step": 5,
            },
            False,
        ),
        (
            [
                {"structure": "struct1", "step": 10, "density": 3.0},
                {"structure": "struct2", "step": 20, "density": 3.1},
            ],
            {"steps": 10, "density": 3.0},  # user steps: matterviz's x axis
            True,
        ),
        # a scalar named like a reserved array ('lattice') stays JSON
        (
            [{"structure": "struct1", "lattice": 3.0}],
            {"lattice": 3.0},
            False,
        ),
    ],
)
def test_trajectory_widget_property_extraction(
    trajectory_input: list[dict[str, Any]],
    expected_properties: dict[str, Any],
    is_binary: bool,
) -> None:
    """Numeric frame properties become binary per-frame scalars, others JSON frames,
    user steps (else frame indices) the x axis either way.
    """
    structures = _fe2_frames(len(trajectory_input))
    trajectory_with_structures = [
        {**item, "structure": structures[idx]}
        for idx, item in enumerate(trajectory_input)
    ]
    traj = TrajectoryWidget(trajectory=trajectory_with_structures).trajectory
    n_frames = len(trajectory_input)
    if is_binary:
        arrays = decode_envelope(traj)
        assert arrays["positions"].shape == (n_frames, 2, 3)
        first_frame = {key: arrays[key][0] for key in expected_properties}
        assert first_frame == expected_properties
        assert set(arrays) == {"positions", "numbers", "lattice", *expected_properties}
        return
    assert set(traj) == {"frames", "metadata"}
    steps = [item.get("step", idx) for idx, item in enumerate(trajectory_input)]
    assert [frame["step"] for frame in traj["frames"]] == steps
    assert traj["frames"][0]["metadata"] == expected_properties


def test_trajectory_widget_kemio_frames_and_info() -> None:
    """Kemio io.Frames (also via create_widget) and Structure info keys (extxyz
    frame metadata) become per-frame scalars, site data vectors per-atom arrays,
    leaving the frames unchanged.
    """
    velocities = np.arange(6.0).reshape(2, 3)
    structures = [
        struct.replace(site_data={"velocities": velocities})
        for struct in _fe2_frames(2)
    ]
    for idx, struct in enumerate(structures):
        struct.info = {"temperature": 300.0 + idx, "properties": {"density": 3.0}}
    frames = [Frame(struct, energy=-1.0 - idx) for idx, struct in enumerate(structures)]
    for traj, energies in ((frames, [-1, -2]), (structures, None)):
        widget = create_widget(traj)
        assert isinstance(widget, TrajectoryWidget)
        arrays = decode_envelope(widget.trajectory)
        assert arrays["temperature"].tolist() == [300, 301]
        assert arrays["density"].tolist() == [3, 3]
        np.testing.assert_array_equal(arrays["velocities"], [velocities] * 2)
        assert (arrays["energy"].tolist() if "energy" in arrays else None) == energies
    assert structures[0].info["temperature"] == 300


@pytest.mark.parametrize("extra", [{}, {"stress": np.ones(6), "frame_stress": 42}])
def test_partial_or_conflicting_stress_stays_json(extra: dict[str, Any]) -> None:
    """Keep missing stresses and user-defined channels intact."""
    atoms = kemio.Structure(None, ["H"], [[0, 0, 0]], pbc=(False,) * 3)
    widget = TrajectoryWidget()
    widget.set_trait(
        "trajectory",
        [{"structure": atoms, "stress": [1] * 6}, {"structure": atoms, **extra}],
    )
    frames = widget.trajectory["frames"]
    assert frames[0]["metadata"]["stress"] == [1] * 6
    metadata = frames[1].get("metadata", {})
    assert metadata.keys() == extra.keys()
    for key, value in extra.items():
        np.testing.assert_array_equal(metadata[key], value)


@pytest.mark.parametrize("n_frames", [0, 1, 3])
def test_trajectory_widget_structure_sequence(n_frames: int) -> None:
    """Lists of structures sync as binary frames."""
    structures = _fe2_frames(n_frames)
    widget = TrajectoryWidget(trajectory=structures)
    if not n_frames:  # nothing to encode: empty JSON trajectory
        assert widget.trajectory == {"frames": [], "metadata": {}}
        return

    arrays = decode_envelope(widget.trajectory)
    expected_pos = np.array([struct.positions for struct in structures])
    np.testing.assert_allclose(arrays["positions"], expected_pos, rtol=0, atol=1e-6)
    assert arrays["numbers"].tolist() == [26, 26]
    expected_cells = [struct.lattice for struct in structures]
    # one shared [3, 3] cell when all frames agree, else [n_frames, 3, 3]
    expected_cells = expected_cells[0] if n_frames == 1 else expected_cells
    np.testing.assert_array_equal(arrays["lattice"], expected_cells)
    assert widget.trajectory["pbc"] == [True] * 3


_CUBIC = {"matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]]}
_ANGLES = ("alpha", "beta", "gamma")
_SI = {"element": "Si", "occu": 1.0}


@pytest.mark.parametrize(
    ("lattice_input", "species", "expected_species", "coord_key"),
    [
        (_CUBIC, ["Si"], [_SI], "abc"),  # bare symbols become {element, occu}
        (
            dict(a=4.0, b=5.0, c=6.0, alpha=80.0, beta=85.0, gamma=95.0),
            [{"element": "Si"}],
            [_SI],
            "xyz",
        ),
        (_CUBIC, [{**_SI, "occu": 0.5}, {"element": "Ge", "occu": 0.5}], None, "xyz"),
    ],
)
def test_trajectory_widget_completes_only_non_derivable_fields(
    lattice_input: dict[str, Any],
    species: list[Any],
    expected_species: list[dict[str, Any]] | None,
    coord_key: str,
) -> None:
    """Dict frames gain required fields and serialize NumPy values as JSON numbers."""
    coords = [0.25, 0.5, 0.75]
    structure = {
        "lattice": lattice_input,
        "sites": [{"species": species, coord_key: coords}],
    }
    steps = ({"step": 500}, {}, {"step": 2.5}, {"step": np.int64(3000)})
    frames = [{"structure": structure, **step} for step in steps]
    frames[0]["metadata"] = {"forces": np.zeros((1, 3)), "force_max": 0.5}
    widget = TrajectoryWidget(
        trajectory={"frames": frames, "metadata": {"temperature": np.int64(300)}}
    )
    assert [f["step"] for f in widget.trajectory["frames"]] == [500, 1, 2.5, 3000]
    exported = json.loads(json.dumps(widget.to_dict()))["trajectory"]
    assert exported["metadata"]["temperature"] == 300
    assert exported["frames"][3]["step"] == 3000
    # per-atom forces move to the sites; an explicit force_max is kept
    assert exported["frames"][0]["metadata"] == {"force_max": 0.5, "force_norm": 0.0}
    frame = widget.trajectory["frames"][0]
    lattice = frame["structure"]["lattice"]
    assert set(lattice) == {"matrix", *lattice_input}  # no a/b/c/.../volume/pbc added
    if "matrix" not in lattice_input:  # pymatgen's from_parameters: c along z
        alpha, beta, gamma = np.radians([lattice_input[key] for key in _ANGLES])
        cos_gamma_star = (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (
            np.sin(alpha) * np.sin(beta)
        )
        gamma_star = np.arccos(cos_gamma_star)
        expected_matrix = [
            [4.0 * np.sin(beta), 0.0, 4.0 * np.cos(beta)],
            [
                -5.0 * np.sin(alpha) * np.cos(gamma_star),
                5.0 * np.sin(alpha) * np.sin(gamma_star),
                5.0 * np.cos(alpha),
            ],
            [0.0, 0.0, 6.0],
        ]
        np.testing.assert_allclose(lattice["matrix"], expected_matrix, atol=1e-12)
    assert frame["structure"]["sites"] == [
        {
            "species": expected_species or species,
            coord_key: coords,
            "label": "Si1",
            "properties": {"force": [0.0, 0.0, 0.0]},
        }
    ]
    assert widget.trajectory["frames"][1]["structure"]["sites"][0]["properties"] == {}


@pytest.mark.parametrize(
    "properties",
    [
        pytest.param({}, id="no-metadata"),
        {"energy": -1.23, "forces": [[0.0, 0.3, 0.4], [0.0, 0.0, 0.0]]},
        {"temperature": 300, "pressure": 1.0},
    ],
)
def test_trajectory_widget_with_structure_metadata(properties: dict[str, Any]) -> None:
    """Structure properties become frame metadata, per-atom forces site vectors."""
    (structure,) = _fe2_frames(1)
    structure.info = {"properties": properties}

    arrays = decode_envelope(TrajectoryWidget(trajectory=structure).trajectory)
    scalars = {key: arr.tolist() for key, arr in arrays.items() if arr.ndim == 1}
    scalars.pop("numbers")
    if "forces" in properties:
        assert scalars == {
            "energy": [-1.23],
            "force_max": [pytest.approx(0.5, rel=1e-15)],
            "force_norm": [pytest.approx(0.5 / np.sqrt(2), rel=1e-15)],
        }
        # float32 vectors: within 1 float32 ulp of the float64 input
        np.testing.assert_allclose(
            arrays["force"][0], properties["forces"], rtol=1.2e-7, atol=0
        )
    else:
        assert scalars == {key: [val] for key, val in properties.items()}


def test_trajectory_widget_normalization_regressions(
    fe3co4_disordered: kemio.Structure,
) -> None:
    """Numpy forces become per-site numeric lists (matterviz 0.8 ignores vectors in
    frame metadata) plus force_max/force_norm, malformed forces raise, the layout trait
    survives to_dict(), and inputs are not mutated (regressions).
    """
    forces = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    frames = [{"structure": fe3co4_disordered, "forces": forces, "energy": -1.5}]
    widget = TrajectoryWidget(trajectory=frames, layout="horizontal")

    (frame,) = widget.trajectory["frames"]
    assert frame["metadata"] == {
        "energy": -1.5,
        "force_max": 2.0,
        "force_norm": pytest.approx(np.sqrt(2.5), rel=1e-15),
    }
    site_forces = [site["properties"]["force"] for site in frame["structure"]["sites"]]
    assert site_forces == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    for bad_frame, match in (
        ({"force": [0.1, 0.2, 0.3]}, r"frame 0 'force' must be per-atom.*\(3,\)"),
        ({"forces": forces, "force": forces}, "both 'force' and 'forces'"),
    ):
        with pytest.raises(ValueError, match=match):
            TrajectoryWidget(trajectory=[{"structure": fe3co4_disordered, **bad_frame}])
    # a frames dict takes structure objects too, and frames lacking one raise
    obj_frames = {"frames": [{"structure": fe3co4_disordered, "energy": -1.5}]}
    (obj_frame,) = TrajectoryWidget(trajectory=obj_frames).trajectory["frames"]
    assert len(obj_frame["structure"]["sites"]) == len(fe3co4_disordered)
    with pytest.raises(
        ValueError, match=r"frame 0 must be .* 'structure' key, got \['sites'"
    ):
        TrajectoryWidget(trajectory={"frames": [{"sites": [], "lattice": {}}]})
    json.dumps(widget.trajectory)  # round-trips as standard JSON
    # layout used to be filtered from to_dict() by _EXCLUDED_TRAITS, so headless
    # to_img/to_html exports silently ignored it
    assert widget.to_dict().get("layout") == "horizontal"

    # normalization must not mutate the caller's structure dicts (a shallow
    # lattice copy used to leak setdefault/assignments into the input)
    struct_dict = structure_to_pymatgen_dict(fe3co4_disordered)
    struct_dict["lattice"].pop("pbc", None)
    lattice_before = {**struct_dict["lattice"]}
    TrajectoryWidget(trajectory={"frames": [{"structure": struct_dict}]})
    assert struct_dict["lattice"] == lattice_before
    sites_before = json.dumps(struct_dict["sites"])
    TrajectoryWidget(
        trajectory={
            "frames": [{"structure": struct_dict, "metadata": {"forces": forces}}]
        }
    )
    assert json.dumps(struct_dict["sites"]) == sites_before
