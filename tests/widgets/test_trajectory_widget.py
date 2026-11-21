"""Tests for TrajectoryWidget rendering and notebook integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import traitlets as tl

from pymatviz import TrajectoryWidget
from tests.widgets.conftest import decode_envelope


if TYPE_CHECKING:
    from typing import Any

    from pymatgen.core import Structure


# Defined here, not in tests/widgets/conftest.py: pytest 9.1 binds conftest fixtures
# to the first Dir node collected for their directory, so they go missing when CLI
# args revisit tests/widgets after another directory, e.g.
# `pytest tests/widgets/a.py tests/b.py tests/widgets/c.py`.
@pytest.fixture
def multi_frame_trajectory(fe3co4_disordered: Structure) -> dict[str, Any]:
    """Five-frame trajectory dict of the same disordered structure."""
    frames = [
        {
            "structure": fe3co4_disordered.as_dict(),
            "step": idx,
            "metadata": {"step": idx},
        }
        for idx in range(5)
    ]
    return {"frames": frames}


def test_widget_creates_view_model(multi_frame_trajectory: dict[str, Any]) -> None:
    """Widget must create proper view model for frontend."""
    widget = TrajectoryWidget(trajectory=multi_frame_trajectory)
    assert widget.widget_type == "trajectory"

    # Synced traits expose the widget state contract used by the frontend.
    class_traits = widget.class_traits()
    for trait_name in (
        "trajectory",
        "current_step_idx",
        "layout",
        "display_mode",
        "plot_type",
        "distribution_property",
        "show_controls",
    ):
        trait = class_traits[trait_name]
        assert trait.metadata.get("sync") is True

    assert widget.trajectory == json.loads(json.dumps(multi_frame_trajectory))
    assert widget.current_step_idx == 0
    assert widget.layout == "auto"
    assert widget.display_mode == "auto"
    assert widget.plot_type == "time-series"
    assert widget.distribution_property is None
    # None keeps the component default (controls pinned for trajectories)
    assert widget.show_controls is None
    assert widget.data_inspector_open is True


@pytest.mark.parametrize("input_format", ["dict", "list", "tuple", "structure"])
def test_widget_trajectory_updates(
    multi_frame_trajectory: dict[str, Any],
    fe3co4_disordered: Structure,
    input_format: str,
) -> None:
    """Widget must handle trajectory updates correctly."""
    widget = TrajectoryWidget()
    assert widget.trajectory is None
    assert widget.current_step_idx == 0

    widget.trajectory = multi_frame_trajectory
    assert widget.trajectory == json.loads(json.dumps(multi_frame_trajectory))

    widget.current_step_idx = 2
    assert widget.current_step_idx == 2

    # Test trajectory update (step doesn't reset automatically)
    updated_structure = fe3co4_disordered.copy(properties={"energy": -1.5})
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
    expected_structure = json.loads(json.dumps(updated_structure.as_dict()))
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
    multi_frame_trajectory: dict[str, Any], fe3co4_disordered: Structure
) -> None:
    """Test complete widget lifecycle including state persistence."""
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
    assert widget.current_step_idx == 2

    new_trajectory = {"frames": [fe3co4_disordered] * 10}
    widget.trajectory = new_trajectory
    expected_structure = json.loads(json.dumps(fe3co4_disordered.as_dict()))
    assert [frame["structure"] for frame in widget.trajectory["frames"]] == [
        expected_structure
    ] * 10

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
                }
            ],
            {"other_tensor": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], "temp": 300},
            False,
        ),
        (
            [
                {"structure": "struct1", "step": 0, "density": 3.0},
                {"structure": "struct2", "step": 1, "density": 3.1},
            ],
            {"step": 0, "density": 3.0},
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
    """Numeric frame properties become binary per-frame scalars, others JSON frames."""
    from pymatgen.core import Lattice, Structure

    structures = [
        Structure(
            lattice=Lattice.cubic(3.0 + idx * 0.1),
            species=("Fe", "Fe"),
            coords=((0, 0, 0), (0.5, 0.5, 0.5)),
        )
        for idx in range(len(trajectory_input))
    ]
    trajectory_with_structures = [
        {**item, "structure": structures[idx]}
        for idx, item in enumerate(trajectory_input)
    ]
    traj = TrajectoryWidget(trajectory=trajectory_with_structures).trajectory
    n_frames = len(trajectory_input)
    if is_binary:
        arrays = decode_envelope(traj)
        assert arrays["positions"].shape == (n_frames, 2, 3)
        assert {key: arrays[key][0] for key in expected_properties} == (
            expected_properties
        )
        assert set(arrays) == {"positions", "numbers", "lattice", *expected_properties}
        return
    assert set(traj) == {"frames", "metadata"}
    assert [frame["step"] for frame in traj["frames"]] == list(range(n_frames))
    assert traj["frames"][0]["metadata"] == expected_properties


@pytest.mark.parametrize("shape", [(6,), (3, 3)])
def test_ase_stress_transfer(shape: tuple[int, ...]) -> None:
    """Transfer raw ASE stress losslessly as binary frame_stress."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    frames = [Atoms("H", positions=[[idx, 0, 0]]) for idx in range(2)]
    tensors = np.random.default_rng(seed=0).normal(size=(len(frames), *shape))
    for atoms, stress in zip(frames, tensors, strict=True):
        atoms.calc = SinglePointCalculator(atoms, stress=stress)

    widget = TrajectoryWidget()
    widget.trajectory = frames
    encoded = widget.trajectory
    wire = decode_envelope(encoded)["frame_stress"]
    assert wire.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(wire, tensors.reshape(2, -1))


@pytest.mark.parametrize(
    "extra", [{}, {"stress": np.eye(3)}, {"stress": np.ones(6), "frame_stress": 42}]
)
def test_partial_or_conflicting_stress_stays_json(extra: dict[str, Any]) -> None:
    """Keep missing stresses, mixed shapes, and user-defined channels intact."""
    from ase import Atoms

    atoms = Atoms("H")
    widget = TrajectoryWidget()
    widget.trajectory = [
        {"structure": atoms, "stress": [1] * 6},
        {"structure": atoms, **extra},
    ]
    frames = widget.trajectory["frames"]
    assert frames[0]["metadata"]["stress"] == [1] * 6
    metadata = frames[1].get("metadata", {})
    assert metadata.keys() == extra.keys()
    for key, value in extra.items():
        np.testing.assert_array_equal(metadata[key], value)


@pytest.mark.parametrize(
    "n_frames",
    [0, 1, 3],
)
def test_trajectory_widget_structure_sequence(n_frames: int) -> None:
    """Test TrajectoryWidget handles lists of structures and pymatgen Trajectory."""
    from pymatgen.core import Lattice, Structure

    structures = [
        Structure(
            lattice=Lattice.cubic(3.0 + idx * 0.1),
            species=("Fe", "Fe"),
            coords=((0, 0, 0), (0.5, 0.5, 0.5)),
        )
        for idx in range(n_frames)
    ]

    widget = TrajectoryWidget(trajectory=structures)
    if not n_frames:  # nothing to encode: empty JSON trajectory
        assert widget.trajectory == {"frames": [], "metadata": {}}
        return

    arrays = decode_envelope(widget.trajectory)
    expected_pos = np.array([struct.cart_coords for struct in structures])
    np.testing.assert_allclose(arrays["positions"], expected_pos, rtol=0, atol=1e-6)
    assert arrays["numbers"].tolist() == [26, 26]
    expected_cells = [struct.lattice.matrix for struct in structures]
    # one shared [3, 3] cell when all frames agree, else [n_frames, 3, 3]
    expected_cells = expected_cells[0] if n_frames == 1 else expected_cells
    np.testing.assert_array_equal(arrays["lattice"], expected_cells)
    assert widget.trajectory["pbc"] == [True] * 3

    from pymatgen.core.trajectory import Trajectory

    traj = Trajectory.from_structures(structures, constant_lattice=False)
    traj_arrays = decode_envelope(TrajectoryWidget(trajectory=traj).trajectory)
    np.testing.assert_array_equal(
        traj_arrays["positions"], expected_pos.astype("float32")
    )
    np.testing.assert_array_equal(traj_arrays["lattice"], expected_cells)
    # The frontend also accepts molecular JSON without the Python constructor's charge.
    serialized = {
        "@class": "Trajectory",
        "coords": [[[0, 0, 0]]],
        "species": ["H"],
        "lattice": None,
    }
    assert TrajectoryWidget(trajectory=serialized).trajectory == serialized


_CUBIC = {"matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]]}
_SI = {"element": "Si", "occu": 1.0}


@pytest.mark.parametrize(
    ("lattice_input", "species", "expected_species", "coord_key"),
    [
        (_CUBIC, ["Si"], [_SI], "abc"),  # bare symbols become {element, occu}
        (
            dict(a=4.0, b=5.0, c=6.0, alpha=90.0, beta=90.0, gamma=90.0),
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
    if "matrix" not in lattice_input:
        np.testing.assert_allclose(
            lattice["matrix"], np.diag([4.0, 5.0, 6.0]), atol=1e-12
        )
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
    ("metadata_field", "metadata_value"),
    [
        pytest.param(None, {}, id="no-metadata"),
        ("properties", {"energy": -1.23, "forces": [[0.0, 0.3, 0.4], [0.0, 0.0, 0.0]]}),
        ("info", {"temperature": 300, "pressure": 1.0}),
    ],
)
def test_trajectory_widget_with_structure_metadata(
    metadata_field: str | None, metadata_value: dict[str, Any]
) -> None:
    """TrajectoryWidget forwards structure properties/info to frame metadata, per-atom
    forces to the sites.
    """
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        lattice=Lattice.cubic(3.0),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    if metadata_field == "properties":
        structure.properties = metadata_value
    elif metadata_field == "info":
        object.__setattr__(structure, "info", metadata_value)

    arrays = decode_envelope(TrajectoryWidget(trajectory=structure).trajectory)
    scalars = {key: arr.tolist() for key, arr in arrays.items() if arr.ndim == 1}
    scalars.pop("numbers")
    if "forces" in metadata_value:
        assert scalars == {
            "energy": [-1.23],
            "force_max": [pytest.approx(0.5, rel=1e-15)],
            "force_norm": [pytest.approx(0.5 / np.sqrt(2), rel=1e-15)],
        }
        # float32 vectors: within 1 float32 ulp of the float64 input
        np.testing.assert_allclose(
            arrays["force"][0], metadata_value["forces"], rtol=1.2e-7, atol=0
        )
    else:
        assert scalars == {key: [val] for key, val in metadata_value.items()}


@pytest.mark.parametrize(
    ("symbols", "positions", "cell", "energy"),
    [
        pytest.param(
            "H2O", [[0, 0, 0], [0, 0, 1], [0, 1, 0]], None, -1.5, id="molecule"
        ),
        pytest.param(
            "Fe2", [[0, 0, 0], [0.5, 0.5, 0.5]], [3, 3, 3], -2.0, id="periodic"
        ),
    ],
)
@pytest.mark.parametrize("mson_atoms", [False, True], ids=["Atoms", "MSONAtoms"])
def test_trajectory_widget_with_ase_atoms(
    symbols: str,
    positions: list[list[float]],
    cell: list[int] | None,
    energy: float,
    mson_atoms: bool,
) -> None:
    """ASE frames (incl. pymatgen MSONAtoms, whose as_dict() is not a structure dict)
    keep info and calculator results (which AseAtomsAdaptor drops) with and without a
    simulation cell; calculator forces become site vectors.
    """
    ase_atoms = pytest.importorskip("ase").Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from pymatgen.io.ase import MSONAtoms

    atoms_cls = MSONAtoms if mson_atoms else ase_atoms

    atoms = atoms_cls(symbols, positions=positions, cell=cell)
    atoms.info = {"temperature": 300}
    forces = np.zeros((len(atoms), 3))
    forces[0] = [0.0, 0.0, 2.0]
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    payload = TrajectoryWidget(trajectory=atoms).trajectory
    arrays = decode_envelope(payload)
    np.testing.assert_allclose(arrays["positions"][0], positions, rtol=0, atol=1e-7)
    assert arrays["numbers"].tolist() == atoms.numbers.tolist()
    # a molecule has no cell (and so no pbc); a periodic frame keeps its cell
    assert ("lattice" in arrays) == (cell is not None)
    assert payload.get("pbc") == (atoms.pbc.tolist() if cell else None)
    assert {key: arrays[key].tolist() for key in ("temperature", "energy")} == {
        "temperature": [300],
        "energy": [energy],
    }
    assert arrays["force_max"].tolist() == [2.0]
    np.testing.assert_allclose(arrays["force_norm"], [2 / np.sqrt(len(atoms))])
    assert arrays["force"][0].tolist() == forces.tolist()


def test_trajectory_widget_atom_type_mapping_trait() -> None:
    """atom_type_mapping (LAMMPS type -> element) is None by default, syncs as given
    and rejects non-dicts.
    """
    assert TrajectoryWidget().atom_type_mapping is None
    mapping = {1: "Si", "2": "O"}
    widget = TrajectoryWidget(data_url="https://example.com/dump.lammpstrj")
    widget.atom_type_mapping = mapping
    assert widget.to_dict()["atom_type_mapping"] == mapping
    with pytest.raises(tl.TraitError):
        TrajectoryWidget(atom_type_mapping=["Si", "O"])


def test_trajectory_widget_normalization_regressions(
    fe3co4_disordered: Structure,
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
    json.dumps(widget.trajectory)  # round-trips as standard JSON
    # layout used to be filtered from to_dict() by _EXCLUDED_TRAITS, so headless
    # to_img/to_html exports silently ignored it
    assert widget.to_dict().get("layout") == "horizontal"

    # normalization must not mutate the caller's structure dicts (a shallow
    # lattice copy used to leak setdefault/assignments into the input)
    struct_dict = fe3co4_disordered.as_dict()
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
