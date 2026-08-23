"""Tests for TrajectoryWidget rendering and notebook integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import traitlets as tl

from pymatviz import TrajectoryWidget
from tests.widgets.conftest import (
    assert_widget_build_files,
    assert_widget_notebook_integration,
)


if TYPE_CHECKING:
    from typing import Any

    from pymatgen.core import Structure


def test_widget_build_files_and_display_trajectory() -> None:
    """Widget must load build files and display properly."""
    widget = TrajectoryWidget()
    assert_widget_build_files(widget)


def test_widget_notebook_integration() -> None:
    """Widget must integrate properly with notebook environments."""
    widget = TrajectoryWidget()
    assert_widget_notebook_integration(widget)


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
        "show_controls",
    ):
        trait = class_traits[trait_name]
        assert trait.metadata.get("sync") is True

    assert widget.trajectory == multi_frame_trajectory
    assert widget.current_step_idx == 0
    assert widget.layout == "auto"
    assert widget.display_mode == "structure+scatter"
    assert widget.show_controls is True

    # Test that trajectory can be serialized
    json.dumps(widget.trajectory)


def test_widget_trajectory_updates(
    multi_frame_trajectory: dict[str, Any], fe3co4_disordered: Structure
) -> None:
    """Widget must handle trajectory updates correctly."""
    widget = TrajectoryWidget()
    assert widget.trajectory is None
    assert widget.current_step_idx == 0

    # Test trajectory assignment
    widget.trajectory = multi_frame_trajectory
    assert widget.trajectory == multi_frame_trajectory

    # Test step navigation
    widget.current_step_idx = 2
    assert widget.current_step_idx == 2

    # Test trajectory update (step doesn't reset automatically)
    new_trajectory = {"frames": [fe3co4_disordered, fe3co4_disordered]}
    widget.trajectory = new_trajectory
    assert widget.trajectory == new_trajectory
    assert widget.current_step_idx == 2  # Remains unchanged


def test_widget_complete_lifecycle(
    multi_frame_trajectory: dict[str, Any], fe3co4_disordered: Structure
) -> None:
    """Test complete widget lifecycle including state persistence."""
    # Create widget with custom settings
    widget = TrajectoryWidget(
        trajectory=multi_frame_trajectory,
        style="width: 800px; height: 600px",
        show_controls=False,
        layout="horizontal",
        display_mode="structure",
    )

    # Test initial state
    assert widget.trajectory == multi_frame_trajectory
    assert widget.current_step_idx == 0
    assert widget.style == "width: 800px; height: 600px"
    assert widget.show_controls is False
    assert widget.layout == "horizontal"
    assert widget.display_mode == "structure"

    # Test step navigation
    widget.current_step_idx = 2
    assert widget.current_step_idx == 2

    # Test trajectory update
    new_trajectory = {"frames": [fe3co4_disordered] * 10}
    widget.trajectory = new_trajectory
    assert widget.trajectory == new_trajectory

    # Test state persistence
    state = {
        "trajectory": widget.trajectory,
        "current_step_idx": widget.current_step_idx,
        "style": widget.style,
        "show_controls": widget.show_controls,
        "layout": widget.layout,
        "display_mode": widget.display_mode,
    }

    # Create new widget from state
    restored_widget = TrajectoryWidget(**state)

    # Verify state preservation
    for key, value in state.items():
        if key != "trajectory":
            assert getattr(restored_widget, key) == value

    restored_trajectory = restored_widget.trajectory
    assert restored_trajectory is not None
    assert len(restored_trajectory["frames"]) == len(state["trajectory"]["frames"])


@pytest.mark.parametrize(
    ("trajectory_input", "expected_frames", "expected_properties"),
    [
        # Basic properties
        (
            [{"structure": "struct1", "energy": -1.23, "force": [0.1, 0.2, 0.3]}],
            1,
            {"energy": -1.23, "force": [0.1, 0.2, 0.3]},
        ),
        # Complex properties
        (
            [
                {
                    "structure": "struct1",
                    "stress": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                    "temp": 300,
                }
            ],
            1,
            {"stress": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], "temp": 300},
        ),
        # Multiple frames
        (
            [
                {"structure": "struct1", "step": 0, "lattice": 3.0},
                {"structure": "struct2", "step": 1, "lattice": 3.1},
            ],
            2,
            {"step": 0, "lattice": 3.0},
        ),
    ],
)
def test_trajectory_widget_property_extraction(
    trajectory_input: list[dict[str, Any]],
    expected_frames: int,
    expected_properties: dict[str, Any],
) -> None:
    """Test TrajectoryWidget correctly extracts properties from dict format."""
    from pymatgen.core import Lattice, Structure

    # Create structures and replace placeholders
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

    widget = TrajectoryWidget(trajectory=trajectory_with_structures)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == expected_frames

    frame1 = widget.trajectory["frames"][0]
    assert frame1["step"] == 0
    assert all(frame1["metadata"][k] == v for k, v in expected_properties.items())

    # Test step indices are sequential
    for idx, frame in enumerate(widget.trajectory["frames"]):
        assert frame["step"] == idx

    # Test no extra fields in trajectory dict
    assert set(widget.trajectory) == {"frames", "metadata"}


@pytest.mark.parametrize(
    ("trajectory_input", "expected_frames"),
    [([], 0), (["struct1"], 1), (["struct1", "struct2", "struct3"], 3)],
)
def test_trajectory_widget_backward_compatibility(
    trajectory_input: list[str], expected_frames: int
) -> None:
    """Test TrajectoryWidget handles list of structures."""
    from pymatgen.core import Lattice, Structure

    structures = [
        Structure(
            lattice=Lattice.cubic(3.0 + idx * 0.1),
            species=("Fe", "Fe"),
            coords=((0, 0, 0), (0.5, 0.5, 0.5)),
        )
        for idx in range(len(trajectory_input))
    ]

    widget = TrajectoryWidget(trajectory=structures)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == expected_frames
    assert all(
        frame["step"] == idx for idx, frame in enumerate(widget.trajectory["frames"])
    )

    # Test no extra fields in trajectory dict
    if widget.trajectory is not None:
        actual_keys = set(widget.trajectory)
        assert actual_keys == {"frames", "metadata"}


@pytest.mark.parametrize(
    ("trajectory_input", "expected_result"),
    [
        (None, None),
        ([], {"frames": [], "metadata": {}}),
    ],
)
def test_trajectory_widget_edge_cases(
    trajectory_input: Any,
    expected_result: dict[str, list[dict[str, Any]] | dict[str, Any]] | None,
) -> None:
    """Test TrajectoryWidget handles edge cases correctly."""
    result = TrajectoryWidget(trajectory=trajectory_input).trajectory
    if expected_result is not None:
        assert result == expected_result  # Test exact match for non-None results
    else:
        assert result is None


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
    """Dict frames get step, occu, label, properties and a lattice matrix; fields
    matterviz's trajectory_from_json recomputes (a/b/c/angles/volume/pbc, the
    missing abc<->xyz) are not emitted, keeping the JSON payload lean. Explicit steps
    (int, float, numpy) and occupancies are kept, missing steps default to the index.
    """
    coords = [0.25, 0.5, 0.75]
    structure = {
        "lattice": lattice_input,
        "sites": [{"species": species, coord_key: coords}],
    }
    steps = ({"step": 500}, {}, {"step": 2.5}, {"step": np.int64(3000)})
    frames = [{"structure": structure, **step} for step in steps]
    widget = TrajectoryWidget(trajectory={"frames": frames})
    assert [f["step"] for f in widget.trajectory["frames"]] == [500, 1, 2.5, 3000]
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
            "properties": {},
        }
    ]


def test_trajectory_widget_single_structure_extra_fields() -> None:
    """Test TrajectoryWidget handles single structures without extra fields."""
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        lattice=Lattice.cubic(3.0),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )

    widget = TrajectoryWidget(trajectory=structure)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == 1
    assert widget.trajectory["frames"][0]["step"] == 0

    # Test no extra fields in single structure trajectory
    assert set(widget.trajectory) == {"frames", "metadata"}


def test_trajectory_string_input_raises_error() -> None:
    """Test that passing string to trajectory parameter raises error."""
    with pytest.raises(TypeError, match="Unsupported trajectory type"):
        TrajectoryWidget(trajectory="just a string")


@pytest.mark.parametrize(
    ("metadata_field", "metadata_value"),
    [
        ("properties", {"energy": -1.23, "forces": [[0.1, 0.2, 0.3]]}),
        ("info", {"temperature": 300, "pressure": 1.0}),
    ],
)
def test_trajectory_widget_with_structure_metadata(
    metadata_field: str, metadata_value: dict[str, Any]
) -> None:
    """TrajectoryWidget forwards structure properties/info to frame metadata."""
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        lattice=Lattice.cubic(3.0),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    if metadata_field == "properties":
        structure.properties = metadata_value
    else:
        object.__setattr__(structure, "info", metadata_value)

    widget = TrajectoryWidget(trajectory=structure)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == 1
    frame = widget.trajectory["frames"][0]
    assert "metadata" in frame
    assert frame["metadata"] == metadata_value


def test_trajectory_widget_with_ase_atoms() -> None:
    """Test TrajectoryWidget handles ASE Atoms objects."""
    pytest.importorskip("ase")
    from ase import Atoms

    # Create ASE Atoms with no cell (molecular system)
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.info = {"energy": -1.5}

    widget = TrajectoryWidget(trajectory=atoms)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == 1
    frame = widget.trajectory["frames"][0]
    assert frame["step"] == 0
    assert "metadata" in frame
    assert frame["metadata"]["energy"] == -1.5


def test_trajectory_widget_with_ase_atoms_with_cell() -> None:
    """Test TrajectoryWidget handles ASE Atoms with cell."""
    pytest.importorskip("ase")
    from ase import Atoms

    # Create ASE Atoms with cell
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[3, 3, 3])
    atoms.info = {"energy": -2.0}

    widget = TrajectoryWidget(trajectory=atoms)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == 1
    frame = widget.trajectory["frames"][0]
    assert frame["step"] == 0
    assert "metadata" in frame
    assert frame["metadata"]["energy"] == -2.0


def test_trajectory_widget_vector_configs_trait() -> None:
    """TrajectoryWidget vector_configs trait round-trips correctly."""
    configs = {
        "force_DFT": {"visible": True, "color": "#e74c3c", "scale": None},
        "force_MLFF": {"visible": False, "color": "#3498db", "scale": 2.0},
    }
    widget = TrajectoryWidget(vector_configs=configs, vector_scale=0.5)
    assert widget.vector_configs == configs
    assert widget.vector_scale == 0.5
    assert TrajectoryWidget().vector_configs is None


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
    """Numpy metadata serializes as numeric lists (not stringified reprs), the
    layout trait survives to_dict(), and inputs are not mutated (regressions).
    """
    forces = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    frames = [{"structure": fe3co4_disordered, "forces": forces, "energy": -1.5}]
    widget = TrajectoryWidget(trajectory=frames, layout="horizontal")

    metadata = widget.trajectory["frames"][0]["metadata"]
    assert metadata["forces"] == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    assert metadata["energy"] == -1.5
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
