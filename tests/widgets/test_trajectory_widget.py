"""Tests for TrajectoryWidget rendering and notebook integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pymatviz import TrajectoryWidget
from tests.widgets.conftest import (
    assert_widget_build_files,
    assert_widget_edge_cases,
    assert_widget_notebook_integration,
)


if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure


def test_widget_build_files_and_display_trajectory() -> None:
    """Widget must load build files and display properly."""
    widget = TrajectoryWidget()
    assert_widget_build_files(widget)


def test_widget_notebook_integration() -> None:
    """Widget must integrate properly with notebook environments."""
    widget = TrajectoryWidget()
    assert_widget_notebook_integration(widget)


@pytest.mark.parametrize(
    ("file_suffix", "file_content", "create_func"),
    [
        (
            ".xyz",
            "2\nStep 0\nH 0.0 0.0 0.0\nO 0.0 0.0 1.0",
            lambda file: file.write("2\nStep 0\nH 0.0 0.0 0.0\nO 0.0 0.0 1.0"),
        ),
        (
            ".npz",
            None,
            lambda file: np.savez(
                file.name, trajectory=np.random.default_rng(seed=0).random((3, 5, 3))
            ),
        ),
        (
            ".pkl",
            None,
            lambda file: __import__("pickle").dump(
                [{"structure": "test", "energy": 1.0}], file
            ),
        ),
        (".txt", None, lambda file: file.write(b"not a trajectory file")),
    ],
)
def test_trajectory_file_loading_various_formats(
    file_suffix: str, file_content: str | None, create_func: Any, tmp_path: Path
) -> None:
    """Test loading various trajectory file formats via data_url."""
    temp_path = tmp_path / f"trajectory{file_suffix}"

    if file_content:
        temp_path.write_text(file_content)
    else:
        with open(temp_path, "wb") as temp_file:
            create_func(temp_file)

    widget = TrajectoryWidget(data_url=str(temp_path))
    assert widget.data_url == str(temp_path)
    assert widget.trajectory is None  # Frontend handles loading


@pytest.mark.parametrize(
    ("file_suffix", "compression_func"),
    [
        (".xyz.gz", lambda f, content: __import__("gzip").open(f, "wt").write(content)),
        (
            ".zip",
            lambda f, content: __import__("zipfile")
            .ZipFile(f, "w")
            .writestr("trajectory.xyz", content),
        ),
    ],
)
def test_trajectory_file_loading_compressed_formats(
    file_suffix: str, compression_func: Any, tmp_path: Path
) -> None:
    """Test loading compressed trajectory files via data_url."""
    xyz_content = "2\nStep 0\nH 0.0 0.0 0.0\nO 0.0 0.0 1.0"

    temp_path = tmp_path / f"trajectory{file_suffix}"
    compression_func(str(temp_path), xyz_content)

    widget = TrajectoryWidget(data_url=str(temp_path))
    assert widget.data_url == str(temp_path)
    assert widget.trajectory is None


@pytest.mark.parametrize(
    "data_url",
    [
        "nonexistent_file.xyz",
        "/absolute/path/to/file.xyz",
        "./relative/path/to/file.xyz",
    ],
)
def test_trajectory_file_path_handling(data_url: str) -> None:
    """Test that various file paths are correctly handled via data_url."""
    widget = TrajectoryWidget(data_url=data_url)
    assert widget.data_url == data_url
    assert widget.trajectory is None


def test_trajectory_file_loading_with_metadata(tmp_path: Path) -> None:
    """Test loading trajectory files that include metadata via data_url."""
    from pymatgen.core import Lattice, Structure

    simple_structure = Structure(
        lattice=Lattice.cubic(3.0),
        species=["Fe", "Fe"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    trajectory_data = [{"structure": simple_structure, "energy": 1.0, "step": 0}]

    temp_path = tmp_path / "trajectory.pkl"
    import pickle

    with open(temp_path, "wb") as temp_file:
        pickle.dump(trajectory_data, temp_file)

    widget = TrajectoryWidget(data_url=str(temp_path))
    assert widget.data_url == str(temp_path)
    assert widget.trajectory is None


def test_widget_creates_view_model(multi_frame_trajectory: dict[str, Any]) -> None:
    """Widget must create proper view model for frontend."""
    widget = TrajectoryWidget(trajectory=multi_frame_trajectory)

    # Test that widget has proper attributes for anywidget
    assert hasattr(widget, "trajectory"), "Widget missing trajectory attribute"
    assert hasattr(widget, "current_step_idx"), (
        "Widget missing current_step_idx attribute"
    )
    assert hasattr(widget, "layout"), "Widget missing layout attribute"
    assert hasattr(widget, "display_mode"), "Widget missing display_mode attribute"
    assert hasattr(widget, "show_controls"), "Widget missing show_controls attribute"

    # Test that attributes are synchronized
    assert widget.trajectory == multi_frame_trajectory
    assert widget.current_step_idx == 0
    assert widget.layout == "auto"
    assert widget.display_mode == "structure+scatter"
    assert widget.show_controls is True

    # Test that trajectory can be serialized
    json.dumps(widget.trajectory)


@pytest.mark.parametrize("step_idx", [0, 2, 4, 10, -1])
def test_widget_step_navigation(
    multi_frame_trajectory: dict[str, Any], step_idx: int
) -> None:
    """Widget must handle step navigation correctly."""
    widget = TrajectoryWidget(trajectory=multi_frame_trajectory)
    widget.current_step_idx = step_idx
    assert widget.current_step_idx == step_idx

    # Test that step is tagged for sync
    trait = widget.class_traits()["current_step_idx"]
    assert trait.metadata.get("sync") is True, "Step property not synced"


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
        assert getattr(restored_widget, key) == value


def test_widget_performance_and_large_trajectories(
    fe3co4_disordered: Structure,
) -> None:
    """Test widget performance with large trajectories."""
    # Test large trajectory handling
    long_trajectory = {"frames": [fe3co4_disordered] * 100}

    widget = TrajectoryWidget(trajectory=long_trajectory)
    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == 100

    # Test step navigation with large trajectory
    widget.current_step_idx = 50
    assert widget.current_step_idx == 50

    widget.current_step_idx = 99
    assert widget.current_step_idx == 99


def test_widget_edge_cases_and_error_handling_trajectory(
    multi_frame_trajectory: dict[str, Any],
) -> None:
    """Test widget edge cases and error handling."""
    # Test widget with no trajectory
    widget = TrajectoryWidget()
    assert widget.trajectory is None
    assert widget.current_step_idx == 0

    # Test widget handles missing/corrupted build files gracefully
    widget = TrajectoryWidget(trajectory=multi_frame_trajectory)
    assert_widget_edge_cases(widget)

    # Test trajectory serialization
    json.dumps(widget.trajectory)  # Should not raise exception


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
    trajectory_input: list[dict[str, Any]], expected_frames: int
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
        ([], {"frames": [], "metadata": {}}),  # type: ignore[dict-item]
        (
            {"frames": [{"structure": "test", "step": 0}]},
            {"frames": [{"structure": "test", "step": 0}]},
        ),
    ],
)
def test_trajectory_widget_edge_cases(
    trajectory_input: Any,
    expected_result: dict[str, list[dict[str, Any]]] | None,
) -> None:
    """Test TrajectoryWidget handles edge cases correctly."""
    result = TrajectoryWidget(trajectory=trajectory_input).trajectory
    if expected_result is not None:
        assert result == expected_result  # Test exact match for non-None results
    else:
        assert result is None


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


def test_trajectory_widget_with_structure_properties() -> None:
    """Test TrajectoryWidget handles structures with properties and info."""
    from pymatgen.core import Lattice, Structure

    # Create structure with properties only
    structure = Structure(
        lattice=Lattice.cubic(3.0),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    structure.properties = {"energy": -1.23, "forces": [[0.1, 0.2, 0.3]]}

    widget = TrajectoryWidget(trajectory=structure)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == 1
    frame = widget.trajectory["frames"][0]
    assert "metadata" in frame
    assert frame["metadata"]["energy"] == -1.23
    assert frame["metadata"]["forces"] == [[0.1, 0.2, 0.3]]


def test_trajectory_widget_with_structure_info() -> None:
    """Test TrajectoryWidget handles structures with info attribute."""
    from pymatgen.core import Lattice, Structure

    # Create structure with info only (no properties)
    structure = Structure(
        lattice=Lattice.cubic(3.0),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    structure.info = {"temperature": 300, "pressure": 1.0}

    widget = TrajectoryWidget(trajectory=structure)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == 1
    frame = widget.trajectory["frames"][0]
    assert "metadata" in frame
    assert frame["metadata"]["temperature"] == 300
    assert frame["metadata"]["pressure"] == 1.0


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
    atoms.properties = {"energy": -2.0}

    widget = TrajectoryWidget(trajectory=atoms)

    assert widget.trajectory is not None
    assert len(widget.trajectory["frames"]) == 1
    frame = widget.trajectory["frames"][0]
    assert frame["step"] == 0
    assert "metadata" in frame
    assert frame["metadata"]["energy"] == -2.0
