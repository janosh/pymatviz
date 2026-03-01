"""Tests for TrajectoryWidget rendering and notebook integration."""

from __future__ import annotations

import gzip
import json
import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymatviz import TrajectoryWidget
from tests.widgets.conftest import (
    assert_widget_build_files,
    assert_widget_notebook_integration,
)


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from pymatgen.core import Structure


def _write_gzip(filepath: str, content: str) -> None:
    """Write content to a gzip file."""
    with gzip.open(filepath, "wt") as gz_file:
        gz_file.write(content)


def _write_zip(filepath: str, content: str) -> None:
    """Write content to a zip file containing trajectory.xyz."""
    with zipfile.ZipFile(filepath, "w") as zip_file:
        zip_file.writestr("trajectory.xyz", content)


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
    [(".xyz.gz", _write_gzip), (".zip", _write_zip)],
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
        if key != "trajectory":
            assert getattr(restored_widget, key) == value

    restored_trajectory = restored_widget.trajectory
    assert restored_trajectory is not None
    assert len(restored_trajectory["frames"]) == len(state["trajectory"]["frames"])


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

    # Build-asset sanity check on a regular instance.
    widget = TrajectoryWidget(trajectory=multi_frame_trajectory)
    assert_widget_build_files(widget)

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


def _build_invalid_trajectory_input(
    case_name: str, valid_structure: dict[str, Any]
) -> dict[str, Any]:
    """Create compact invalid trajectory dict variants for schema validation tests."""

    def _single_frame(structure_data: Any) -> dict[str, Any]:
        """Wrap structure payload in a single-frame trajectory dict."""
        return {"frames": [{"structure": structure_data, "step": 0}]}

    def _site_with_species(
        element_symbol: str, *, include_abc: bool, empty_species: bool = False
    ) -> dict[str, Any]:
        """Build a minimal site payload with optional fractional coordinates."""
        site_data: dict[str, Any] = {
            "species": []
            if empty_species
            else [{"element": element_symbol, "occu": 1.0}]
        }
        if include_abc:
            site_data["abc"] = [0, 0, 0]
        return site_data

    invalid_cases: dict[str, dict[str, Any]] = {
        "missing_frames": {"metadata": {}},
        "empty_frames": {"frames": []},
        "non_dict_frame": {"frames": [123]},
        "missing_structure": {"frames": [{"step": 0}]},
        "bad_structure_type": {"frames": [{"structure": "bad", "step": 0}]},
        "missing_lattice_params": _single_frame(
            {
                "lattice": {"a": 1.0, "b": 1.0, "c": 1.0},
                "sites": [_site_with_species("Si", include_abc=True)],
            }
        ),
        "empty_sites": _single_frame({**valid_structure, "sites": []}),
        "missing_site_coords": _single_frame(
            {**valid_structure, "sites": [_site_with_species("Si", include_abc=False)]}
        ),
        "empty_species": _single_frame(
            {
                **valid_structure,
                "sites": [
                    _site_with_species("Si", include_abc=True, empty_species=True)
                ],
            }
        ),
        "missing_coords_second_site": _single_frame(
            {
                **valid_structure,
                "sites": [
                    _site_with_species("Si", include_abc=True),
                    _site_with_species("O", include_abc=False),
                ],
            }
        ),
    }
    return invalid_cases[case_name]


@pytest.mark.parametrize(
    ("case_name", "error_cls", "match"),
    [
        ("missing_frames", ValueError, "missing required key 'frames'"),
        ("empty_frames", ValueError, "frames' is empty"),
        ("non_dict_frame", TypeError, "frame must be a dict"),
        ("missing_structure", ValueError, "missing required key 'structure'"),
        ("bad_structure_type", TypeError, "'structure' must be a dict"),
        (
            "missing_lattice_params",
            ValueError,
            "must provide either 'matrix' or all of",
        ),
        ("empty_sites", ValueError, "empty 'sites'"),
        ("empty_species", ValueError, "species' must be a non-empty list"),
        ("missing_site_coords", ValueError, "coordinate key 'abc' or 'xyz'"),
        ("missing_coords_second_site", ValueError, "coordinate key 'abc' or 'xyz'"),
    ],
)
def test_trajectory_widget_invalid_dict_schema_raises_helpful_error(
    case_name: str,
    error_cls: type[Exception],
    match: str,
    fe3co4_disordered: Structure,
) -> None:
    """Invalid trajectory dicts should fail with actionable schema errors."""
    valid_structure = fe3co4_disordered.as_dict()
    trajectory_input = _build_invalid_trajectory_input(case_name, valid_structure)
    with pytest.raises(error_cls, match=match):
        TrajectoryWidget(trajectory=trajectory_input)


@pytest.mark.parametrize(
    ("structure_input", "expected_site_coord_key", "expected_lattice_key"),
    [
        (
            {
                "lattice": {"matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]]},
                "sites": [{"species": [{"element": "Si"}], "abc": [0.25, 0.5, 0.75]}],
            },
            "xyz",
            "volume",
        ),
        (
            {
                "lattice": {
                    "a": 4.0,
                    "b": 5.0,
                    "c": 6.0,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                },
                "sites": [{"species": [{"element": "Si"}], "xyz": [1.0, 2.5, 3.0]}],
            },
            "abc",
            "matrix",
        ),
    ],
)
def test_trajectory_widget_derives_missing_fields(
    structure_input: dict[str, Any],
    expected_site_coord_key: str,
    expected_lattice_key: str,
) -> None:
    """Trajectory dict inputs derive missing lattice/site fields."""
    widget = TrajectoryWidget(
        trajectory={"frames": [{"structure": structure_input, "step": 0}]}
    )
    structure = widget.trajectory["frames"][0]["structure"]
    site = structure["sites"][0]
    assert expected_lattice_key in structure["lattice"]
    assert expected_site_coord_key in site
    assert "label" in site
    assert "properties" in site


def test_trajectory_widget_accepts_string_species_entries() -> None:
    """String species entries should normalize without assuming mapping access."""
    widget = TrajectoryWidget(
        trajectory={
            "frames": [
                {
                    "structure": {
                        "lattice": {"matrix": [[4, 0, 0], [0, 4, 0], [0, 0, 4]]},
                        "sites": [{"species": ["Si"], "abc": [0.0, 0.0, 0.0]}],
                    },
                    "step": 0,
                }
            ]
        }
    )
    structure = widget.trajectory["frames"][0]["structure"]
    site = structure["sites"][0]
    assert site["species"] == ["Si"]
    assert site["label"] == "Si1"
    assert "xyz" in site


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
