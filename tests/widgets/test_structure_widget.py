"""Tests for StructureWidget rendering and notebook integration."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest

from pymatviz import StructureWidget
from tests.widgets.conftest import (
    assert_widget_build_files,
    assert_widget_edge_cases,
    assert_widget_notebook_integration,
    assert_widget_property_sync,
)


if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize(
    "structure_fixture",
    ["structures", "fe3co4_disordered", "fe3co4_disordered_with_props"],
)
def test_widget_structure_inputs(
    structure_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Widget must handle different structure input types and formats."""
    fixture_value = request.getfixturevalue(structure_fixture)

    # Handle both single structures and tuples of structures
    structures = fixture_value if isinstance(fixture_value, tuple) else (fixture_value,)

    for structure in structures:
        # Test with Structure object
        widget = StructureWidget(structure=structure)
        assert widget.widget_type == "structure"
        assert widget.structure == structure.as_dict()

        # Test with structure dict
        structure_dict = structure.as_dict()
        widget = StructureWidget(structure=structure_dict)
        assert widget.structure == structure_dict

        # Test structure assignment and update
        widget = StructureWidget()
        widget.structure = structure_dict
        assert widget.structure == structure_dict


@pytest.mark.parametrize(
    ("invalid_structure", "expected_error"),
    [("invalid_structure", TypeError), (123, TypeError), (["invalid"], TypeError)],
)
def test_widget_invalid_structure_handling(
    invalid_structure: Any, expected_error: type[Exception]
) -> None:
    """Widget must handle invalid structure inputs gracefully."""
    with pytest.raises(expected_error):
        StructureWidget(structure=invalid_structure)


@pytest.mark.parametrize(
    ("property_name", "test_values"),
    [
        ("atom_radius", [1.0, 1.5, 2.0]),
        ("show_bonds", [True, False]),
        ("color_scheme", ["Jmol", "CPK", "VESTA"]),
        ("style", [None, "width: 400px; height: 600px", "width: 600px; height: 800px"]),
        ("show_controls", [True, False]),
        ("enable_info_pane", [True, False]),
    ],
)
def test_widget_property_sync_structure(
    structures: tuple[Structure, Structure],
    property_name: str,
    test_values: list[Any],
) -> None:
    """Widget properties must sync with frontend and handle various values."""
    widget = StructureWidget(structure=structures[0])

    for test_value in test_values:
        assert_widget_property_sync(widget, property_name, test_value)


def test_widget_build_files_and_display_structure() -> None:
    """Widget must load build files and display properly."""
    widget = StructureWidget()
    assert_widget_build_files(widget)


def test_widget_notebook_integration_structure(
    structures: tuple[Structure, Structure],
) -> None:
    """Widget must integrate properly with notebook environments."""
    widget = StructureWidget(structure=structures[0])
    assert_widget_notebook_integration(widget)


def test_widget_complete_lifecycle(structures: tuple[Structure, Structure]) -> None:
    """Test complete widget lifecycle including state persistence."""
    structure_dict = structures[0].as_dict()

    # Create widget with custom settings
    widget = StructureWidget(
        structure=structure_dict,
        atom_radius=1.5,
        show_bonds=True,
        color_scheme="Jmol",
        style="width: 800px; height: 600px",
        show_controls=False,
    )

    # Test initial state
    assert widget.structure == structure_dict
    assert widget.atom_radius == 1.5
    assert widget.show_bonds is True
    assert widget.color_scheme == "Jmol"
    assert widget.style == "width: 800px; height: 600px"
    assert widget.show_controls is False

    # Test state persistence
    state = {
        "structure": widget.structure,
        "atom_radius": widget.atom_radius,
        "show_bonds": widget.show_bonds,
        "color_scheme": widget.color_scheme,
        "style": widget.style,
        "show_controls": widget.show_controls,
    }

    # Create new widget from state
    restored_widget = StructureWidget(**state)

    # Verify state preservation
    for key, value in state.items():
        assert getattr(restored_widget, key) == value


def test_widget_performance_and_large_structures(
    structures: tuple[Structure, Structure],
) -> None:
    """Test widget performance with large structures."""
    # Test creation performance
    start_time = time.perf_counter()
    for _ in range(10):
        StructureWidget(structure=structures[0])
    creation_time = time.perf_counter() - start_time
    assert creation_time < 0.3

    # Test large structure handling (10x10x10 supercell with 2000 atoms)
    large_structure = structures[0] * 10
    assert len(large_structure) == 2_000

    # Widget should handle large structure without crashing
    widget = StructureWidget(structure=large_structure)
    assert widget.structure is not None
    assert len(widget.structure["sites"]) == 2000


def test_widget_edge_cases_and_error_handling_structure(
    structures: tuple[Structure, Structure],
) -> None:
    """Test widget edge cases and error handling."""
    # Test widget with no structure
    widget = StructureWidget()
    assert widget.structure is None

    # Test widget handles missing/corrupted build files gracefully
    widget = StructureWidget(structure=structures[0].as_dict())
    assert_widget_edge_cases(widget)

    # Test structure serialization
    json.dumps(widget.structure)  # Should not raise exception


def test_widget_with_disordered_structure(
    fe3co4_disordered: Structure, fe3co4_disordered_with_props: Structure
) -> None:
    """Test widget with disordered structures and site properties."""
    # Test disordered structure without properties
    widget1 = StructureWidget(structure=fe3co4_disordered)
    assert widget1.structure == fe3co4_disordered.as_dict()

    # Test disordered structure with properties
    widget2 = StructureWidget(structure=fe3co4_disordered_with_props)
    assert widget2.structure == fe3co4_disordered_with_props.as_dict()

    uniq_prop_keys = {  # Verify the specific properties are preserved
        key for site in widget2.structure["sites"] for key in site.get("properties", {})
    }

    assert uniq_prop_keys == {"magmom", "force"}
