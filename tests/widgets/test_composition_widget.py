"""Tests for CompositionWidget rendering and notebook integration."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pymatgen.core import Composition

from pymatviz.widgets.composition import CompositionWidget
from tests.widgets.conftest import (
    assert_widget_build_files,
    assert_widget_notebook_integration,
    assert_widget_property_sync,
)


@pytest.mark.parametrize(
    ("composition_input", "expected_elements"),
    [
        ("Fe2O3", {"Fe", "O"}),
        ({"Fe": 2, "O": 3}, {"Fe", "O"}),
        ("Li0.5FePO4", {"Li", "Fe", "P", "O"}),
        ("CuSO4(H2O)5", {"Cu", "S", "O", "H"}),
        ("Al0.2Co0.2Cr0.2Fe0.2Ni0.2", {"Al", "Co", "Cr", "Fe", "Ni"}),
        ("Fe", {"Fe"}),  # Single element
    ],
)
def test_widget_composition_inputs(
    composition_input: str | dict[str, float], expected_elements: set[str]
) -> None:
    """Widget must handle different composition input types and formats."""
    widget = CompositionWidget(composition=composition_input)
    assert widget.widget_type == "composition"
    assert widget.composition is not None
    assert isinstance(widget.composition, dict)
    assert set(widget.composition) == expected_elements

    # Test composition is properly converted to dict
    expected_comp = Composition(composition_input).as_dict()
    assert widget.composition == expected_comp


@pytest.mark.parametrize(
    ("invalid_composition", "expected_error"),
    [("InvalidComposition123", ValueError), ({"invalid": "composition"}, TypeError)],
)
def test_widget_invalid_composition_handling(
    invalid_composition: Any, expected_error: type[Exception]
) -> None:
    """Widget must handle invalid composition inputs gracefully."""
    with pytest.raises(expected_error):
        CompositionWidget(composition=invalid_composition)


@pytest.mark.parametrize(
    ("property_name", "test_values"),
    [
        ("show_percentages", [False, True]),
        ("color_scheme", ["Jmol", "CPK", "Vesta"]),
        ("style", [None, "width: 400px; height: 600px", "width: 600px; height: 800px"]),
        ("mode", ["pie", "bar", "bubble"]),
    ],
)
def test_widget_property_sync_composition(
    property_name: str, test_values: list[Any]
) -> None:
    """Widget properties must sync with frontend and handle various values."""
    widget = CompositionWidget(composition="Fe2O3")

    for test_value in test_values:
        assert_widget_property_sync(widget, property_name, test_value)


def test_widget_build_files_and_display_composition() -> None:
    """Widget must load build files and display properly."""
    widget = CompositionWidget(composition="Fe2O3")
    assert_widget_build_files(widget)

    # Test composition is JSON serializable
    json.dumps(widget.composition)


def test_widget_notebook_integration_composition() -> None:
    """Widget must integrate properly with notebook environments."""
    widget = CompositionWidget(composition="Fe2O3")
    assert_widget_notebook_integration(widget)


def test_widget_complete_lifecycle() -> None:
    """Test complete widget lifecycle including state persistence."""
    # Create widget with custom settings
    widget = CompositionWidget(
        composition="Fe2O3",
        show_percentages=True,
        color_scheme="CPK",
        mode="bar",
        style="width: 800px; height: 600px",
    )

    # Test initial state
    assert widget.composition == Composition("Fe2O3").as_dict()
    assert widget.show_percentages is True
    assert widget.color_scheme == "CPK"
    assert widget.mode == "bar"
    assert widget.style == "width: 800px; height: 600px"

    # Test state persistence
    state = {
        "composition": widget.composition,
        "show_percentages": widget.show_percentages,
        "color_scheme": widget.color_scheme,
        "mode": widget.mode,
        "style": widget.style,
    }

    # Create new widget from state
    restored_widget = CompositionWidget(**state)

    # Verify state preservation
    for key, value in state.items():
        assert getattr(restored_widget, key) == value


def test_widget_complex_composition_handling() -> None:
    """Test widget handles complex and large-number compositions correctly."""
    # Test complex composition handling
    complex_comp = Composition(
        "Li0.1Na0.05K0.05Mg0.1Ca0.05Ba0.05Al0.1Si0.2Ti0.05V0.05Cr0.05"
        "Mn0.1Fe0.15Co0.05Ni0.05Cu0.05Zn0.05O1.0"
    )
    widget = CompositionWidget(composition=complex_comp)
    assert widget.composition is not None
    assert len(widget.composition) > 10  # Many elements

    # Test large number handling
    large_comp = {"Fe": 1000.0, "O": 1500.0}
    widget = CompositionWidget(composition=large_comp)
    expected_comp = Composition(large_comp).as_dict()
    assert widget.composition == expected_comp


def test_widget_edge_cases_composition() -> None:
    """Test widget edge cases and special scenarios."""
    # Test empty composition (valid in pymatgen)
    empty_widget = CompositionWidget(composition={})
    assert empty_widget.composition == {}

    # Build-asset sanity check on a regular instance.
    widget = CompositionWidget(composition="Fe2O3")
    assert_widget_build_files(widget)
