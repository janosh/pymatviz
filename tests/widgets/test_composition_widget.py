"""Tests for CompositionWidget rendering and notebook integration."""

from __future__ import annotations

import json
from typing import Any

import kemio
import pytest
import traitlets as tl

from pymatviz.widgets.composition import CompositionWidget
from tests.widgets.conftest import assert_widget_property_sync


@pytest.mark.parametrize(
    "composition_input",
    [
        "Fe2O3",
        {"Fe": 2, "O": 3},
        "Li0.5FePO4",
        "CuSO4(H2O)5",
        "Al0.2Co0.2Cr0.2Fe0.2Ni0.2",
        "Fe",
        pytest.param({}, id="empty"),
        pytest.param({"Fe": 1000.0, "O": 1500.0}, id="large-counts"),
        pytest.param(
            kemio.Composition(
                "Li0.1Na0.05K0.05Mg0.1Ca0.05Ba0.05Al0.1Si0.2Ti0.05V0.05Cr0.05"
                "Mn0.1Fe0.15Co0.05Ni0.05Cu0.05Zn0.05O1.0"
            ),
            id="many-elements",
        ),
    ],
)
def test_widget_composition_inputs(
    composition_input: str | dict[str, float] | kemio.Composition,
) -> None:
    """Widget must handle different composition input types and formats."""
    widget = CompositionWidget(composition=composition_input)
    assert widget.widget_type == "composition"
    expected = kemio.Composition(composition_input).as_dict()
    assert widget.composition == expected
    json.dumps(widget.composition)
    widget = CompositionWidget()  # later assignments convert too
    widget.composition = composition_input
    assert widget.composition == expected


@pytest.mark.parametrize(
    ("invalid_composition", "expected_error", "match"),
    [
        ("InvalidComposition123", tl.TraitError, 'unknown element "Invalid"'),
        ({"invalid": "composition"}, tl.TraitError, "must be real number, not str"),
    ],
)
def test_widget_invalid_composition_handling(
    invalid_composition: Any, expected_error: type[Exception], match: str
) -> None:
    """Invalid formulas and non-numeric amounts raise naming the problem."""
    with pytest.raises(expected_error, match=match):
        CompositionWidget(composition=invalid_composition)


@pytest.mark.parametrize(
    ("property_name", "test_values"),
    [
        ("show_percentages", [False, True]),
        ("color_scheme", ["Jmol", "Alloy", "Vesta"]),
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
