"""Tests for CompositionWidget rendering and notebook integration."""

from __future__ import annotations

import importlib.util
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


def _pmg_species_inputs() -> list[Any]:
    """Fe2+ O2- as a pymatgen Composition and as a {Species: amount} dict."""
    from pymatgen.core import Composition, Species

    return [
        Composition({"Fe2+": 1, "O2-": 1}),
        {Species("Fe", 2): 1, Species("O", -2): 1},
    ]


@pytest.mark.parametrize(
    ("species_input", "expected"),
    [
        ({"Fe2+": 1, "O2-": 1}, {"Fe2+": 1.0, "O2-": 1.0}),
        ({"Fe3+": 0.5, "Fe2+": 0.5, "O2-": 1}, {"Fe3+": 0.5, "Fe2+": 0.5, "O2-": 1.0}),
        ({"Na+": 1, "Cl": 1}, {"Na+": 1.0, "Cl": 1.0}),  # species mixed with elements
        ({"Fe2.5+": 2, "O2-": 5}, {"Fe2.5+": 2.0, "O2-": 5.0}),
        ({26: 1, "O2-": 1}, {"Fe": 1.0, "O2-": 1.0}),  # atomic numbers as symbols
        ({"Fe2+": 0, "O2-": 1}, {"O2-": 1.0}),  # zero amounts dropped, like kemio
    ],
)
def test_widget_keeps_oxidation_states(
    species_input: dict[Any, float], expected: dict[str, float]
) -> None:
    """Species keys sync as given: MatterViz draws Fe2+ and Fe3+ as separate
    segments colored by Fe.
    """
    composition = CompositionWidget(species_input).composition
    assert composition == expected
    assert {type(amount) for amount in composition.values()} == {float}


def test_widget_keeps_pymatgen_species() -> None:
    """Pymatgen Compositions, {Species: amount} dicts and species structures keep
    their species; Species properties are ignored and keys naming the same
    species are summed.
    """
    from pymatgen.core import Lattice, Species, Structure

    fe_o_structure = Structure(
        Lattice.cubic(4), ["Fe2+", "O2-"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    spin_species = {Species("Fe", 2, spin=5): 1, Species("O", -2): 1}
    for comp in [*_pmg_species_inputs(), fe_o_structure, spin_species]:
        assert CompositionWidget(comp).composition == {"Fe2+": 1.0, "O2-": 1.0}
    same_species = {"Fe2+": 1, Species("Fe", 2): 2, Species("Fe", 2.5): 1}
    expected = {"Fe2+": 3.0, "Fe2.5+": 1.0}
    assert CompositionWidget(same_species).composition == expected


@pytest.mark.parametrize(
    ("species_input", "match"),
    [
        ({"Xx2+": 1, "O2-": 1}, r"unknown element\(s\) \['Xx'\]"),
        ({"Fe2+": 1, "Qq": 1}, r"unknown element\(s\) \['Qq'\]"),
        ({"Fe2+": "a lot"}, "could not convert string to float"),
        ({"Fe02+": 1}, r"must look like Fe2\+ or O-, got \['Fe02\+'\]"),
        ({"Fe12+": 1, "O": 1}, r"got \['Fe12\+'\]"),  # MatterViz rejects these too
        (
            {"Fe2+": float("nan"), "O2-": 1},
            r"amounts must be finite, got \['Fe2\+'\]",
        ),
    ],
)
def test_widget_rejects_invalid_species(
    species_input: dict[str, Any], match: str
) -> None:
    """Species compositions with unknown elements or non-numeric amounts raise."""
    with pytest.raises(tl.TraitError, match=match):
        CompositionWidget(species_input)


@pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="needs playwright"
)
def test_widget_species_composition_renders() -> None:
    """A species composition renders headlessly (needs a matterviz-anywidget bundle
    whose composition charts accept species keys).
    """
    report = CompositionWidget({"Fe2+": 1, "O2-": 1}).render_report(timeout=60)
    assert report.ok, report.error
    assert not report.is_blank


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
