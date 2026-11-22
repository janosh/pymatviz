"""Native sunburst and treemap hierarchies for chemical system distributions."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import pytest
from pymatgen.core import Composition, Structure

import pymatviz as pmv


if TYPE_CHECKING:
    from pymatviz.typing import FormulaGroupBy


type ChemicalChart = functools.partial[
    pmv.widgets.TreemapWidget | pmv.widgets.SunburstWidget
]


@pytest.fixture(params=["treemap", "sunburst"])
def chart(request: pytest.FixtureRequest) -> ChemicalChart:
    """Run the same hierarchy contract through both chart kinds."""
    return functools.partial(pmv.chem_sys_hierarchy, kind=request.param)


@pytest.mark.parametrize("group_by", ["chem_sys", "formula", "reduced_formula"])
@pytest.mark.parametrize("overflow", ["other", "drop"])
def test_system_limit(
    chart: ChemicalChart, group_by: FormulaGroupBy, overflow: Literal["other", "drop"]
) -> None:
    """Keep the most frequent whole system per arity (O beats Fe by count), with
    alphabetical tie-breaking (Fe-O beats Li-O at 3 each) across all formulas.
    """
    samples = ["Li2O"] * 3 + ["Fe2O3", "Fe4O6", "FeO", "Na2O", "Fe", "O", "O"]
    widget = chart(
        samples, group_by=group_by, max_systems=1, overflow=overflow, label_text="value"
    )
    assert widget.widget_type == chart.keywords["kind"]
    assert widget.label_text == "value"
    unary, binary = widget.data
    assert [unary["id"], binary["id"]] == ["unary", "binary"]
    assert all("value" not in branch for branch in (unary, binary))
    oxygen, *unary_other = unary["children"]
    iron_oxide, *binary_other = binary["children"]
    assert (oxygen["id"], oxygen["label"]) == ("unary/O", "O")
    assert (iron_oxide["id"], iron_oxide["label"]) == ("binary/Fe-O", "Fe-O")
    if group_by == "chem_sys":
        assert "children" not in iron_oxide
        assert (oxygen["value"], iron_oxide["value"]) == (2, 3)
    else:
        assert all("value" not in branch for branch in (oxygen, iron_oxide))
        o_formula = "O2" if group_by == "reduced_formula" else "O"  # reduced O -> O2
        assert oxygen["children"] == [
            {"id": f"unary/O/{o_formula}", "label": o_formula, "value": 2}
        ]
        formulas = (
            {"FeO": 1, "Fe2O3": 2}
            if group_by == "reduced_formula"
            else {"FeO": 1, "Fe2O3": 1, "Fe4O6": 1}
        )
        assert iron_oxide["children"] == [
            {"id": f"binary/Fe-O/{formula}", "label": formula, "value": count}
            for formula, count in sorted(formulas.items())
        ]
    if overflow == "other":
        assert unary_other == [
            {"id": "unary/__omitted__", "label": "Other (1 more not shown)", "value": 1}
        ]
        assert binary_other == [
            {
                "id": "binary/__omitted__",
                "label": "Other (2 more not shown)",
                "value": 4,
            }
        ]
    else:
        assert not unary_other
        assert not binary_other


@pytest.mark.parametrize("limit", [None, 99])
def test_mixed_inputs(chart: ChemicalChart, limit: int | None) -> None:
    """Mixed inputs produce nodes only for arities present in the data."""
    structure = Structure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["Fe", "O"], [[0, 0, 0], [0.5] * 3]
    )
    inputs = ["Fe-O", " Fe2O3 ", structure, Composition("LiPO4"), "Zr48Al8Cu32Ni4Ag8"]
    widget = chart(inputs, max_systems=limit)
    assert [(root["label"], root["children"]) for root in widget.data] == [
        ("binary", [{"id": "binary/Fe-O", "label": "Fe-O", "value": 3}]),
        ("ternary", [{"id": "ternary/Li-O-P", "label": "Li-O-P", "value": 1}]),
        (
            "quinary",
            [{"id": "quinary/Ag-Al-Cu-Ni-Zr", "label": "Ag-Al-Cu-Ni-Zr", "value": 1}],
        ),
    ]


@pytest.mark.parametrize(
    ("samples", "kwargs", "error", "message"),
    [
        ([], {}, ValueError, "Empty input"),
        (["?"], {}, ValueError, "Invalid formula"),
        ([1], {}, TypeError, "Expected a formula string, mapping, Composition or"),
        (["Fe"], {"overflow": "merge"}, ValueError, "Invalid overflow="),
        (["Fe"], {"max_cells": 1}, TypeError, "max_cells"),
    ],
)
def test_invalid_inputs(
    chart: ChemicalChart,
    samples: list[Any],
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    """Reject invalid data, unknown modes and unknown widget options."""
    with pytest.raises(error, match=message):
        chart(samples, **kwargs)
