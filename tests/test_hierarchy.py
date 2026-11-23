"""Tests for pymatviz.hierarchy: chemical system, chemical environment and space group
sunbursts and treemaps.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal, get_args
from unittest.mock import MagicMock

import kemio
import numpy as np
import pandas as pd
import pytest
from kemio.options import CrystalSystem

import pymatviz as pmv
from pymatviz import hierarchy
from pymatviz.data import TEST_FILES
from pymatviz.hierarchy import spacegroup_sunburst
from pymatviz.structures import to_structure
from tests.conftest import SI_STRUCTS, cubic_structure, rocksalt


if TYPE_CHECKING:
    from pymatviz.typing import FormulaGroupBy


def spg_leaves(widget: pmv.widgets.SunburstWidget) -> dict[str, tuple[str, int]]:
    """Map leaf IDs to (label, value) of a crystal system/space group sunburst."""
    return {
        leaf["id"]: (leaf["label"], leaf["value"])
        for root in widget.data
        for leaf in root["children"]
    }


def test_spacegroup_sunburst_all_groups() -> None:
    """All 230 space groups nest under the 7 crystal systems with label+value text."""
    widget = spacegroup_sunburst(range(1, 231))
    assert isinstance(widget, pmv.widgets.SunburstWidget)
    assert widget.label_text == "label+value"
    assert {root["id"] for root in widget.data} == set(get_args(CrystalSystem))
    leaves = spg_leaves(widget)
    assert len(leaves) == 230
    assert leaves["cubic/225"] == ("225", 1)
    assert spacegroup_sunburst([1], label_text="percent").label_text == "percent"


@pytest.mark.parametrize(
    ("invalid", "err", "match"),
    [
        *[(val, ValueError, "missing values") for val in (None, float("nan"), pd.NA)],
        *[(val, ValueError, r"must be in \[1, 230\]") for val in (0, 231, 2.5)],
        ("invalid", TypeError, "one kind"),
        (True, TypeError, "bool"),
    ],
)
def test_spacegroup_sunburst_invalid_data(
    invalid: Any, err: type[Exception], match: str
) -> None:
    """Reject missing, out-of-range, mixed-kind and bool groups instead of losing
    or mislabeling slices.
    """
    with pytest.raises(err, match=match):
        spacegroup_sunburst([1, invalid])


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([1], {"triclinic/1": ("1", 1)}),
        # symbol spellings and int/float numbers of one group share a canonical slice
        (
            ["Fm-3m", "F m -3 m", "P1"],
            {"cubic/Fm-3m": ("Fm-3m", 2), "triclinic/P1": ("P1", 1)},
        ),
        ([225, 225.0], {"cubic/225": ("225", 2)}),
        (np.array([225, 225, 1]), {"cubic/225": ("225", 2), "triclinic/1": ("1", 1)}),
        (["P1"], {"triclinic/P1": ("P1", 1)}),
        (
            pd.Series([*[1] * 3, *[2] * 10, *[3] * 5]),
            {
                "triclinic/2": ("2", 10),
                "monoclinic/3": ("3", 5),
                "triclinic/1": ("1", 3),
            },
        ),
        (SI_STRUCTS, {"monoclinic/12": ("12", 1), "tetragonal/129": ("129", 1)}),
    ],
)
def test_spacegroup_sunburst_counts(data: Any, expected: dict[str, Any]) -> None:
    """Numbers, symbols, Series and structures are counted per space group."""
    assert spg_leaves(spacegroup_sunburst(data)) == expected


def test_spacegroup_sunburst_symbol_counts(spg_symbols: list[str]) -> None:
    """Space-group symbols are grouped into crystal systems with counts."""
    widget = spacegroup_sunburst(spg_symbols)
    totals = {
        root["id"]: sum(leaf["value"] for leaf in root["children"])
        for root in widget.data
    }
    assert (totals["monoclinic"], totals["cubic"], totals["hexagonal"]) == (4, 5, 3)
    leaves = spg_leaves(widget)
    assert leaves["monoclinic/C2/m"][0] == "C2/m"
    assert leaves["cubic/P-43m"][0] == "P-43m"


@pytest.mark.parametrize("pbc", [(False,) * 3, (True, True, False)])
def test_spacegroup_sunburst_rejects_non_periodic(
    structures: list[kemio.Structure], pbc: tuple[bool, bool, bool]
) -> None:
    """Structures not periodic along all three axes are rejected with how to fix it."""
    struct = kemio.Structure(np.eye(3) * 3, ["Si"], [[0, 0, 0]], pbc=pbc)
    with pytest.raises(
        TypeError, match=r"periodic along a, b and c.*atoms\.pbc = True"
    ):
        spacegroup_sunburst([*structures, struct])
    assert spacegroup_sunburst([*structures, struct.replace(pbc=(True,) * 3)])


@pytest.mark.parametrize(
    ("max_spacegroups", "overflow", "expected_systems", "n_hidden"),
    # spg 1-2 are triclinic (2 entries), spg 3-7 monoclinic (5 entries)
    [
        (None, "other", 7, []),  # no limit
        (10, "other", 7, []),  # more than any system holds
        (1, "other", 4, [1, 4]),
        (2, "other", 5, [3]),  # triclinic fits fully
        (3, "other", 6, [2]),
        (1, "drop", 2, []),
        (2, "drop", 4, []),
        (3, "drop", 5, []),  # top 3 monoclinic + both triclinic
    ],
)
def test_spacegroup_sunburst_max_spacegroups(
    max_spacegroups: int | None,
    overflow: Literal["other", "drop"],
    expected_systems: int,
    n_hidden: list[int],
) -> None:
    """max_spacegroups limits space groups shown *per crystal system* (regression:
    the limit used to be applied globally, silently dropping entire crystal systems).
    """
    widget = pmv.spacegroup_sunburst(
        range(1, 8), max_spacegroups=max_spacegroups, overflow=overflow
    )
    space_groups = dict(spg_leaves(widget).values())
    assert len(space_groups) == expected_systems
    others = sorted(
        (label, val) for label, val in space_groups.items() if "more not shown" in label
    )
    assert others == sorted((f"Other ({num} more not shown)", num) for num in n_hidden)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_spacegroups": 0}, "at least 1 or None, got limit=0"),
        ({"max_spacegroups": -1}, "at least 1 or None, got limit=-1"),
        ({"max_spacegroups": 1, "overflow": "invalid"}, "Invalid overflow="),
    ],
)
def test_spacegroup_sunburst_invalid_limits(kwargs: dict[str, Any], match: str) -> None:
    """max_spacegroups below 1 (instead of silently meaning no limit) and unknown
    overflow modes raise.
    """
    with pytest.raises(ValueError, match=match):
        pmv.spacegroup_sunburst([1, 2, 3], **kwargs)


@pytest.mark.parametrize("normalize", [False, True])
def test_chem_env_sunburst_rock_salt(normalize: bool) -> None:
    """ChemEnv classifies all eight rocksalt sites as CN 6 octahedra (O:6)."""
    rock_salt = rocksalt()
    empty = kemio.Structure(rock_salt.lattice, [], np.zeros((0, 3)))
    widget = pmv.chem_env_hierarchy([empty, rock_salt], normalize=normalize)
    expected = 1 if normalize else len(rock_salt)
    assert widget.data == [
        dict(
            id="CN 6",
            label="CN 6",
            children=[dict(id="CN 6/O:6", label="O:6", value=expected)],
        )
    ]


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
        expected_children = [
            {"id": f"binary/Fe-O/{formula}", "label": formula, "value": count}
            for formula, count in sorted(formulas.items())
        ]
        assert iron_oxide["children"] == expected_children
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
    structure = cubic_structure(3, ["Fe", "O"], [[0, 0, 0], [0.5] * 3])
    inputs = [
        "Fe-O",
        " Fe2O3 ",
        structure,
        kemio.Composition("LiPO4"),
        "Zr48Al8Cu32Ni4Ag8",
    ]
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


@pytest.mark.parametrize("chart", ["treemap", "sunburst"])
@pytest.mark.parametrize("overflow", ["other", "drop"])
def test_chem_env_hierarchy(
    chart: Literal["treemap", "sunburst"],
    overflow: Literal["other", "drop"],
    structures: tuple[kemio.Structure, kemio.Structure],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate duplicate symbols, weight sites per structure, rank full CNs, break
    ties and limit both levels.
    """
    # normalized: 1/4 per site of the first structure, 1/2 of the second
    envs = [
        [(6, "O:6"), (4, "T:4"), (4, "see-saw-like:4"), (4, "see-saw-like:4")],
        [(4, "T:4"), (8, "C:8")],
    ]
    collect = MagicMock(return_value=envs)
    monkeypatch.setattr(hierarchy, "site_environments", collect)
    widget = pmv.chem_env_hierarchy(
        structures,
        kind=chart,
        normalize=True,
        max_coordination_numbers=1,
        max_environments=1,
        overflow=overflow,
        value_format=".2f",
    )
    collect.assert_called_once()
    assert len(collect.call_args.args[0]) == 2
    assert widget.widget_type == chart
    assert widget.value_format == ".2f"
    root, *omitted = widget.data
    children = [dict(id="CN 4/T:4", label="T:4", value=0.75)]
    if overflow == "other":
        children.append(
            dict(id="CN 4/__omitted__", label="Other (1 more not shown)", value=0.5)
        )
    assert root == dict(id="CN 4", label="CN 4", children=children)
    assert omitted == (
        [dict(id="__omitted__", label="Other (2 more not shown)", value=0.75)]
        if overflow == "other"
        else []
    )


@pytest.mark.parametrize(
    ("structure", "kwargs", "err_cls", "match"),
    [
        (rocksalt(), {"overflow": "keep"}, ValueError, "overflow"),
        (rocksalt(), {"kind": "pie"}, ValueError, "Invalid kind='pie'"),
        (
            kemio.Structure(None, ["H"] * 2, [[0] * 3, [0.7, 0, 0]], pbc=(False,) * 3),
            {},
            TypeError,
            "Expected structures periodic along a, b and c",
        ),
    ],
    ids=["invalid-overflow", "invalid-kind", "non-periodic"],
)
def test_chem_env_rejects_before_classifying(
    structure: kemio.Structure,
    kwargs: dict[str, Any],
    err_cls: type[Exception],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid overflow, kind and nonperiodic inputs fail before neighbor analysis."""
    collect = MagicMock()
    monkeypatch.setattr(hierarchy, "site_environments", collect)
    all_kwargs: dict[str, Any] = {"kind": "treemap", **kwargs}
    with pytest.raises(err_cls, match=match):
        pmv.chem_env_hierarchy(structure, **all_kwargs)
    collect.assert_not_called()


def test_site_environments_chemenv_without_reference_geometry() -> None:
    """Sites whose CN has no ChemEnv reference geometry are labeled CN:<n>."""
    struct = to_structure(f"{TEST_FILES}/structures/mp-12712.json")
    (envs,) = hierarchy.site_environments([struct])
    assert [envs[idx] for idx in (24, 32, 40)] == [(14, "CN:14")] * 3
