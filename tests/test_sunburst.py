"""Test sunburst plots for space groups and chemical systems."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, get_args
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.sunburst import ShowCounts, spacegroup_sunburst
from pymatviz.sunburst.chem_env import _process_chem_env_data_sunburst
from pymatviz.sunburst.helpers import _limit_slices


if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go

    from pymatviz.typing import FormulaGroupBy

CHEMENV_MODULE = "pymatgen.analysis.chemenv.coordination_environments"


@pytest.fixture
def fe_structure() -> Structure:
    """Single-site simple cubic Fe structure."""
    return Structure(Lattice.cubic(4.0), ["Fe"], [[0.0, 0.0, 0.0]])


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_spacegroup_sunburst(show_counts: ShowCounts) -> None:
    """Test spacegroup_sunburst with different show_counts options."""
    fig = spacegroup_sunburst(range(1, 231), show_counts=show_counts)

    assert set(fig.data[0].parents) == {"", *get_args(pmv.typing.CrystalSystem)}
    assert fig.data[0].branchvalues == "total"

    if show_counts == "value":
        assert fig.data[0].texttemplate is not None
        assert "N=" in fig.data[0].texttemplate
    elif show_counts == "value+percent":
        assert fig.data[0].texttemplate is not None
        assert "N=" in fig.data[0].texttemplate
        assert "percentEntry" in fig.data[0].texttemplate
    elif show_counts == "percent":
        assert fig.data[0].textinfo is not None
        assert "percent" in fig.data[0].textinfo
    elif show_counts is False:
        assert fig.data[0].textinfo is None


def test_spacegroup_sunburst_invalid_show_counts() -> None:
    """Test that invalid show_counts values raise ValueError."""
    with pytest.raises(ValueError, match=r"Invalid.*show_counts"):
        spacegroup_sunburst([1], show_counts="invalid")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("invalid", [None, float("nan"), pd.NA, 0, 231, 2.5, "invalid"])
def test_spacegroup_sunburst_invalid_data(invalid: Any) -> None:
    """Reject missing and invalid groups instead of losing or mislabeling slices."""
    with pytest.raises(ValueError, match=r"missing values|Invalid space group"):
        spacegroup_sunburst([1, invalid])


def test_spacegroup_sunburst_single_item() -> None:
    """Test with single-item input."""
    fig = spacegroup_sunburst([1], show_counts="value")
    assert len(fig.data[0].ids) == 2  # one for crystal system, one for spg number

    fig = spacegroup_sunburst(["P1"], show_counts="value+percent")
    assert len(fig.data[0].ids) == 2


def test_spacegroup_sunburst_series_counts() -> None:
    """Test with other types of input."""
    # test with pandas series
    series = pd.Series([*[1] * 3, *[2] * 10, *[3] * 5])
    fig = spacegroup_sunburst(series, show_counts="value")
    values = [*map(int, fig.data[0].values)]
    assert values == [10, 5, 3, 13, 5], f"actual {values=}"


def test_spacegroup_sunburst_symbol_counts(spg_symbols: list[str]) -> None:
    """Space-group symbols are grouped into crystal systems with counts."""
    fig = spacegroup_sunburst(spg_symbols, show_counts="value")
    trace = fig.data[0]
    values_by_label = dict(zip(trace.labels, map(int, trace.values), strict=True))
    parents_by_label = dict(zip(trace.labels, trace.parents, strict=True))

    assert values_by_label["monoclinic"] == 4
    assert values_by_label["cubic"] == 5
    assert values_by_label["hexagonal"] == 3
    assert parents_by_label["C2/m"] == "monoclinic"
    assert parents_by_label["P-43m"] == "cubic"


def test_spacegroup_sunburst_structure_counts(
    structures: tuple[Structure, Structure],
) -> None:
    """Structure inputs are converted to space group counts via moyopy."""
    fig = spacegroup_sunburst(structures, show_counts="value")
    trace = fig.data[0]

    assert list(trace.labels) == ["12", "129", "monoclinic", "tetragonal"]
    assert list(trace.parents) == ["monoclinic", "tetragonal", "", ""]
    assert [*map(int, trace.values)] == [1, 1, 1, 1]


def test_spacegroup_sunburst_requires_moyopy(
    structures: list[Structure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report the missing required backend for Structure input."""
    monkeypatch.setitem(sys.modules, "moyopy", None)
    with pytest.raises(ModuleNotFoundError):
        spacegroup_sunburst(structures)


def test_spacegroup_sunburst_reports_moyopy_conversion_failure(
    structures: list[Structure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report index and type when moyopy cannot convert a structure."""

    def reject_structure(_struct: Any) -> int:
        """Simulate unsupported structure input."""
        raise TypeError("unsupported structure")

    monkeypatch.setattr(
        "pymatviz.sunburst.spacegroup.get_spacegroup_number", reject_structure
    )

    with pytest.raises(TypeError, match="structure at index 0 \\(Structure\\)"):
        spacegroup_sunburst(structures)


@pytest.mark.parametrize(
    ("max_slices", "max_slices_mode", "expected_systems", "expected_others"),
    [
        # Edge cases for max_slices
        (None, "other", 7, None),  # no limit
        (0, "other", 7, None),  # treated as None
        (-1, "other", 7, None),  # negative treated as None
        (10, "other", 7, None),  # greater than number of systems
        # Normal cases with "other" mode. max_slices applies per crystal system:
        # spg 1-2 are triclinic (2 entries), spg 3-7 monoclinic (5 entries)
        (
            1,
            "other",
            4,
            [("Other (1 more not shown)", 1), ("Other (4 more not shown)", 4)],
        ),
        (2, "other", 5, [("Other (3 more not shown)", 3)]),  # triclinic fits fully
        (3, "other", 6, [("Other (2 more not shown)", 2)]),
        # Cases with "drop" mode (per crystal system)
        (1, "drop", 2, None),  # top 1 per crystal system
        (2, "drop", 4, None),  # top 2 per crystal system
        (3, "drop", 5, None),  # top 3 monoclinic + both triclinic
    ],
)
def test_spacegroup_sunburst_max_slices(
    max_slices: int | None,
    max_slices_mode: Literal["other", "drop"],
    expected_systems: int,
    expected_others: list[tuple[str, int]] | None,
) -> None:
    """Test spacegroup_sunburst with max_slices functionality and edge cases.

    max_slices limits space groups shown *per crystal system* (regression: the
    limit used to be applied globally, silently dropping entire crystal systems).
    """
    # Create a dataset with multiple space groups per crystal system
    spg_numbers = (1, 2, 3, 4, 5, 6, 7)

    fig = pmv.spacegroup_sunburst(
        spg_numbers, max_slices=max_slices, max_slices_mode=max_slices_mode
    )

    # Get all space group entries (excluding crystal system level)
    space_groups = {
        label: val
        for idx, (label, val) in enumerate(
            zip(fig.data[0].labels, fig.data[0].values, strict=True)
        )
        if fig.data[0].parents[idx] != ""  # exclude root level
    }
    assert len(space_groups) == expected_systems

    # Check for expected "Other" entries
    if expected_others:
        other_entries = [
            (label, val)
            for label, val in space_groups.items()
            if "more not shown" in str(label)
        ]
        assert len(other_entries) == len(expected_others)

        for expected_label, expected_count in expected_others:
            found = any(
                expected_label in label and val == expected_count
                for label, val in other_entries
            )
            assert found, (
                f"{expected_label=} with {expected_count=} not in {other_entries=}"
            )
    else:
        assert not any("Other" in str(label) for label in space_groups)


@pytest.mark.parametrize(
    ("sunburst_fn", "data"),
    [(pmv.spacegroup_sunburst, [1, 2, 3]), (pmv.chem_sys_sunburst, ["Fe2O3"])],
    ids=["spacegroup", "chem_sys"],
)
def test_sunburst_max_slices_mode_invalid(
    sunburst_fn: Callable[..., go.Figure], data: list[Any]
) -> None:
    """Invalid max_slices_mode raises."""
    with pytest.raises(ValueError, match="Invalid max_slices_mode="):
        sunburst_fn(data, max_slices=1, max_slices_mode="invalid")


def test_chem_sys_sunburst_basic() -> None:
    """Test chem_sys_sunburst plot with various scenarios."""
    systems = ["Fe-O", "Li-P-O", "Li-Fe-P-O", "Fe", "O", "Li-O"]
    fig = pmv.chem_sys_sunburst(systems)
    assert len(fig.data) == 1

    labels = fig.data[0].labels
    parents = fig.data[0].parents

    # Check arity levels are present and correctly named
    arity_levels = {
        label for label in labels if label in {"unary", "binary", "ternary"}
    }
    assert arity_levels == {"unary", "binary", "ternary"}

    # Check specific systems
    unary_elements = {
        label for idx, label in enumerate(labels) if parents[idx] == "unary"
    }
    assert unary_elements == {"Fe", "O"}

    binary_systems = {
        label for idx, label in enumerate(labels) if parents[idx] == "binary"
    }
    assert binary_systems == {"Fe-O", "Li-O"}

    ternary_systems = {
        label for idx, label in enumerate(labels) if parents[idx] == "ternary"
    }
    assert ternary_systems == {"Li-O-P"}


def test_chem_sys_sunburst_empty_level() -> None:
    """Test that the sunburst plot handles systems with gaps in arity correctly."""
    # Only unary and ternary systems, no binary
    systems = ["Fe", "O", "Li-Fe-O"]
    fig = pmv.chem_sys_sunburst(systems)

    labels = fig.data[0].labels
    parents = fig.data[0].parents

    # Check arity levels (should only have unary and ternary since no binary systems)
    arity_levels = {
        label for label in labels if label in {"unary", "binary", "ternary"}
    }
    assert arity_levels == {"unary", "ternary"}

    # Check that no systems have binary as parent
    binary_systems = {
        label for idx, label in enumerate(labels) if parents[idx] == "binary"
    }
    assert not binary_systems, "There should be no binary systems"


@pytest.mark.parametrize("invalid_show_counts", ["invalid", "bad_value", 123, True])
def test_chem_sys_sunburst_invalid_show_counts(invalid_show_counts: Any) -> None:
    """Test that invalid show_counts values raise ValueError."""
    with pytest.raises(ValueError, match=r"Invalid.*show_counts"):
        pmv.chem_sys_sunburst(["Fe-O"], show_counts=invalid_show_counts)


def test_chem_sys_sunburst_input_types(structures: list[Structure]) -> None:
    """Test chem_sys_sunburst plot with various input types."""
    from pymatgen.core import Composition

    # Test with mixed input types
    systems = [
        "Fe2O3",  # formula string
        Composition("LiPO4"),  # pymatgen composition
        "Na2CO3",  # formula string
        structures[0],  # pymatgen structure
        "Li-Fe-P-O",  # chemical system string
        "Fe-O",  # chemical system string
    ]
    fig = pmv.chem_sys_sunburst(systems)

    # Check that Fe2O3 and Fe-O are counted together
    binary_count = sum(
        val
        for idx, val in enumerate(fig.data[0].values)
        if fig.data[0].parents[idx] == "binary" and fig.data[0].labels[idx] == "Fe-O"
    )
    assert binary_count == 2, "Fe2O3 and Fe-O should be merged"

    # Test with only Structure objects
    assert len(pmv.chem_sys_sunburst(structures).data[0].labels) > 0

    # Test with invalid input type
    with pytest.raises(TypeError, match="Expected str, Composition or Structure"):
        pmv.chem_sys_sunburst([1])  # ty: ignore[invalid-argument-type]


def test_chem_sys_sunburst_high_arity() -> None:
    """Test chem_sys_sunburst plot with high arity systems."""
    formulas = [
        "Zr42.5Ga2.5Cu55",  # ternary
        "Zr37.5Ga5Cu57.5",  # ternary
        "Zr40Ga7.5Cu52.5",  # ternary
        "Zr48Al8Cu32Ag8Pd4",  # quinary
        "Zr48Al8Cu30Ag8Pd6",  # quinary
    ]
    fig = pmv.chem_sys_sunburst(formulas)

    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values  # noqa: PD011

    # Check arity levels
    arity_levels = {label for label in labels if label in {"ternary", "quinary"}}
    assert arity_levels == {"ternary", "quinary"}

    # Get counts for each arity level
    arity_counts = {
        labels[idx]: values[idx]
        for idx in range(len(labels))
        if parents[idx] == ""  # root level = arity level
    }
    assert arity_counts == {"ternary": 3, "quinary": 2}


@pytest.mark.parametrize(
    ("input_data", "error_msg"),
    [
        ([], "Empty input: data sequence is empty"),
        ([""], "Invalid formula"),
        ([" "], "Invalid formula"),
        (["NotAFormula"], "Invalid formula"),
        (["123"], "Invalid formula"),
    ],
)
def test_chem_sys_sunburst_invalid_inputs(
    input_data: list[str], error_msg: str
) -> None:
    """Test chem_sys_sunburst with empty and invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        pmv.chem_sys_sunburst(input_data)


def test_chem_sys_sunburst_case_and_whitespace() -> None:
    """Test chem_sys_sunburst handles case and whitespace variations."""
    systems = ["Fe2O3", " Fe2O3", "Fe2O3 ", " Fe2O3 "]
    fig = pmv.chem_sys_sunburst(systems)

    # All variations should be counted as the same system
    binary_count = sum(
        val
        for idx, val in enumerate(fig.data[0].values)
        if fig.data[0].parents[idx] == "binary" and fig.data[0].labels[idx] == "Fe-O"
    )
    assert binary_count == 4, "All Fe2O3 variations should be counted together"


@pytest.mark.parametrize(
    ("group_by", "expected_counts"),
    [
        (
            "formula",  # each formula counted separately
            {"binary/Fe-O": 3, "binary/Fe-O/Fe2O3": 1, "binary/Fe-O/Fe4O6": 1},
        ),
        (
            "reduced_formula",  # Fe4O6 reduces to Fe2O3
            {"binary/Fe-O": 3, "binary/Fe-O/Fe2O3": 2, "binary/Fe-O/FeO": 1},
        ),
        ("chem_sys", {"binary/Fe-O": 3, "binary/Li-O": 1}),  # no formula level
    ],
)
def test_chem_sys_sunburst_grouping(
    group_by: FormulaGroupBy, expected_counts: dict[str, int]
) -> None:
    """Test chem_sys_sunburst with different grouping modes."""
    trace = pmv.chem_sys_sunburst(
        ["Fe2O3", "Fe4O6", "FeO", "Li2O"], group_by=group_by
    ).data[0]
    counts = dict(zip(trace.ids, map(int, trace["values"]), strict=True))
    assert {key: counts.get(key) for key in expected_counts} == expected_counts
    assert (len(counts) == 3) == (group_by == "chem_sys")


@pytest.mark.parametrize(
    ("max_slices", "expected_systems", "expected_other"),
    [
        (None, 4, None),  # no limit
        (0, 4, None),  # treated as None
        (1, 2, ("Other (3 more not shown)", 3)),
        (2, 3, ("Other (2 more not shown)", 2)),
        (10, 4, None),  # greater than number of systems
    ],
)
def test_chem_sys_sunburst_max_slices(
    max_slices: int | None,
    expected_systems: int,
    expected_other: tuple[str, int] | None,
) -> None:
    """Test chem_sys_sunburst with max_slices functionality."""
    systems = ["Fe2O3", "Li2O", "Na2O", "K2O"]
    fig = pmv.chem_sys_sunburst(systems, max_slices=max_slices, max_slices_mode="other")

    binary = {
        label: val
        for idx, (label, val) in enumerate(
            zip(fig.data[0].labels, fig.data[0].values, strict=True)
        )
        if fig.data[0].parents[idx] == "binary"
    }
    assert len(binary) == expected_systems

    if expected_other:
        other_label, other_count = expected_other
        other = next(label for label in binary if "Other" in label)
        assert other == other_label
        assert binary[other] == other_count
    else:
        assert not any("Other" in label for label in binary)


@pytest.mark.parametrize(
    ("max_slices_mode", "expected_systems"),
    [("other", 3), ("drop", 2)],
)
def test_chem_sys_sunburst_max_slices_mode(
    max_slices_mode: Literal["other", "drop"],
    expected_systems: int,
) -> None:
    """Test chem_sys_sunburst with different max_slices_mode values."""
    systems = ["Fe2O3", "Fe2O3", "Fe2O3", "Li2O", "Li2O", "Na2O"]
    fig = pmv.chem_sys_sunburst(systems, max_slices=2, max_slices_mode=max_slices_mode)

    binary = {
        fig.data[0].labels[idx]: fig.data[0].values[idx]  # noqa: PD011
        for idx in range(len(fig.data[0].labels))
        if fig.data[0].parents[idx] == "binary"
    }
    assert len(binary) == expected_systems
    assert binary["Fe-O"] == 3
    assert binary["Li-O"] == 2


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_chem_env_sunburst_show_counts(show_counts: ShowCounts) -> None:
    """Test chem_env_sunburst text formatting with representative processed data."""
    chem_env_data = [
        {"coord_num": 4, "chem_env_symbol": "T:4", "count": 2.0},
        {"coord_num": 6, "chem_env_symbol": "O:6", "count": 1.0},
    ]

    fig = _process_chem_env_data_sunburst(
        chem_env_data=chem_env_data,
        max_slices_cn=None,
        max_slices_ce=None,
        max_slices_mode="other",
        show_counts=show_counts,
    )

    assert len(fig.data) == 1
    assert fig.data[0].textinfo == "none"
    assert fig.data[0].marker.line.color == "white"

    text_template = fig.data[0].texttemplate
    assert (
        text_template
        == {
            "value": "%{label}: %{value:.2f}",
            "percent": "%{label}: %{percentParent:.1%}",
            "value+percent": "%{label}: %{value:.2f} (%{percentParent:.1%})",
            False: "%{label}",
        }[show_counts]
    )


@pytest.mark.parametrize(
    ("max_slices_mode", "expected_labels", "expected_parents", "expected_values"),
    [
        (
            "other",
            ["T:4", "Other CEs (2<br>more not shown)", "Other CNs", "4", "Other CNs"],
            ["4", "4", "Other CNs", "", ""],
            [5.0, 3.0, 4.0, 8.0, 4.0],
        ),
        ("drop", ["T:4", "4"], ["4", ""], [5.0, 5.0]),
    ],
)
def test_chem_env_sunburst_limits_cn_and_ce_slices(
    max_slices_mode: Literal["other", "drop"],
    expected_labels: list[str],
    expected_parents: list[str],
    expected_values: list[float],
) -> None:
    """CN/CE slice limits aggregate or drop lower-count environments."""
    chem_env_data = [
        {"coord_num": 4, "chem_env_symbol": "T:4", "count": 5},
        {"coord_num": 4, "chem_env_symbol": "C:4", "count": 2},
        {"coord_num": 4, "chem_env_symbol": "D:4", "count": 1},
        {"coord_num": 5, "chem_env_symbol": "A:5", "count": 4},
    ]

    fig = _process_chem_env_data_sunburst(
        chem_env_data=chem_env_data,
        max_slices_cn=1,
        max_slices_ce=1,
        max_slices_mode=max_slices_mode,
        show_counts="value",
    )
    trace = fig.data[0]

    assert list(trace.labels) == expected_labels
    assert list(trace.parents) == expected_parents
    assert [*map(float, trace.values)] == expected_values


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        *[
            (dict(show_counts=bad_val), rf"Invalid show_counts={bad_val!r}")
            for bad_val in ["invalid", "bad_value", 123, True]
        ],
        (
            dict(chem_env_settings="invalid_method"),
            "Invalid chem_env_settings='invalid_method', must be one of",
        ),
        # validated even when no slice limit is set (used to be silently ignored)
        (dict(max_slices_mode="keep"), "Invalid max_slices_mode='keep'"),
    ],
)
def test_chem_env_sunburst_invalid_options(
    fe_structure: Structure, kwargs: dict[str, Any], match: str
) -> None:
    """Invalid show_counts, chem_env_settings or max_slices_mode raise ValueError."""
    with pytest.raises(ValueError, match=match):
        pmv.chem_env_sunburst([fe_structure], **kwargs)


def test_chem_env_sunburst_no_data(fe_structure: Structure) -> None:
    """Per-structure ChemEnv failure yields an empty figure with a 'no data' title."""
    with patch(
        f"{CHEMENV_MODULE}.coordination_geometry_finder.LocalGeometryFinder"
    ) as mock_lgf:
        mock_lgf.return_value.compute_structure_environments.side_effect = RuntimeError(
            "ChemEnv failed"
        )
        with pytest.warns(UserWarning, match="ChemEnv analysis failed"):
            fig = pmv.chem_env_sunburst([fe_structure], chem_env_settings="chemenv")

    assert len(fig.data) == 0
    assert fig.layout.title.text == "No CN/CE data to display"


def test_chem_env_sunburst_crystal_nn_method(fe_structure: Structure) -> None:
    """Test chem_env_sunburst with CrystalNN method (default)."""
    fig = pmv.chem_env_sunburst([fe_structure], chem_env_settings="crystal_nn")
    assert len(fig.data) == 1


@pytest.mark.parametrize(
    ("coord_envs", "symbol_cn_mapping", "expected_labels"),
    [
        ([[{"ce_symbol": "T:4"}]], {"T:4": 4}, ["T:4", "4"]),
        (None, {}, None),  # no coordination environments
    ],
)
def test_chem_env_sunburst_chemenv_mocked(
    fe_structure: Structure,
    coord_envs: list[list[dict[str, str]]] | None,
    symbol_cn_mapping: dict[str, int],
    expected_labels: list[str] | None,
) -> None:
    """Mocked ChemEnv environments become sunburst slices, none gives empty fig."""
    with (
        patch(f"{CHEMENV_MODULE}.coordination_geometry_finder.LocalGeometryFinder"),
        patch(
            f"{CHEMENV_MODULE}.coordination_geometries.AllCoordinationGeometries"
        ) as mock_geoms,
        patch(
            f"{CHEMENV_MODULE}.structure_environments.LightStructureEnvironments"
            ".from_structure_environments",
            return_value=MagicMock(coordination_environments=coord_envs),
        ),
    ):
        mock_geoms.return_value.get_symbol_cn_mapping.return_value = symbol_cn_mapping
        fig = pmv.chem_env_sunburst([fe_structure], chem_env_settings="chemenv")

    if expected_labels is None:
        assert fig.layout.title.text == "No CN/CE data to display"
    else:
        assert list(fig.data[0].labels) == expected_labels


@pytest.mark.parametrize("label_col", [Key.formula, "coord_num"])
def test_limit_slices_edge_cases(label_col: str) -> None:
    """Label aggregate slices for formula and coordination-number frames."""
    df_test = pd.DataFrame({"coord_num": [4, 5, 6, 7], "count": [10, 5, 3, 1]})
    if label_col == Key.formula:
        df_test[Key.formula] = ["A1", "A2", "A3", "A4"]
    result = _limit_slices(
        df_test,
        group_col="coord_num",
        count_col="count",
        max_slices=2,
        max_slices_mode="other",
        other_label="Other CNs",
    )
    assert len(result) == 3
    assert result["count"].tolist() == [10, 5, 4]
    assert result.iloc[-1][label_col] == "Other CNs (2 more not shown)"
    assert df_test["coord_num"].tolist() == [4, 5, 6, 7]


@pytest.mark.parametrize(
    ("max_slices", "expected_len"),
    [(-5, 4), (0, 4), (10, 4)],  # Edge cases: negative, zero, too large
)
def test_limit_slices_boundary_conditions(max_slices: int, expected_len: int) -> None:
    """Test boundary conditions in _limit_slices function."""
    df_test = pd.DataFrame(
        [
            {"group": "A", "count": 10},
            {"group": "A", "count": 5},
            {"group": "A", "count": 3},
            {"group": "A", "count": 1},
        ]
    )

    result = _limit_slices(
        df_test,
        group_col="group",
        count_col="count",
        max_slices=max_slices,
        max_slices_mode="other",
    )

    assert len(result) == expected_len


def test_limit_slices_invalid_mode() -> None:
    """Test _limit_slices with invalid max_slices_mode."""
    df_test = pd.DataFrame([{"group": "A", "count": 1}])
    with pytest.raises(ValueError, match=r"Invalid.*max_slices_mode"):
        _limit_slices(
            df_test,
            group_col="group",
            count_col="count",
            max_slices=1,
            max_slices_mode="invalid",  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("child_col", "expected_empty_cols"),
    [
        (Key.chem_sys, [Key.formula, Key.spg_num]),
        (Key.formula, [Key.chem_sys, Key.spg_num]),
        (Key.spg_num, [Key.chem_sys, Key.formula]),
        ("group", [Key.chem_sys, Key.formula, Key.spg_num]),
    ],
)
def test_limit_slices_child_col_exclusion(
    child_col: str, expected_empty_cols: list[str]
) -> None:
    """Test when child_col_for_other_label is set, other columns get empty strings."""
    df_test = pd.DataFrame(
        [
            {
                "group": "A",
                "count": 10,
                Key.chem_sys: "Fe-O",
                Key.formula: "Fe2O3",
                Key.spg_num: "1",
            },
            {
                "group": "A",
                "count": 5,
                Key.chem_sys: "Li-O",
                Key.formula: "Li2O",
                Key.spg_num: "2",
            },
            {
                "group": "A",
                "count": 1,
                Key.chem_sys: "Na-O",
                Key.formula: "Na2O",
                Key.spg_num: "3",
            },
        ]
    )

    result = _limit_slices(
        df_test,
        group_col="group",
        count_col="count",
        max_slices=2,
        max_slices_mode="other",
        child_col_for_other_label=child_col,
    )

    other_rows = result[result["count"] == 1]
    assert len(other_rows) == 1
    other_row = other_rows.iloc[0]

    # The specified child_col should have the "Other" label
    assert "Other" in other_row[child_col]

    # Other legacy columns should be empty strings
    for col in expected_empty_cols:
        assert other_row[col] == ""


def test_sunburst_text_wrapping_functionality() -> None:
    """Test that long chemical environment names are wrapped with line breaks."""
    mock_data = [
        {"coord_num": 4, "chem_env_symbol": "rectangular see-saw-like:4", "count": 10},
        {"coord_num": 5, "chem_env_symbol": "short", "count": 5},
    ]
    fig = _process_chem_env_data_sunburst(
        chem_env_data=mock_data,
        max_slices_cn=None,
        max_slices_ce=None,
        max_slices_mode="other",
        show_counts="value",
    )
    assert list(fig.data[0].labels) == [
        "rectangular<br>see-saw-like:4",
        "short",
        "4",
        "5",
    ]


def test_chem_sys_sunburst_max_slices_per_arity() -> None:
    """max_slices limits each arity independently (regression: a global limit
    lumped whole arity levels into 'Other' under the wrong parent).
    """
    systems = ["Fe2O3", "Li2O", "Na2O", "K2O", "CaO", "LiFeO2", "NaFeO2", "KFeO2"]
    fig = pmv.chem_sys_sunburst(systems, max_slices=2, max_slices_mode="other")
    trace = fig.data[0]
    by_parent: dict[str, dict[str, float]] = {}  # trace["values"] avoids PD011
    for label, parent, val in zip(
        trace.labels, trace.parents, trace["values"], strict=True
    ):
        by_parent.setdefault(parent, {})[label] = val
    # both arities survive: 2 shown + 1 "Other" slice aggregating its own arity
    assert by_parent["binary"]["Other (3 more not shown)"] == 3
    assert by_parent["ternary"]["Other (1 more not shown)"] == 1
    assert len(by_parent["binary"]) == len(by_parent["ternary"]) == 3
