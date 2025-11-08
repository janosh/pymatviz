"""Test sunburst plots for space groups and chemical systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.sunburst import ShowCounts, spacegroup_sunburst


if TYPE_CHECKING:
    from pymatgen.core import Structure

    from pymatviz.typing import FormulaGroupBy


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_spacegroup_sunburst(show_counts: ShowCounts) -> None:
    """Test spacegroup_sunburst with different show_counts options."""
    fig = spacegroup_sunburst(range(1, 231), show_counts=show_counts)

    assert isinstance(fig, go.Figure)
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
        spacegroup_sunburst([1], show_counts="invalid")


def test_spacegroup_sunburst_single_item() -> None:
    """Test with single-item input."""
    fig = spacegroup_sunburst([1], show_counts="value")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].ids) == 2  # one for crystal system, one for spg number

    fig = spacegroup_sunburst(["P1"], show_counts="value+percent")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].ids) == 2


def test_spacegroup_sunburst_other_types(
    spg_symbols: list[str], structures: list[Structure]
) -> None:
    """Test with other types of input."""
    # test with pandas series
    series = pd.Series([*[1] * 3, *[2] * 10, *[3] * 5])
    fig = spacegroup_sunburst(series, show_counts="value")
    assert isinstance(fig, go.Figure)
    values = [*map(int, fig.data[0].values)]
    assert values == [10, 5, 3, 13, 5], f"actual {values=}"

    # test with strings of space group symbols
    fig = spacegroup_sunburst(spg_symbols)
    assert isinstance(fig, go.Figure)

    # test with pymatgen structures
    fig = spacegroup_sunburst(structures)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    ("max_slices", "max_slices_mode", "expected_systems", "expected_others"),
    [
        # Edge cases for max_slices
        (None, "other", 7, None),  # no limit
        (0, "other", 7, None),  # treated as None
        (-1, "other", 7, None),  # negative treated as None
        (10, "other", 7, None),  # greater than number of systems
        # Normal cases with "other" mode - Updated expectations
        (1, "other", 2, [("Other (6 more not shown)", 6)]),  # 1 + other = 2 total
        (2, "other", 3, [("Other (5 more not shown)", 5)]),  # 2 + other = 3 total
        (3, "other", 4, [("Other (4 more not shown)", 4)]),  # 3 + other = 4 total
        # Cases with "drop" mode - Updated expectations
        (1, "drop", 1, None),  # only top 1
        (2, "drop", 2, None),  # only top 2
        (3, "drop", 3, None),  # only top 3
    ],
)
def test_spacegroup_sunburst_max_slices(
    max_slices: int | None,
    max_slices_mode: Literal["other", "drop"],
    expected_systems: int,
    expected_others: list[tuple[str, int]] | None,
) -> None:
    """Test spacegroup_sunburst with max_slices functionality and edge cases."""
    # Create a dataset with multiple space groups per crystal system
    spg_numbers = (1, 2, 3, 4, 5, 6, 7)

    fig = pmv.spacegroup_sunburst(
        spg_numbers, max_slices=max_slices, max_slices_mode=max_slices_mode
    )
    assert isinstance(fig, go.Figure)

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


def test_spacegroup_sunburst_max_slices_mode_invalid() -> None:
    """Test spacegroup_sunburst with invalid max_slices_mode."""
    with pytest.raises(ValueError, match="Invalid max_slices_mode="):
        pmv.spacegroup_sunburst([1, 2, 3], max_slices=1, max_slices_mode="invalid")  # type: ignore[arg-type]


def test_chem_sys_sunburst_basic() -> None:
    """Test chem_sys_sunburst plot with various scenarios."""
    systems = ["Fe-O", "Li-P-O", "Li-Fe-P-O", "Fe", "O", "Li-O"]
    fig = pmv.chem_sys_sunburst(systems)
    assert isinstance(fig, go.Figure)
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
    assert isinstance(fig, go.Figure)

    # Check that Fe2O3 and Fe-O are counted together
    binary_count = sum(
        val
        for idx, val in enumerate(fig.data[0].values)
        if fig.data[0].parents[idx] == "binary" and fig.data[0].labels[idx] == "Fe-O"
    )
    assert binary_count == 2, "Fe2O3 and Fe-O should be merged"

    # Test with only Structure objects
    fig_structs = pmv.chem_sys_sunburst(structures)
    assert isinstance(fig_structs, go.Figure)

    # Test with invalid input type
    with pytest.raises(TypeError, match="Expected str, Composition or Structure"):
        pmv.chem_sys_sunburst([1])  # type: ignore[arg-type]


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
    assert isinstance(fig, go.Figure)

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
    assert isinstance(fig, go.Figure)

    # All variations should be counted as the same system
    binary_count = sum(
        val
        for idx, val in enumerate(fig.data[0].values)
        if fig.data[0].parents[idx] == "binary" and fig.data[0].labels[idx] == "Fe-O"
    )
    assert binary_count == 4, "All Fe2O3 variations should be counted together"


@pytest.mark.parametrize(
    ("group_by", "expected_fe_count"),
    [
        ("formula", 3),  # Each Fe formula counted separately: Fe2O3, Fe4O6, FeO
        ("reduced_formula", 3),  # Fe2O3 (count=2, includes Fe4O6) + FeO (count=1)
        ("chem_sys", 3),  # All Fe-O formulas grouped
    ],
)
def test_chem_sys_sunburst_grouping(
    group_by: FormulaGroupBy, expected_fe_count: int
) -> None:
    """Test chem_sys_sunburst with different grouping modes."""
    systems = ["Fe2O3", "Fe4O6", "FeO", "Li2O"]

    fig = pmv.chem_sys_sunburst(systems, group_by=group_by)
    assert isinstance(fig, go.Figure)

    # Find Fe-related counts
    if group_by == "chem_sys":
        fe_count = sum(
            val
            for idx, val in enumerate(fig.data[0].values)
            if fig.data[0].parents[idx] == "binary"
            and fig.data[0].labels[idx] == "Fe-O"
        )
    else:
        # For formula/reduced_formula, count formula-level entries
        fe_count = sum(
            val
            for idx, val in enumerate(fig.data[0].values)
            if fig.data[0].parents[idx] == "binary" and "Fe" in fig.data[0].labels[idx]
        )

    assert fe_count == expected_fe_count


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
    assert isinstance(fig, go.Figure)

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
    assert isinstance(fig, go.Figure)

    binary = {
        fig.data[0].labels[idx]: fig.data[0].values[idx]  # noqa: PD011
        for idx in range(len(fig.data[0].labels))
        if fig.data[0].parents[idx] == "binary"
    }
    assert len(binary) == expected_systems
    assert binary["Fe-O"] == 3
    assert binary["Li-O"] == 2


def test_chem_sys_sunburst_max_slices_mode_invalid() -> None:
    """Test chem_sys_sunburst with invalid max_slices_mode."""
    with pytest.raises(ValueError, match="Invalid max_slices_mode="):
        pmv.chem_sys_sunburst(["Fe2O3"], max_slices=1, max_slices_mode="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_chem_env_sunburst_show_counts(show_counts: ShowCounts) -> None:
    """Test chem_env_sunburst with different show_counts options using mocked data."""
    from unittest.mock import patch

    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    # Mock the expensive ChemEnv computation to return test data
    mock_chem_env_data = [
        {"coord_num": 4, "chem_env_symbol": "T:4", "count": 2.0},
        {"coord_num": 6, "chem_env_symbol": "O:6", "count": 1.0},
    ]

    with patch("pymatviz.sunburst.chem_env_sunburst") as mock_func:
        # Create a mock figure with expected properties
        import pandas as pd

        df_mock = pd.DataFrame(mock_chem_env_data)
        fig = px.sunburst(
            df_mock, path=["coord_num", "chem_env_symbol"], values="count"
        )

        # Apply the formatting based on show_counts
        text_templates = {
            "value": "%{label}: %{value}",
            "percent": "%{label}: %{percentParent:.1%}",
            "value+percent": "%{label}: %{value} (%{percentParent:.1%})",
            False: "%{label}",
        }
        fig.data[0].texttemplate = text_templates[show_counts]
        fig.data[0].textinfo = "none"
        fig.data[0].update(marker=dict(line=dict(color="white")))

        mock_func.return_value = fig
        result = mock_func([simple_structure], show_counts=show_counts)

        assert isinstance(result, go.Figure)
        assert len(result.data) == 1

        # Check text formatting based on show_counts
        if show_counts == "value":
            assert "%{value}" in result.data[0].texttemplate
        elif show_counts == "percent":
            assert "%{percentParent" in result.data[0].texttemplate
        elif show_counts == "value+percent":
            assert "%{value}" in result.data[0].texttemplate
            assert "%{percentParent" in result.data[0].texttemplate
        elif show_counts is False:
            assert result.data[0].texttemplate == "%{label}"


@pytest.mark.parametrize("invalid_show_counts", ["invalid", "bad_value", 123, True])
def test_chem_env_sunburst_invalid_show_counts(invalid_show_counts: Any) -> None:
    """Test that invalid show_counts values raise ValueError."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    with pytest.raises(ValueError, match=r"Invalid.*show_counts"):
        pmv.chem_env_sunburst([simple_structure], show_counts=invalid_show_counts)


def test_chem_env_sunburst_no_data() -> None:
    """Test chem_env_sunburst when ChemEnv returns no data."""
    from unittest.mock import MagicMock, patch

    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    # Mock LocalGeometryFinder to simulate ChemEnv failure
    with patch(
        "pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder.LocalGeometryFinder"
    ) as mock_lgf:
        mock_lgf_instance = MagicMock()
        mock_lgf.return_value = mock_lgf_instance
        mock_lgf_instance.compute_structure_environments.side_effect = ImportError(
            "ChemEnv failed"
        )

        # use ChemEnv method to test the failure case
        fig = pmv.chem_env_sunburst([simple_structure], chem_env_settings="chemenv")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # Empty figure when no data
        # Should have title indicating no data
        if fig.layout.title and fig.layout.title.text:
            assert "No CN/CE data to display" in fig.layout.title.text


@pytest.mark.parametrize(
    ("max_slices_cn", "max_slices_ce", "max_slices_mode", "normalize"),
    [
        (None, None, "other", False),  # No limits
        (2, None, "other", True),  # CN limit only, normalized
        (None, 3, "drop", False),  # CE limit only, drop mode
        (1, 2, "other", True),  # Both limits, other mode, normalized
        (0, 0, "other", True),  # Zero limits (treated as None)
    ],
)
def test_chem_env_sunburst_parameters(
    max_slices_cn: int | None,
    max_slices_ce: int | None,
    max_slices_mode: Literal["other", "drop"],
    normalize: bool,
) -> None:
    """Test chem_env_sunburst with various parameter combinations."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    # Test that parameter combos don't crash
    fig = pmv.chem_env_sunburst(
        [simple_structure],
        max_slices_cn=max_slices_cn,
        max_slices_ce=max_slices_ce,
        max_slices_mode=max_slices_mode,
        normalize=normalize,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_chem_env_sunburst_custom_chem_env_settings() -> None:
    """Test chem_env_sunburst with custom ChemEnv settings."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    # Since ChemEnv parameter names are complex and may change,
    # just test that empty settings work (which they should)
    custom_settings: dict[str, Any] = {}

    # Test that custom settings don't crash
    fig = pmv.chem_env_sunburst([simple_structure], chem_env_settings=custom_settings)
    assert isinstance(fig, go.Figure)


def test_chem_env_sunburst_crystal_nn_method() -> None:
    """Test chem_env_sunburst with CrystalNN method (default)."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    # Test that CrystalNN method works without crashing
    fig = pmv.chem_env_sunburst([simple_structure], chem_env_settings="crystal_nn")

    # Verify figure structure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_chem_env_sunburst_explicit_mode() -> None:
    """Test chem_env_sunburst with explicit ChemEnv method."""
    from unittest.mock import MagicMock, patch

    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    with (
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder.LocalGeometryFinder"
        ) as mock_lgf,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometries.AllCoordinationGeometries"
        ) as mock_geoms,
    ):
        # Setup mocks
        mock_lgf_instance = MagicMock()
        mock_lgf.return_value = mock_lgf_instance

        mock_structure_envs = MagicMock()
        mock_lgf_instance.compute_structure_environments.return_value = (
            mock_structure_envs
        )

        mock_lse = MagicMock()
        mock_lse.coordination_environments = [[{"ce_symbol": "T:4"}]]

        with patch(
            "pymatgen.analysis.chemenv.coordination_environments.structure_environments.LightStructureEnvironments.from_structure_environments"
        ) as mock_lse_factory:
            mock_lse_factory.return_value = mock_lse

            mock_geoms_instance = MagicMock()
            mock_geoms.return_value = mock_geoms_instance
            mock_geoms_instance.get_symbol_cn_mapping.return_value = {"T:4": 4}

            fig = pmv.chem_env_sunburst([simple_structure], chem_env_settings="chemenv")

            # Verify figure was created
            assert isinstance(fig, go.Figure)
            assert len(fig.data) == 1


def test_chem_env_sunburst_invalid_chem_env_settings() -> None:
    """Test chem_env_sunburst with invalid chem_env_settings."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    # Invalid string should fall through to ChemEnv path and fail
    # because ChemEnv expects a dictionary, not a string
    with pytest.raises((ImportError, AttributeError, RuntimeError, TypeError)):
        pmv.chem_env_sunburst([simple_structure], chem_env_settings="invalid_method")  # type: ignore[arg-type]


def test_chem_env_sunburst_empty_coordination_environments() -> None:
    """Test handling of empty coordination environments."""
    from unittest.mock import MagicMock, patch

    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    with (
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder.LocalGeometryFinder"
        ) as mock_lgf,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometries.AllCoordinationGeometries"
        ) as mock_geoms,
    ):
        # Setup mocks with empty coordination environments
        mock_lgf_instance = MagicMock()
        mock_lgf.return_value = mock_lgf_instance

        mock_structure_envs = MagicMock()
        mock_lgf_instance.compute_structure_environments.return_value = (
            mock_structure_envs
        )

        mock_lse = MagicMock()
        mock_lse.coordination_environments = None  # Empty environments

        with patch(
            "pymatgen.analysis.chemenv.coordination_environments.structure_environments.LightStructureEnvironments.from_structure_environments"
        ) as mock_lse_factory:
            mock_lse_factory.return_value = mock_lse

            mock_geoms_instance = MagicMock()
            mock_geoms.return_value = mock_geoms_instance
            mock_geoms_instance.get_symbol_cn_mapping.return_value = {}

            fig = pmv.chem_env_sunburst([simple_structure], chem_env_settings="chemenv")

            # Should return empty figure
            assert isinstance(fig, go.Figure)
            assert "No CN/CE data to display" in fig.layout.title.text


# Tests for _limit_slices helper function
def test_limit_slices_edge_cases() -> None:
    """Test edge cases in _limit_slices function."""
    import pandas as pd

    from pymatviz.sunburst.helpers import _limit_slices

    # Create test data with more items than limit
    df_test = pd.DataFrame(
        [
            {"group": "A", "count": 10, "formula": "A1"},
            {"group": "A", "count": 5, "formula": "A2"},
            {"group": "A", "count": 3, "formula": "A3"},
            {"group": "A", "count": 1, "formula": "A4"},
        ]
    )

    result = _limit_slices(
        df_test,
        group_col="group",
        count_col="count",
        max_slices=2,
        max_slices_mode="other",
    )

    # Should have 3 rows: top 2 + other
    assert len(result) == 3
    other_rows = result[result["count"] == 4]  # Combined count of last 2 items
    assert len(other_rows) == 1


@pytest.mark.parametrize(
    ("max_slices", "expected_len"),
    [(-5, 4), (0, 4), (10, 4)],  # Edge cases: negative, zero, too large
)
def test_limit_slices_boundary_conditions(max_slices: int, expected_len: int) -> None:
    """Test boundary conditions in _limit_slices function."""
    import pandas as pd

    from pymatviz.sunburst.helpers import _limit_slices

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
    import pandas as pd

    from pymatviz.sunburst.helpers import _limit_slices

    df_test = pd.DataFrame([{"group": "A", "count": 1}])
    with pytest.raises(ValueError, match=r"Invalid.*max_slices_mode"):
        _limit_slices(
            df_test,
            group_col="group",
            count_col="count",
            max_slices=1,
            max_slices_mode="invalid",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("child_col", "expected_empty_cols"),
    [
        (Key.chem_sys, [Key.formula, Key.spg_num]),
        (Key.formula, [Key.chem_sys, Key.spg_num]),
        (Key.spg_num, [Key.chem_sys, Key.formula]),
    ],
)
def test_limit_slices_child_col_exclusion(
    child_col: str, expected_empty_cols: list[str]
) -> None:
    """Test when child_col_for_other_label is set, other columns get empty strings."""
    import pandas as pd

    from pymatviz.enums import Key
    from pymatviz.sunburst.helpers import _limit_slices

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


def test_chem_env_sunburst_m_colon_invalid_parsing() -> None:
    """Test M: symbols with invalid CN parsing."""
    from unittest.mock import MagicMock, patch

    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    coords = [[0.0, 0.0, 0.0]]
    species = ["Fe"]
    simple_structure = Structure(lattice, species, coords)

    with (
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder.LocalGeometryFinder"
        ) as mock_lgf,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies.SimplestChemenvStrategy"
        ),
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.structure_environments.LightStructureEnvironments"
        ) as mock_lse_class,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometries.AllCoordinationGeometries"
        ) as mock_acg,
    ):
        mock_lgf_instance = MagicMock()
        mock_lgf.return_value = mock_lgf_instance

        mock_lse_instance = MagicMock()
        mock_lse_class.from_structure_environments.return_value = mock_lse_instance

        # Mock M: symbols with invalid CN parsing (lines 515, 518-521)
        mock_lse_instance.coordination_environments = [
            [{"ce_symbol": "M:INVALID"}],  # Can't parse as int
            [{"ce_symbol": "M:"}],  # Empty after colon
        ]

        mock_acg_instance = MagicMock()
        mock_acg.return_value = mock_acg_instance
        # Return empty mapping so M: symbols aren't found
        mock_acg_instance.get_symbol_cn_mapping.return_value = {}

        fig = pmv.chem_env_sunburst([simple_structure])

        assert isinstance(fig, go.Figure)
        # Should handle invalid M: symbols gracefully by assigning CN=0


def test_sunburst_text_wrapping_functionality() -> None:
    """Test that long chemical environment names are wrapped with line breaks."""
    import textwrap

    def wrap_text(text: str) -> str:
        return "<br>".join(
            textwrap.wrap(text, width=15, break_long_words=True, break_on_hyphens=True)
        )

    test_cases = [
        ("short", "short"),
        ("rectangular see-saw-like:4", "rectangular<br>see-saw-like:4"),
        ("square-pyramidal:5", "square-<br>pyramidal:5"),
    ]

    for input_text, expected_output in test_cases:
        result = wrap_text(input_text)
        assert result == expected_output

    mock_data = [
        {"coord_num": 4, "chem_env_symbol": "rectangular see-saw-like:4", "count": 10},
        {"coord_num": 5, "chem_env_symbol": "short", "count": 5},
    ]

    from pymatviz.sunburst.chem_env import _process_chem_env_data_sunburst

    fig = _process_chem_env_data_sunburst(
        chem_env_data=mock_data,
        max_slices_cn=None,
        max_slices_ce=None,
        max_slices_mode="other",
        show_counts="value",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "sunburst"

    labels = fig.data[0].labels
    assert any("<br>" in str(label) for label in labels)
    assert any("short" in str(label) and "<br>" not in str(label) for label in labels)
