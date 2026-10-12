"""Test treemap plots for coordination numbers and chemical environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args
from unittest.mock import MagicMock, patch

import plotly.graph_objects as go
import pytest

from pymatviz.treemap.chem_env import (
    _process_chem_env_data_treemap,
    chem_env_treemap,
    default_cn_formatter,
)
from pymatviz.typing import ShowCounts


if TYPE_CHECKING:
    from collections.abc import Generator

    from pymatgen.core import Structure


@pytest.fixture
def mock_structure() -> MagicMock:
    """Create a mock structure for testing."""
    return MagicMock()


@pytest.fixture
def chemenv_mocks() -> Generator[dict[str, Any], None, None]:
    """Setup common ChemEnv mocks."""
    with (
        patch("pymatviz.chem_env.normalize_structures") as mock_norm,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder.LocalGeometryFinder"
        ) as mock_lgf,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometries.AllCoordinationGeometries"
        ) as mock_geoms,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.structure_environments.LightStructureEnvironments.from_structure_environments"
        ) as mock_lse_factory,
    ):
        # Setup basic mock structure
        mock_structure = MagicMock()
        mock_norm.return_value.values.return_value = [mock_structure]

        # Setup LocalGeometryFinder
        mock_lgf_instance = MagicMock()
        mock_lgf.return_value = mock_lgf_instance
        mock_structure_envs = MagicMock()
        mock_lgf_instance.compute_structure_environments.return_value = (
            mock_structure_envs
        )

        # Setup LightStructureEnvironments
        mock_lse = MagicMock()
        mock_lse_factory.return_value = mock_lse

        # Setup AllCoordinationGeometries
        mock_geoms_instance = MagicMock()
        mock_geoms.return_value = mock_geoms_instance

        yield {
            "norm": mock_norm,
            "lgf": mock_lgf,
            "lgf_instance": mock_lgf_instance,
            "lse": mock_lse,
            "lse_factory": mock_lse_factory,
            "geoms_instance": mock_geoms_instance,
            "structure": mock_structure,
        }


@pytest.mark.parametrize(
    ("coord_num", "count", "total", "expected"),
    [
        (4, 2.5, 10, "CN 4 (N=2.5, 25.0%)"),  # float >= 1: 2 decimals, trimmed
        (4, 0.123456, 10, "CN 4 (N=0.123, 1.2%)"),  # float < 1: 3 decimals
        (6, 1000, 5000, "CN 6 (N=1,000, 20.0%)"),  # int: thousands separator
    ],
)
def test_default_cn_formatter(
    coord_num: int, count: float, total: float, expected: str
) -> None:
    """Format integer, fractional, and non-integral coordination counts."""
    assert default_cn_formatter(coord_num, count, total) == expected


def test_chem_env_treemap_chemenv_setup_error_propagates(
    mock_structure: MagicMock,
) -> None:
    """ChemEnv setup failures raise instead of yielding an empty figure."""
    with (
        patch("pymatviz.chem_env.normalize_structures") as mock_norm,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder.LocalGeometryFinder"
        ) as mock_lgf,
    ):
        mock_norm.return_value.values.return_value = [mock_structure]
        mock_lgf.side_effect = ImportError("ChemEnv not available")

        with pytest.raises(ImportError, match="ChemEnv not available"):
            chem_env_treemap(mock_structure, chem_env_settings="chemenv")


@pytest.mark.parametrize(
    ("failing_target", "error"),
    [
        ("classify", ValueError("Order parameters not available")),
        ("get_nn_info", RuntimeError("CrystalNN analysis failed")),
    ],
)
def test_chem_env_treemap_crystal_nn_analysis_error(
    failing_target: str, error: Exception
) -> None:
    """Per-structure CrystalNN failures warn and skip the structure."""
    mock_structure = MagicMock()
    mock_structure.__len__.return_value = 1

    with (
        patch("pymatviz.chem_env.normalize_structures") as mock_norm,
        patch("pymatgen.analysis.local_env.CrystalNN") as mock_crystal_nn,
        patch(
            "pymatviz.chem_env.classify_local_env_with_order_params"
        ) as mock_classify,
    ):
        mock_norm.return_value.values.return_value = [mock_structure]
        mock_get_nn_info = mock_crystal_nn.return_value.get_nn_info
        mock_get_nn_info.return_value = [{"site": MagicMock()}]
        failing_mock = (
            mock_classify if failing_target == "classify" else mock_get_nn_info
        )
        failing_mock.side_effect = error

        with pytest.warns(
            UserWarning, match=f"CrystalNN analysis failed for structure 0.*{error}"
        ):
            fig = chem_env_treemap(mock_structure, chem_env_settings="crystal_nn")

    assert fig.layout.title.text == "No CN/CE data to display"


CN_DATA = [
    {"coord_num": 4, "chem_env_symbol": "T:4", "count": 10},
    {"coord_num": 6, "chem_env_symbol": "O:6", "count": 8},
    {"coord_num": 8, "chem_env_symbol": "C:8", "count": 6},
    {"coord_num": 3, "chem_env_symbol": "T:3", "count": 4},
    {"coord_num": 5, "chem_env_symbol": "SP:5", "count": 2},
]
CE_DATA = [
    {"coord_num": 4, "chem_env_symbol": "T:4", "count": 10},
    {"coord_num": 4, "chem_env_symbol": "SP:4", "count": 8},
    {"coord_num": 4, "chem_env_symbol": "SS:4", "count": 6},
    {"coord_num": 4, "chem_env_symbol": "SQ:4", "count": 4},
    {"coord_num": 6, "chem_env_symbol": "O:6", "count": 12},
]


@pytest.mark.parametrize(
    ("chem_env_data", "max_cells_cn", "max_cells_ce", "expected_labels"),
    [
        (
            CN_DATA,
            3,
            None,
            [
                "T:4",
                "O:6",
                "C:8",
                "Other CNs (2<br>more not shown)",
                "CN 4",
                "CN 6",
                "CN 8",
                "Other CNs (2 more not shown)",
            ],
        ),
        (
            CE_DATA,
            None,
            2,
            ["T:4", "SP:4", "Other CEs (2<br>more not shown)", "O:6", "CN 4", "CN 6"],
        ),
        (CN_DATA[:2], 10, None, ["T:4", "O:6", "CN 4", "CN 6"]),  # limit > n_cns
        (CN_DATA[:2], None, 10, ["T:4", "O:6", "CN 4", "CN 6"]),  # limit > n_ces
        (  # long CE names are wrapped with <br>, short ones untouched
            [
                {
                    "coord_num": 4,
                    "chem_env_symbol": "rectangular see-saw-like:4",
                    "count": 10,
                },
                {"coord_num": 5, "chem_env_symbol": "short", "count": 5},
            ],
            None,
            None,
            ["rectangular<br>see-saw-like:4", "short", "CN 4", "CN 5"],
        ),
    ],
    ids=["cn_limit", "ce_limit", "cn_limit_too_large", "ce_limit_too_large", "wrap"],
)
def test_process_chem_env_data_treemap_labels(
    chem_env_data: list[dict[str, Any]],
    max_cells_cn: int | None,
    max_cells_ce: int | None,
    expected_labels: list[str],
) -> None:
    """CN/CE cell limits lump excess cells into 'Other' entries; long names wrap."""
    fig = _process_chem_env_data_treemap(
        chem_env_data=chem_env_data,
        max_cells_cn=max_cells_cn,
        max_cells_ce=max_cells_ce,
        show_counts="value",
        cn_formatter=False,
    )
    assert list(fig.data[0].labels) == expected_labels


def test_chem_env_treemap_no_data() -> None:
    """Test chem_env_treemap with no data returns empty figure."""
    with patch("pymatviz.chem_env.normalize_structures") as mock_norm:
        mock_norm.return_value.values.return_value = []
        fig = chem_env_treemap([])
        assert isinstance(fig, go.Figure)
        assert "No CN/CE data to display" in fig.layout.title.text


@pytest.mark.parametrize(
    ("chem_env_settings", "coordination_envs", "ce_mapping", "expected_valid"),
    [
        # Valid ChemEnv cases
        ("chemenv", [[{"ce_symbol": "T:4"}]], {"T:4": 4}, True),
        (
            "chemenv",
            [[{"ce_symbol": "T:4"}, {"ce_symbol": "O:6"}]],
            {"T:4": 4, "O:6": 6},
            True,
        ),
        # Empty coordination environments
        ("chemenv", None, {}, False),
        ("chemenv", [[]], {}, False),
        # custom settings dicts are forwarded to LocalGeometryFinder
        ({"custom": "setting"}, [[{"ce_symbol": "T:4"}]], {"T:4": 4}, True),
    ],
)
def test_chem_env_treemap_chemenv_scenarios(
    chemenv_mocks: dict[str, Any],
    chem_env_settings: Any,
    coordination_envs: list[list[dict[str, Any]]] | None,
    ce_mapping: dict[str, int],
    expected_valid: bool | str,
) -> None:
    """Test various ChemEnv scenarios."""
    mocks = chemenv_mocks
    mocks["lse"].coordination_environments = coordination_envs
    mocks["geoms_instance"].get_symbol_cn_mapping.return_value = ce_mapping

    fig = chem_env_treemap(mocks["structure"], chem_env_settings=chem_env_settings)
    assert isinstance(fig, go.Figure)
    expected_settings = {} if chem_env_settings == "chemenv" else chem_env_settings
    mocks["lgf_instance"].setup_parameters.assert_called_once_with(**expected_settings)

    if expected_valid:
        assert len(fig.data) == 1
        assert fig.data[0].type == "treemap"
        assert hasattr(fig.data[0], "labels")
    else:
        assert "No CN/CE data to display" in fig.layout.title.text


@pytest.mark.parametrize(
    ("max_cells_cn", "max_cells_ce"),
    [
        (None, None),  # No limits
        (0, 0),  # Zero treated as None
        (-1, -1),  # Negative treated as None
        (1, None),  # CN limit only
        (None, 1),  # CE limit only
        (1, 1),  # Both limits
        (100, 100),  # Limits larger than data
    ],
)
def test_chem_env_treemap_max_cells(
    structures: tuple[Structure, Structure],
    max_cells_cn: int | None,
    max_cells_ce: int | None,
) -> None:
    """Test max_cells parameters."""
    fig = chem_env_treemap(
        [structures[0]],
        chem_env_settings="crystal_nn",
        max_cells_cn=max_cells_cn,
        max_cells_ce=max_cells_ce,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_chem_env_treemap_show_counts(
    chemenv_mocks: dict[str, Any],
    show_counts: ShowCounts,
) -> None:
    """Test different show_counts options.

    Includes regression test for show_counts=False to ensure CN labels
    don't contain count/percentage information when user opts out.
    """
    mocks = chemenv_mocks
    mocks["lse"].coordination_environments = [[{"ce_symbol": "T:4"}]]
    mocks["geoms_instance"].get_symbol_cn_mapping.return_value = {"T:4": 4}

    fig = chem_env_treemap(
        mocks["structure"], show_counts=show_counts, chem_env_settings="chemenv"
    )
    assert len(fig.data) == 1
    assert fig.layout.paper_bgcolor == "rgba(0, 0, 0, 0)"
    assert "values" in fig.data[0]
    assert "labels" in fig.data[0]

    # show_counts must control the cell text template (regression: it used to
    # have no effect on cell text at all)
    assert fig.data[0].texttemplate == {
        "value": "%{label}: %{value:.2f}",
        "percent": "%{label}: %{percentParent:.1%}",
        "value+percent": "%{label}: %{value:.2f} (%{percentParent:.1%})",
        False: "%{label}",
    }.get(show_counts)

    # CN labels only carry counts/percentages unless show_counts=False
    cn_label = next(label for label in fig.data[0].labels if label.startswith("CN"))
    assert cn_label == ("CN 4" if show_counts is False else "CN 4 (N=1, 100.0%)")


@pytest.mark.parametrize(
    ("normalize", "cn_formatter", "expected_label_content"),
    [
        (False, None, None),  # Default behavior
        (True, None, None),  # With normalization
        (
            False,
            lambda cn, count, total: f"Custom CN {cn}: {count}/{total}",
            "Custom CN",
        ),
        (False, False, "CN 4"),  # cn_formatter=False
    ],
)
def test_chem_env_treemap_formatting_options(
    chemenv_mocks: dict[str, Any],
    normalize: bool,
    cn_formatter: Any,
    expected_label_content: str | None,
) -> None:
    """Test various formatting options."""
    mocks = chemenv_mocks
    mocks["lse"].coordination_environments = [[{"ce_symbol": "T:4"}]]
    mocks["geoms_instance"].get_symbol_cn_mapping.return_value = {"T:4": 4}

    fig = chem_env_treemap(
        mocks["structure"],
        normalize=normalize,
        cn_formatter=cn_formatter,
        chem_env_settings="chemenv",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    if expected_label_content:
        labels = fig.data[0].labels
        assert any(expected_label_content in str(label) for label in labels)

        if cn_formatter is False:
            # Should not contain percentage or count info when cn_formatter=False
            assert not any("%" in str(label) for label in labels)
            assert not any("N=" in str(label) for label in labels)


def test_chem_env_treemap_failure_handling(chemenv_mocks: dict[str, Any]) -> None:
    """Per-structure ChemEnv failures warn and yield an empty figure."""
    mocks = chemenv_mocks
    mocks["lgf_instance"].setup_structure.side_effect = RuntimeError("ChemEnv failed")

    with pytest.warns(UserWarning, match="ChemEnv analysis failed for structure 0"):
        fig = chem_env_treemap(mocks["structure"], chem_env_settings="chemenv")
    assert isinstance(fig, go.Figure)
    assert "No CN/CE data to display" in fig.layout.title.text


def test_chem_env_treemap_crystal_nn_alias(mock_structure: MagicMock) -> None:
    """Test that 'crystal_nn' uses CrystalNN method."""
    mock_structure.__len__.return_value = 1

    with (
        patch("pymatviz.chem_env.normalize_structures") as mock_norm,
        patch("pymatgen.analysis.local_env.CrystalNN") as mock_crystal_nn,
        patch(
            "pymatviz.chem_env.classify_local_env_with_order_params"
        ) as mock_classify,
    ):
        mock_norm.return_value.values.return_value = [mock_structure]
        mock_crystal_nn_instance = MagicMock()
        mock_crystal_nn.return_value = mock_crystal_nn_instance
        mock_neighbor = MagicMock()
        mock_crystal_nn_instance.get_nn_info.return_value = [{"site": mock_neighbor}]
        mock_classify.return_value = "CN:1"

        fig = chem_env_treemap(mock_structure, chem_env_settings="crystal_nn")
        assert isinstance(fig, go.Figure)
        mock_crystal_nn.assert_called_once()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(show_counts="invalid"), r"Invalid show_counts='invalid', must be one of"),
        (
            dict(chem_env_settings="invalid_method"),
            r"Invalid chem_env_settings='invalid_method', must be one of",
        ),
    ],
)
def test_chem_env_treemap_invalid_options(
    mock_structure: MagicMock, kwargs: dict[str, Any], match: str
) -> None:
    """Invalid show_counts or chem_env_settings strings raise ValueError."""
    with pytest.raises(ValueError, match=match):
        chem_env_treemap(mock_structure, **kwargs)
