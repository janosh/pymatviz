"""Test treemap plots for coordination numbers and chemical environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args
from unittest.mock import MagicMock, patch

import plotly.graph_objects as go
import pytest

from pymatviz.treemap.chem_env import chem_env_treemap, default_cn_formatter
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
        patch("pymatviz.treemap.chem_env.normalize_structures") as mock_norm,
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


def test_default_cn_formatter_edge_cases() -> None:
    """Test default_cn_formatter with edge cases to improve coverage."""
    # Test with float count >= 1 (line 19-22 coverage)
    result = default_cn_formatter(4, 2.5, 10)
    assert "2.5" in result
    assert "25.0%" in result

    # Test with float count < 1
    result = default_cn_formatter(4, 0.123456, 10)
    assert "0.123" in result
    assert "1.2%" in result

    # Test with integer count
    result = default_cn_formatter(6, 1000, 5000)
    assert "1,000" in result
    assert "20.0%" in result


def test_chem_env_treemap_chemenv_import_error(mock_structure: MagicMock) -> None:
    """Test ChemEnv method handling ImportError during setup."""
    with (
        patch("pymatviz.treemap.chem_env.normalize_structures") as mock_norm,
        patch(
            "pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder.LocalGeometryFinder"
        ) as mock_lgf,
    ):
        mock_norm.return_value.values.return_value = [mock_structure]
        mock_lgf.side_effect = ImportError("ChemEnv not available")  # Line 225 coverage

        fig = chem_env_treemap(mock_structure, chem_env_settings="chemenv")
        assert isinstance(fig, go.Figure)
        assert "No CN/CE data to display" in fig.layout.title.text


def test_chem_env_treemap_crystal_nn_analysis_error() -> None:
    """Test CrystalNN method handling analysis errors during processing."""
    mock_structure = MagicMock()
    mock_structure.__len__.return_value = 1
    mock_structure.__iter__.return_value = iter([MagicMock()])

    with (
        patch("pymatviz.treemap.chem_env.normalize_structures") as mock_norm,
        patch("pymatgen.analysis.local_env.CrystalNN") as mock_crystal_nn,
        patch(
            "pymatviz.chem_env.classify_local_env_with_order_params"
        ) as mock_classify,
    ):
        mock_norm.return_value.values.return_value = [mock_structure]
        mock_crystal_nn_instance = MagicMock()
        mock_crystal_nn.return_value = mock_crystal_nn_instance

        # Make classify_local_env_with_order_params raise ImportError
        mock_classify.side_effect = ImportError("Order parameters not available")
        mock_crystal_nn_instance.get_nn_info.return_value = [{"site": MagicMock()}]

        # The function should catch the ImportError and return an empty figure
        with pytest.warns(UserWarning, match="CrystalNN analysis failed"):
            fig = chem_env_treemap(mock_structure, chem_env_settings="crystal_nn")
            assert isinstance(fig, go.Figure)
            assert "No CN/CE data to display" in fig.layout.title.text


def test_chem_env_treemap_crystal_nn_outer_try_catch() -> None:
    """Test CrystalNN method handling exceptions in the outer try-except."""
    mock_structure = MagicMock()
    mock_structure.__len__.return_value = 1

    with (
        patch("pymatviz.treemap.chem_env.normalize_structures") as mock_norm,
        patch("pymatgen.analysis.local_env.CrystalNN") as mock_crystal_nn,
    ):
        mock_norm.return_value.values.return_value = [mock_structure]
        mock_crystal_nn_instance = MagicMock()
        mock_crystal_nn.return_value = mock_crystal_nn_instance

        # Make get_nn_info raise RuntimeError (inner try-catch coverage)
        mock_crystal_nn_instance.get_nn_info.side_effect = RuntimeError(
            "CrystalNN analysis failed"
        )

        # The function should catch the RuntimeError and return an empty figure
        with pytest.warns(UserWarning, match="CrystalNN analysis failed for structure"):
            fig = chem_env_treemap(mock_structure, chem_env_settings="crystal_nn")
            assert isinstance(fig, go.Figure)
            assert fig.layout.title is not None
            assert "No CN/CE data to display" in fig.layout.title.text


def test_chem_env_treemap_max_cells_cn_logic() -> None:
    """Test max_cells_cn logic with edge cases."""
    # Create mock data with multiple CNs
    mock_data = [
        {"coord_num": 4, "chem_env_symbol": "T:4", "count": 10},
        {"coord_num": 6, "chem_env_symbol": "O:6", "count": 8},
        {"coord_num": 8, "chem_env_symbol": "C:8", "count": 6},
        {"coord_num": 3, "chem_env_symbol": "T:3", "count": 4},
        {"coord_num": 5, "chem_env_symbol": "SP:5", "count": 2},
    ]

    # Import the function directly to test it
    from pymatviz.treemap.chem_env import _process_chem_env_data_treemap

    # Test max_cells_cn with "Other CNs" creation (lines 350-376)
    fig = _process_chem_env_data_treemap(
        chem_env_data=mock_data,
        max_cells_cn=3,  # Should create "Other CNs" entry
        max_cells_ce=None,
        show_counts="value",
        cn_formatter=None,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    # Check that we have "Other CNs" entry
    labels = fig.data[0].labels
    assert any("Other CNs" in str(label) for label in labels)
    assert any("more not shown" in str(label) for label in labels)


def test_chem_env_treemap_max_cells_ce_logic() -> None:
    """Test max_cells_ce logic with edge cases."""
    # Create mock data with multiple CEs per CN
    mock_data = [
        {"coord_num": 4, "chem_env_symbol": "T:4", "count": 10},
        {"coord_num": 4, "chem_env_symbol": "SP:4", "count": 8},
        {"coord_num": 4, "chem_env_symbol": "SS:4", "count": 6},
        {"coord_num": 4, "chem_env_symbol": "SQ:4", "count": 4},
        {"coord_num": 6, "chem_env_symbol": "O:6", "count": 12},
    ]

    # Import the function directly to test it
    from pymatviz.treemap.chem_env import _process_chem_env_data_treemap

    # Test max_cells_ce with "Other CEs" creation (lines 387-414)
    fig = _process_chem_env_data_treemap(
        chem_env_data=mock_data,
        max_cells_cn=None,
        max_cells_ce=2,  # Should create "Other CEs" entry for CN 4
        show_counts="value",
        cn_formatter=None,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    # Check that we have "Other CEs" entry
    labels = fig.data[0].labels
    assert any("Other CEs" in str(label) for label in labels)
    assert any("more not shown" in str(label) for label in labels)


def test_chem_env_treemap_max_cells_edge_cases() -> None:
    """Test max_cells parameters edge cases."""
    mock_data = [
        {"coord_num": 4, "chem_env_symbol": "T:4", "count": 10},
        {"coord_num": 6, "chem_env_symbol": "O:6", "count": 8},
    ]

    from pymatviz.treemap.chem_env import _process_chem_env_data_treemap

    # Test with max_cells_cn larger than available CNs (no "Other" should be created)
    fig = _process_chem_env_data_treemap(
        chem_env_data=mock_data,
        max_cells_cn=10,  # More than available CNs
        max_cells_ce=None,
        show_counts="value",
        cn_formatter=None,
    )

    labels = fig.data[0].labels
    assert not any("Other CNs" in str(label) for label in labels)

    # Test with max_cells_ce larger than available CEs (no "Other" should be created)
    fig = _process_chem_env_data_treemap(
        chem_env_data=mock_data,
        max_cells_cn=None,
        max_cells_ce=10,  # More than available CEs
        show_counts="value",
        cn_formatter=None,
    )

    labels = fig.data[0].labels
    assert not any("Other CEs" in str(label) for label in labels)


def test_chem_env_treemap_no_data() -> None:
    """Test chem_env_treemap with no data returns empty figure."""
    with patch("pymatviz.treemap.chem_env.normalize_structures") as mock_norm:
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
        # Invalid settings (silently ignored, won't raise but produce empty results)
        ({"invalid": "method"}, [[{"ce_symbol": "T:4"}]], {"T:4": 4}, True),
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

    if expected_valid:
        assert len(fig.data) == 1
        assert fig.data[0].type == "treemap"
        assert hasattr(fig.data[0], "labels")
    else:
        assert "No CN/CE data to display" in fig.layout.title.text


def test_chem_env_treemap_crystal_nn_method(
    structures: tuple[Structure, Structure],
) -> None:
    """Test chem_env_treemap with CrystalNN method."""
    fig = chem_env_treemap([structures[0]], chem_env_settings="crystal_nn")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "treemap"


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
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert "values" in fig.data[0]
    assert "labels" in fig.data[0]

    # Specific test for show_counts=False: ensure CN labels don't contain counts
    if show_counts is False:
        labels = fig.data[0].labels
        assert labels is not None

        # Find CN labels (coordination number labels)
        cn_labels = [label for label in labels if str(label).startswith("CN")]

        if len(cn_labels) > 0:  # Only test if we have CN labels
            for label in cn_labels:
                label_str = str(label)
                # Should not contain count/percentage indicators
                assert "N=" not in label_str
                assert "(" not in label_str
                assert "%" not in label_str
                # Should be clean format like "CN 4" or "CN:4"
                assert label_str.startswith(("CN ", "CN:"))


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
    """Test handling of ChemEnv analysis failures."""
    mocks = chemenv_mocks
    mocks["lgf_instance"].setup_structure.side_effect = RuntimeError("ChemEnv failed")

    fig = chem_env_treemap(mocks["structure"], chem_env_settings="chemenv")
    assert isinstance(fig, go.Figure)
    assert "No CN/CE data to display" in fig.layout.title.text


def test_chem_env_treemap_crystal_nn_alias(mock_structure: MagicMock) -> None:
    """Test that 'crystal_nn' uses CrystalNN method."""
    mock_structure.__len__.return_value = 1

    with (
        patch("pymatviz.treemap.chem_env.normalize_structures") as mock_norm,
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


def test_chem_env_treemap_invalid_show_counts(mock_structure: MagicMock) -> None:
    """Test invalid show_counts parameter."""
    with patch("pymatviz.treemap.chem_env.normalize_structures") as mock_norm:
        mock_norm.return_value.values.return_value = [mock_structure]
        with pytest.raises(ValueError, match="Invalid.*show_counts"):
            chem_env_treemap(
                mock_structure,
                show_counts="invalid",  # type: ignore[arg-type]
                chem_env_settings="chemenv",
            )


def test_chem_env_treemap_figure_styling(chemenv_mocks: dict[str, Any]) -> None:
    """Test that figure has proper styling applied."""
    mocks = chemenv_mocks
    mocks["lse"].coordination_environments = [[{"ce_symbol": "T:4"}]]
    mocks["geoms_instance"].get_symbol_cn_mapping.return_value = {"T:4": 4}

    fig = chem_env_treemap(mocks["structure"], chem_env_settings="chemenv")
    assert fig.layout.paper_bgcolor == "rgba(0, 0, 0, 0)"


def test_text_wrapping_functionality() -> None:
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

    from pymatviz.treemap.chem_env import _process_chem_env_data_treemap

    fig = _process_chem_env_data_treemap(
        chem_env_data=mock_data,
        max_cells_cn=None,
        max_cells_ce=None,
        show_counts="value",
        cn_formatter=None,
    )

    labels = fig.data[0].labels
    assert any("<br>" in str(label) for label in labels)
    assert any("short" in str(label) and "<br>" not in str(label) for label in labels)
