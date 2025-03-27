"""Unit tests for composition cluster plots."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Composition

import pymatviz as pmv


if TYPE_CHECKING:
    from pymatviz.cluster.composition import ProjectionMethod
    from pymatviz.cluster.composition.plot import ShowChemSys


np_rng = np.random.default_rng(seed=0)


@pytest.fixture
def sample_compositions() -> list[Composition]:
    """List of sample compositions for testing."""
    return [Composition(comp) for comp in ["Fe0.5Co0.5", "Ni0.7Cu0.3", "Zr0.6Ti0.4"]]


@pytest.fixture
def sample_properties() -> np.ndarray:
    """Sample property values for testing."""
    return np.array([1, 10, 100])


def test_basic_functionality(sample_compositions: list[Composition]) -> None:
    """Test basic functionality of chem_cluster_plot."""
    fig = pmv.cluster_compositions(
        compositions=sample_compositions,
        embedding_method="one-hot",
        projection_method="pca",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


@pytest.mark.parametrize("show_chem_sys", ["color", "shape", "color+shape", None])
def test_show_chem_sys(
    sample_compositions: list[Composition], show_chem_sys: ShowChemSys | None
) -> None:
    """Test different chemical system visualization options."""
    fig = pmv.cluster_compositions(
        compositions=sample_compositions,
        embedding_method="one-hot",
        projection_method="pca",
        show_chem_sys=show_chem_sys,
    )

    assert isinstance(fig, go.Figure)

    if show_chem_sys is None:  # No chemical system visualization
        assert len(fig.data) == 1
        assert not hasattr(fig.data[0], "symbol")
        # Should have a default marker color
        assert fig.data[0].marker.color == "#636efa"
    elif show_chem_sys == "shape":  # With shape mode, we get a single trace with
        # different symbols for chemical systems
        assert len(fig.data) == 1
        assert fig.data[0].mode == "markers"

        # Check symbols are set
        data_dict = fig.to_dict()["data"][0]
        assert "marker" in data_dict
        assert "symbol" in data_dict["marker"]
        # Symbol should be a list of valid plotly symbols
        assert isinstance(data_dict["marker"]["symbol"], list)
        # Each composition should have a different symbol
        symbols = data_dict["marker"]["symbol"]
        assert len(set(symbols)) >= min(len(sample_compositions), 3)

        # Should have a default marker color (no property coloring)
        assert fig.data[0].marker.color == "#636efa"
    elif show_chem_sys == "color":
        # For color mode, we should have multiple traces (one per chem system)
        assert len(fig.data) == 3
        assert fig.data[0].type == "scatter"

        # Each trace should have distinct color
        colors = [trace.marker.color for trace in fig.data]
        assert len(set(colors)) == 3  # Three unique colors

        # Each trace should contain 1 point (one composition per chem system)
        for trace in fig.data:
            assert len(trace.x) == 1
            assert len(trace.y) == 1

        # Ensure colors follow standard plotly palette (case-insensitive comparison)
        standard_colors = ["#636efa", "#ef553b", "#00cc96"]
        for color in colors:
            assert color.lower() in [c.lower() for c in standard_colors]
    else:  # color+shape
        # In color+shape mode, we have 1 trace with all points
        assert len(fig.data) == 1

        # Check symbols are set
        data_dict = fig.to_dict()["data"][0]
        assert "marker" in data_dict
        assert "symbol" in data_dict["marker"]

        # Symbol should be a list of valid plotly symbols
        assert isinstance(data_dict["marker"]["symbol"], list)

        # Each composition should have a different symbol
        symbols = data_dict["marker"]["symbol"]
        assert len(set(symbols)) >= min(len(sample_compositions), 3)

        # Check that marker has multiple colors (one per chemical system)
        assert isinstance(fig.data[0].marker.color, (list, tuple))
        # Each chemical system should have a different color
        assert len(fig.data[0].marker.color) == len(sample_compositions)

        # The colors should be standard Plotly colors (case-insensitive)
        standard_colors = ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a"]
        for color in fig.data[0].marker.color:
            assert color.lower() in [c.lower() for c in standard_colors]


@pytest.mark.parametrize("properties_input", ["array", "dict", "series"])
def test_property_coloring(
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
    properties_input: str,
) -> None:
    """Test property coloring with different input types."""
    prop_name = "Test Property"

    if properties_input == "array":
        properties = sample_properties
    elif properties_input == "dict":
        properties = {
            str(comp.formula): val
            for comp, val in zip(sample_compositions, sample_properties, strict=False)
        }
    else:  # series
        properties = pd.Series(
            sample_properties, index=[str(comp.formula) for comp in sample_compositions]
        )

    fig = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=properties,
        prop_name=prop_name,
        embedding_method="one-hot",
        projection_method="pca",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_chemical_system_with_properties(
    sample_compositions: list[Composition], sample_properties: np.ndarray
) -> None:
    """Test detailed behavior of chemical system visualization with properties."""
    # Test shape mode with properties
    fig_shape = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",
        projection_method="pca",
        show_chem_sys="shape",
    )

    assert len(fig_shape.data) == 1

    # Verify shape mapping
    data_dict = fig_shape.to_dict()["data"][0]
    assert isinstance(data_dict["marker"]["symbol"], list)

    # Verify properties are used for coloring
    assert "coloraxis" in data_dict["marker"]
    assert data_dict["marker"]["coloraxis"] == "coloraxis"

    # Verify the colorbar exists with the right title
    assert hasattr(fig_shape.layout, "coloraxis")
    assert fig_shape.layout.coloraxis.colorbar.title.text == "Test Property"

    # Test color mode with properties
    fig_color = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",
        projection_method="pca",
        show_chem_sys="color",  # Should use properties, not chem systems
    )

    assert len(fig_color.data) == 1  # Should have one trace (all properties)

    # Verify properties are used for coloring
    data_dict = fig_color.to_dict()["data"][0]
    assert "coloraxis" in data_dict["marker"]
    assert data_dict["marker"]["coloraxis"] == "coloraxis"

    # Verify the colorbar exists with the right title
    assert hasattr(fig_color.layout, "coloraxis")
    assert fig_color.layout.coloraxis.colorbar.title.text == "Test Property"

    # Test color+shape mode with properties
    fig_both = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",
        projection_method="pca",
        show_chem_sys="color+shape",
    )

    assert len(fig_both.data) == 1  # Should have one trace (all properties)

    # Verify properties are used for coloring
    data_dict = fig_both.to_dict()["data"][0]
    assert "coloraxis" in data_dict["marker"]
    assert data_dict["marker"]["coloraxis"] == "coloraxis"

    # Verify the colorbar exists with the right title
    assert hasattr(fig_both.layout, "coloraxis")
    assert fig_both.layout.coloraxis.colorbar.title.text == "Test Property"

    # Verify shapes are set
    assert isinstance(data_dict["marker"]["symbol"], list)


@pytest.mark.parametrize("sort_value", [True, False, 1, 0, -1])
def test_sorting_options(
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
    sort_value: bool | int,
) -> None:
    """Test different sorting options for property values."""
    # This implementation tests actual data points ordering
    fig = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",
        projection_method="pca",
        sort=sort_value,
    )

    # Check that the figure has coloraxis for property coloring
    assert hasattr(fig.layout, "coloraxis")

    # Get the custom data which contains the composition strings in current order
    data_dict = fig.to_dict()["data"][0]
    custom_data = data_dict["customdata"]
    composition_order = [item[0] for item in custom_data]

    # Get the expected sort order based on properties
    if sort_value in (True, 1):
        # Ascending order
        expected_indices = np.argsort(sample_properties)
        expected_compositions = [
            str(sample_compositions[i].formula) for i in expected_indices
        ]
    elif sort_value == -1:
        # Descending order
        expected_indices = np.argsort(sample_properties)[::-1]
        expected_compositions = [
            str(sample_compositions[i].formula) for i in expected_indices
        ]
    else:
        # No sorting (original order)
        expected_compositions = [str(comp.formula) for comp in sample_compositions]

    # Assert compositions are in the expected order
    # Compare simplified formulas to account for possible normalization differences
    simplified_actual = [comp.split()[0] for comp in composition_order]
    simplified_expected = [comp.split()[0] for comp in expected_compositions]
    assert simplified_actual == simplified_expected


def test_custom_sort_function(
    sample_compositions: list[Composition], sample_properties: np.ndarray
) -> None:
    """Test custom sorting function for property values with detailed validation."""
    # This implementation tests actual data points ordering

    # Define a custom sort function that returns indices in reverse value order
    def custom_sort(values: np.ndarray) -> np.ndarray:
        return np.argsort(values)[::-1]

    fig = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",
        projection_method="pca",
        sort=custom_sort,
    )

    # Check that the figure has coloraxis for property coloring
    assert hasattr(fig.layout, "coloraxis")

    # Get the custom data which contains the composition strings in current order
    data_dict = fig.to_dict()["data"][0]
    custom_data = data_dict["customdata"]
    composition_order = [item[0] for item in custom_data]

    # Get the expected sort order (descending by property value)
    expected_indices = np.argsort(sample_properties)[::-1]
    expected_compositions = [
        str(sample_compositions[i].formula) for i in expected_indices
    ]

    # Assert compositions are in the expected order
    # Compare simplified formulas to account for possible normalization differences
    simplified_actual = [comp.split()[0] for comp in composition_order]
    simplified_expected = [comp.split()[0] for comp in expected_compositions]
    assert simplified_actual == simplified_expected


def test_composite_viz_modes(
    sample_compositions: list[Composition], sample_properties: np.ndarray
) -> None:
    """Test combined visualization modes with different options."""
    # Test color+shape with custom color map
    custom_colors = {
        "Co-Fe": "#ff0000",  # Red
        "Cu-Ni": "#00ff00",  # Green
        "Ti-Zr": "#0000ff",  # Blue
    }

    # Run without properties, using custom color map
    fig = pmv.cluster_compositions(
        compositions=sample_compositions,
        embedding_method="one-hot",
        projection_method="pca",
        show_chem_sys="color",
        color_discrete_map=custom_colors,  # type: ignore[arg-type]
    )

    # Check that colors match our custom map
    trace_names = [trace.name for trace in fig.data]
    trace_colors = [trace.marker.color for trace in fig.data]

    for name, color in zip(trace_names, trace_colors, strict=False):
        assert color == custom_colors[name]

    # Edge case: Empty property array but with show_chem_sys=None
    # Should use default coloring
    fig_no_chem = pmv.cluster_compositions(
        compositions=sample_compositions,
        embedding_method="one-hot",
        projection_method="pca",
        show_chem_sys=None,
    )

    # Should have a single trace
    assert len(fig_no_chem.data) == 1
    # With default color
    assert fig_no_chem.data[0].marker.color == "#636efa"

    # Edge case: With property values and shape mode, using custom marker size
    large_marker = 20
    fig_large = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",
        projection_method="pca",
        show_chem_sys="shape",
        marker_size=large_marker,
    )

    # Check marker size
    assert fig_large.data[0].marker.size == large_marker


@pytest.mark.parametrize(
    ("projection_method", "projection_kwargs", "expected_stats"),
    [
        (
            "pca",
            {},
            {
                "PC1": r"PC1: \d+\.?\d*% \(cumulative: \d+\.?\d*%\)",
                "PC2": r"PC2: \d+\.?\d*% \(cumulative: \d+\.?\d*%\)",
            },
        ),
        (
            "tsne",
            {"perplexity": 1.0, "learning_rate": "auto"},
            {
                "perplexity": r"Perplexity: 1\.0",
                "learning_rate": r"Learning rate: auto",
            },
        ),
        (
            "isomap",
            {"n_neighbors": 2, "metric": "euclidean"},
            {
                "n_neighbors": r"n_neighbors: 2",
                "metric": r"metric: euclidean",
            },
        ),
        (
            "kernel_pca",
            {"kernel": "rbf", "gamma": 0.1},
            {
                "kernel": r"kernel: rbf",
                "gamma": r"gamma: 0\.1",
            },
        ),
    ],
)
def test_projection_stats(
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
    projection_method: ProjectionMethod,
    projection_kwargs: dict[str, Any],
    expected_stats: dict[str, str],
) -> None:
    """Test projection statistics display for different methods."""
    fig = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",
        projection_method=projection_method,
        projection_kwargs=projection_kwargs,
        show_projection_stats=True,
    )

    # Check if stats annotation exists
    stats_annotations = [
        anno
        for anno in fig.layout.annotations
        if any(
            re.search(pattern, anno.text) is not None
            for pattern in expected_stats.values()
        )
    ]
    assert len(stats_annotations) == 1, (
        f"{projection_method} stats annotation not found"
    )

    # Get the stats text
    stats_text = stats_annotations[0].text

    # For PCA, verify variance percentages are reasonable
    if projection_method == "pca":
        # Extract all PC percentages
        pc_percentages = []
        for line in stats_text.split("<br>"):
            if "PC" in line:
                # Extract both individual and cumulative percentages
                match = re.search(
                    r"PC\d+: (\d+\.?\d*)% \(cumulative: (\d+\.?\d*)%\)", line
                )
                assert match is not None, f"Could not parse PCA line: {line}"
                pc_percentages.append((float(match[1]), float(match[2])))

        # Check that we have the expected number of PCs
        assert len(pc_percentages) == 2, f"Expected 2 PCs, got {len(pc_percentages)}"

        # Check percentages are between 0 and 100 (since they're in percentage form)
        for pc, cum in pc_percentages:
            assert 0 <= pc <= 100, f"PC percentage {pc} out of range [0,100]"
            assert 0 <= cum <= 100, f"Cumulative percentage {cum} out of range [0,100]"

        # Check that percentages are in descending order
        assert pc_percentages[0][0] >= pc_percentages[1][0], (
            f"PC percentages not in descending order: {pc_percentages}"
        )

        # Check that cumulative percentages are in ascending order
        assert pc_percentages[0][1] <= pc_percentages[1][1], (
            f"Cumulative percentages not in ascending order: {pc_percentages}"
        )

        # Check that cumulative percentages are reasonable
        assert pc_percentages[-1][1] <= 100, (
            f"Final cumulative percentage {pc_percentages[-1][1]} exceeds 100%"
        )

    # Check that each expected stat is present and matches the pattern
    for pattern in expected_stats.values():
        assert re.search(pattern, stats_text) is not None, (
            f"Expected pattern '{pattern}' not found in stats text: {stats_text}"
        )


@pytest.mark.parametrize("show_projection_stats", [1, (1,), "foo"])
def test_projection_stats_invalid_type(
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
    show_projection_stats: Any,
) -> None:
    """Test that invalid show_projection_stats types raise TypeError."""
    with pytest.raises(
        TypeError, match=re.escape(f"{show_projection_stats=} must be bool or dict")
    ):
        pmv.cluster_compositions(
            compositions=sample_compositions,
            properties=sample_properties,
            prop_name="Test Property",
            embedding_method="one-hot",
            projection_method="pca",
            show_projection_stats=show_projection_stats,
        )


@pytest.mark.parametrize(
    ("embedding_method", "projection_method", "expected_shape", "expected_labels"),
    [
        ("magpie", "pca", (3,), ("Principal Component", "Principal Component")),
        ("one-hot", "tsne", (3,), ("t-SNE Component", "t-SNE Component")),
        ("deml", "pca", (3,), ("Principal Component", "Principal Component")),
        ("matminer", "isomap", (3,), ("Isomap Component", "Isomap Component")),
        (
            "matscholar_el",
            "kernel_pca",
            (3,),
            ("Kernel PCA Component", "Kernel PCA Component"),
        ),
    ],
)
def test_cluster_compositions_methods(
    sample_compositions: list[str],
    sample_properties: dict[str, float],
    embedding_method: str,
    projection_method: str,
    expected_shape: tuple[int, ...],
    expected_labels: tuple[str, str],
) -> None:
    """Test different combinations of embedding and projection methods."""
    # Skip UMAP test if not installed
    if projection_method == "umap":
        pytest.importorskip("umap")

    # Use appropriate kwargs for small datasets
    projection_kwargs = {}
    if projection_method == "tsne":
        projection_kwargs = {"perplexity": 1.0, "learning_rate": "auto"}
    elif projection_method == "umap":
        projection_kwargs = {"n_neighbors": 2, "min_dist": 0.1}
    elif projection_method == "isomap":
        projection_kwargs = {"n_neighbors": 2, "metric": "euclidean"}
    elif projection_method == "kernel_pca":
        projection_kwargs = {"kernel": "rbf", "gamma": 0.1}

    # Test 2D projection
    fig_2d = pmv.cluster_compositions(
        sample_compositions,
        properties=sample_properties,
        prop_name="property",
        embedding_method=embedding_method,  # type: ignore[arg-type]
        projection_method=projection_method,  # type: ignore[arg-type]
        projection_kwargs=projection_kwargs,
        n_components=2,
    )

    # Check that we got a valid figure
    assert isinstance(fig_2d, go.Figure)
    assert len(fig_2d.data) == 1
    assert fig_2d.data[0].type == "scatter"
    assert fig_2d.data[0].x.shape == expected_shape
    assert fig_2d.data[0].y.shape == expected_shape

    # Check axis labels
    assert fig_2d.layout.xaxis.title.text == f"{expected_labels[0]} 1"
    assert fig_2d.layout.yaxis.title.text == f"{expected_labels[1]} 2"

    # Test 3D projection
    fig_3d = pmv.cluster_compositions(
        sample_compositions,
        properties=sample_properties,
        prop_name="property",
        embedding_method=embedding_method,  # type: ignore[arg-type]
        projection_method=projection_method,  # type: ignore[arg-type]
        projection_kwargs=projection_kwargs,
        n_components=3,
    )

    # Check that we got a valid figure
    assert isinstance(fig_3d, go.Figure)
    assert len(fig_3d.data) == 1
    assert fig_3d.data[0].type == "scatter3d"
    assert fig_3d.data[0].x.shape == expected_shape
    assert fig_3d.data[0].y.shape == expected_shape
    assert fig_3d.data[0].z.shape == expected_shape

    # Check 3D axis labels
    assert fig_3d.layout.scene.xaxis.title.text == f"{expected_labels[0]} 1"
    assert fig_3d.layout.scene.yaxis.title.text == f"{expected_labels[1]} 2"
    assert fig_3d.layout.scene.zaxis.title.text == f"{expected_labels[0]} 3"


def test_cluster_compositions_custom_embedding(
    sample_compositions: list[str],
    sample_properties: dict[str, float],
) -> None:
    """Test using a custom embedding function."""

    def custom_embedding(compositions: list[str], **_kwargs: Any) -> np.ndarray:
        """Custom embedding that just returns random values."""
        return np_rng.random((len(compositions), 10))

    fig = pmv.cluster_compositions(
        sample_compositions,
        properties=sample_properties,
        prop_name="property",
        embedding_method=custom_embedding,  # type: ignore[arg-type]
    )

    # Check that we got a valid figure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_cluster_compositions_custom_projection(
    sample_compositions: list[str],
    sample_properties: dict[str, float],
) -> None:
    """Test using a custom projection function."""

    def custom_projection(
        data: np.ndarray, n_components: int = 2, **_kwargs: Any
    ) -> np.ndarray:
        """Custom projection that just returns random values."""
        return np_rng.random((data.shape[0], n_components))

    fig = pmv.cluster_compositions(
        sample_compositions,
        properties=sample_properties,
        prop_name="property",
        projection_method=custom_projection,  # type: ignore[arg-type]
    )

    # Check that we got a valid figure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_cluster_compositions_custom_both(
    sample_compositions: list[str],
    sample_properties: dict[str, float],
) -> None:
    """Test using both custom embedding and projection functions."""

    def custom_embedding(compositions: list[str], **_kwargs: Any) -> np.ndarray:
        """Custom embedding that just returns random values."""
        return np_rng.random((len(compositions), 10))

    def custom_projection(
        data: np.ndarray, n_components: int = 2, **_kwargs: Any
    ) -> np.ndarray:
        """Custom projection that just returns random values."""
        return np_rng.random((data.shape[0], n_components))

    fig = pmv.cluster_compositions(
        sample_compositions,
        properties=sample_properties,
        prop_name="property",
        embedding_method=custom_embedding,  # type: ignore[arg-type]
        projection_method=custom_projection,  # type: ignore[arg-type]
    )

    # Check that we got a valid figure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_cluster_compositions_custom_kwargs(
    sample_compositions: list[str],
    sample_properties: dict[str, float],
) -> None:
    """Test passing kwargs to custom functions."""

    def custom_embedding(
        compositions: list[str], scale: float = 1.0, **_kwargs: Any
    ) -> np.ndarray:
        """Custom embedding that scales random values."""
        return np_rng.random((len(compositions), 10)) * scale

    def custom_projection(
        data: np.ndarray, n_components: int = 2, scale: float = 1.0, **_kwargs: Any
    ) -> np.ndarray:
        """Custom projection that scales random values."""
        return np_rng.random((data.shape[0], n_components)) * scale

    fig = pmv.cluster_compositions(
        sample_compositions,
        properties=sample_properties,
        prop_name="property",
        embedding_method=custom_embedding,  # type: ignore[arg-type]
        projection_method=custom_projection,
        embedding_kwargs={"scale": 2.0},
        projection_kwargs={"scale": 3.0},
    )

    # Check that we got a valid figure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_precomputed_embeddings(
    sample_compositions: list[Composition], sample_properties: np.ndarray
) -> None:
    """Test passing pre-computed embeddings as a dictionary."""
    # Create pre-computed embeddings
    embeddings = {str(comp.formula): np_rng.random(10) for comp in sample_compositions}

    # Test with pre-computed embeddings
    fig = pmv.cluster_compositions(
        compositions=embeddings,
        properties=sample_properties,
        prop_name="Test Property",
        projection_method="pca",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)

    # Test that embedding_method is ignored when using pre-computed embeddings
    fig2 = pmv.cluster_compositions(
        compositions=embeddings,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",  # This should be ignored
        projection_method="pca",
    )
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) == 1
    assert fig2.data[0].type == "scatter"
    assert fig2.data[0].x.shape == (3,)
    assert fig2.data[0].y.shape == (3,)


def test_precomputed_embeddings_with_properties_dict(
    sample_compositions: list[Composition],
) -> None:
    """Test pre-computed embeddings with dictionary properties."""
    # Create pre-computed embeddings
    embeddings = {str(comp.formula): np_rng.random(10) for comp in sample_compositions}

    # Create properties dictionary
    properties = {
        str(comp.formula): float(idx) for idx, comp in enumerate(sample_compositions)
    }

    # Test with pre-computed embeddings and dictionary properties
    fig = pmv.cluster_compositions(
        compositions=embeddings,
        properties=properties,
        prop_name="Test Property",
        projection_method="pca",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_precomputed_embeddings_with_properties_series(
    sample_compositions: list[Composition],
) -> None:
    """Test pre-computed embeddings with pandas Series properties."""
    # Create pre-computed embeddings
    embeddings = {str(comp.formula): np_rng.random(10) for comp in sample_compositions}

    # Create properties Series
    properties = pd.Series(
        [float(idx) for idx in range(len(sample_compositions))],
        index=[str(comp.formula) for comp in sample_compositions],
    )

    # Test with pre-computed embeddings and Series properties
    fig = pmv.cluster_compositions(
        compositions=embeddings,
        properties=properties,
        prop_name="Test Property",
        projection_method="pca",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_precomputed_embeddings_with_chemical_systems(
    sample_compositions: list[Composition],
) -> None:
    """Test pre-computed embeddings with chemical system coloring."""
    # Create pre-computed embeddings
    embeddings = {str(comp.formula): np_rng.random(10) for comp in sample_compositions}

    # Test with pre-computed embeddings and chemical system coloring (color mode)
    fig = pmv.cluster_compositions(
        compositions=embeddings, show_chem_sys="color", projection_method="pca"
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # One trace per chemical system (color mode)
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (1,)
    assert fig.data[0].y.shape == (1,)
    assert fig.data[0].marker.color == "#636efa"

    # Test with shape mode
    fig_shape = pmv.cluster_compositions(
        compositions=embeddings, show_chem_sys="shape", projection_method="pca"
    )
    assert isinstance(fig_shape, go.Figure)
    # Should have one data trace
    assert len(fig_shape.data) == 1
    # First trace contains all the data
    assert fig_shape.data[0].mode == "markers"
    assert not fig_shape.data[0].showlegend

    # Check symbols are set
    data_dict = fig_shape.to_dict()["data"][0]
    assert "marker" in data_dict
    assert "symbol" in data_dict["marker"]
    # Symbol should be a list of valid plotly symbols
    assert isinstance(data_dict["marker"]["symbol"], list)

    # Test with color+shape mode
    fig_both = pmv.cluster_compositions(
        compositions=embeddings, show_chem_sys="color+shape", projection_method="pca"
    )
    assert isinstance(fig_both, go.Figure)
    # In color+shape mode, we have 1 trace with all points
    assert len(fig_both.data) == 1
    data_dict = fig_both.to_dict()["data"][0]
    assert "marker" in data_dict
    assert "symbol" in data_dict["marker"]
    # Symbol should be a list of valid plotly symbols
    assert isinstance(data_dict["marker"]["symbol"], list)


def test_precomputed_embeddings_3d(
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
) -> None:
    """Test pre-computed embeddings with 3D projections and different property types."""
    # Create pre-computed embeddings with 10 dimensions
    embeddings = {str(comp.formula): np_rng.random(10) for comp in sample_compositions}

    # Test with array properties
    fig_array = pmv.cluster_compositions(
        compositions=embeddings,
        properties=sample_properties,
        prop_name="Test Property",
        projection_method="pca",
        n_components=3,
    )

    # Basic figure checks
    assert isinstance(fig_array, go.Figure)
    assert len(fig_array.data) == 1
    assert fig_array.data[0].type == "scatter3d"

    # Check data shapes
    for dim in "xyz":
        assert getattr(fig_array.data[0], dim).shape == (3,)

    # Check axis labels
    assert fig_array.layout.scene.xaxis.title.text == "Principal Component 1"
    assert fig_array.layout.scene.yaxis.title.text == "Principal Component 2"
    assert fig_array.layout.scene.zaxis.title.text == "Principal Component 3"
    assert fig_array.layout.coloraxis is not None
    assert fig_array.layout.coloraxis.colorbar.title.text == "Test Property"

    # Test with dictionary properties
    properties_dict = {
        str(comp.formula): val
        for comp, val in zip(sample_compositions, sample_properties, strict=False)
    }
    fig_dict = pmv.cluster_compositions(
        compositions=embeddings,
        properties=properties_dict,
        prop_name="Test Property",
        projection_method="pca",
        n_components=3,
    )
    assert isinstance(fig_dict, go.Figure)
    assert len(fig_dict.data) == 1
    assert fig_dict.data[0].type == "scatter3d"
    for dim in "xyz":
        assert getattr(fig_dict.data[0], dim).shape == (3,)

    # Test with pandas Series properties
    properties_series = pd.Series(
        sample_properties,
        index=[str(comp.formula) for comp in sample_compositions],
    )
    fig_series = pmv.cluster_compositions(
        compositions=embeddings,
        properties=properties_series,
        prop_name="Test Property",
        projection_method="pca",
        n_components=3,
    )
    assert isinstance(fig_series, go.Figure)
    assert len(fig_series.data) == 1
    assert fig_series.data[0].type == "scatter3d"
    for dim in "xyz":
        assert getattr(fig_series.data[0], dim).shape == (3,)

    # Test without properties (chemical system coloring)
    fig_no_props = pmv.cluster_compositions(
        compositions=embeddings,
        projection_method="pca",
        n_components=3,
        show_chem_sys="color",  # Color mode
    )
    assert isinstance(fig_no_props, go.Figure)
    # With chemical system coloring, we get one trace per system
    assert len(fig_no_props.data) == 3
    for trace in fig_no_props.data:
        assert trace.type == "scatter3d"
        for dim in "xyz":
            assert getattr(trace, dim).shape == (1,)  # One point per chemical system
        # Check hover data
        hover_text = trace.customdata[0][1]
        assert hover_text.startswith("Composition:")
        for word in "PC1", "PC2", "PC3", "Chemical System":
            assert f"<br>{word}" in hover_text

    # Test with shape mode in 3D
    fig_shape_3d = pmv.cluster_compositions(
        compositions=embeddings,
        projection_method="pca",
        n_components=3,
        show_chem_sys="shape",
    )
    assert isinstance(fig_shape_3d, go.Figure)
    # Should have one data trace
    assert len(fig_shape_3d.data) == 1
    # First trace contains all the data
    assert fig_shape_3d.data[0].mode == "markers"
    assert not fig_shape_3d.data[0].showlegend

    # Check symbols are set
    data_dict = fig_shape_3d.to_dict()["data"][0]
    assert "marker" in data_dict
    assert "symbol" in data_dict["marker"]
    # Symbol should be a list of valid plotly symbols
    assert isinstance(data_dict["marker"]["symbol"], list)
    # Verify the trace is of type Scatter3d, not Scatter
    assert fig_shape_3d.data[0].type == "scatter3d"


def test_marker_size_adjustment_3d(
    sample_compositions: list[Composition], sample_properties: np.ndarray
) -> None:
    """Test that marker size is automatically halved for 3D plots."""
    # Test with 2D plot - marker size should be as specified
    marker_size = 12
    for n_components, expected_size in ((2, marker_size), (3, marker_size / 3)):
        fig = pmv.cluster_compositions(
            compositions=sample_compositions,
            properties=sample_properties,
            prop_name="Test Property",
            n_components=n_components,
            marker_size=marker_size,
        )

        # Check marker size is as specified for 2D
        assert fig.data[0].marker.size == expected_size
