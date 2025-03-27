"""Unit tests for chemical clustering visualization functions."""

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
        show_chem_sys=False,  # Disable chemical system coloring
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


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


def test_colorbar_consistency(
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
) -> None:
    """Test colorbar consistency with and without log scaling."""
    prop_name = "Test Property"

    # Create plots with and without log scaling
    fig_linear = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name=prop_name,
        embedding_method="one-hot",
        projection_method="pca",
    )
    fig_log = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=np.log10(sample_properties),
        prop_name=f"log10({prop_name})",
        embedding_method="one-hot",
        projection_method="pca",
    )

    # Check colorbar ranges and ordering
    linear_range = fig_linear.data[0].marker.color
    log_range = fig_log.data[0].marker.color
    assert not np.array_equal(linear_range, log_range)
    assert np.array_equal(np.argsort(linear_range), np.argsort(log_range))


@pytest.mark.parametrize("sort_value", [True, False, 1, 0, -1])
def test_sorting_options(
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
    sort_value: bool | int,
) -> None:
    """Test different sorting options for property values."""
    fig = pmv.cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name="Test Property",
        embedding_method="one-hot",
        projection_method="pca",
        sort=sort_value,
    )

    colors = fig.data[0].marker.color
    if sort_value in (True, 1):
        # Ascending order (highest points plotted last)
        assert np.array_equal(np.argsort(colors), np.argsort(sample_properties))
    elif sort_value == -1:
        # Descending order (highest points plotted first)
        assert np.array_equal(np.argsort(colors), np.argsort(sample_properties)[::-1])
    else:
        # No sorting (False or 0)
        # Note: We can't easily test this case since the order might be arbitrary
        # but we can verify the plot was created successfully
        assert len(colors) == len(sample_properties)


def test_custom_sort_function(
    sample_compositions: list[Composition], sample_properties: np.ndarray
) -> None:
    """Test custom sorting function for property values."""

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

    colors = fig.data[0].marker.color
    assert np.array_equal(np.argsort(colors), np.argsort(sample_properties)[::-1])


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

    # Test with pre-computed embeddings and chemical system coloring
    fig = pmv.cluster_compositions(
        compositions=embeddings, show_chem_sys=True, projection_method="pca"
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (1,)
    assert fig.data[0].y.shape == (1,)
    assert fig.data[0].marker.color == "#636efa"


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
        show_chem_sys=True,
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

    # Test with custom projection kwargs
    fig_custom = pmv.cluster_compositions(
        compositions=embeddings,
        properties=sample_properties,
        prop_name="Test Property",
        projection_method="pca",
        n_components=3,
        projection_kwargs={"random_state": 42},
    )
    assert isinstance(fig_custom, go.Figure)
    assert len(fig_custom.data) == 1
    assert fig_custom.data[0].type == "scatter3d"
    for dim in "xyz":
        assert getattr(fig_custom.data[0], dim).shape == (3,)
