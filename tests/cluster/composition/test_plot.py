"""Unit tests for chemical clustering visualization functions."""

from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Composition

from pymatviz.cluster.composition import ProjectionMethod, cluster_compositions


np_rng = np.random.default_rng(seed=0)


@pytest.fixture
def sample_compositions() -> list[Composition]:
    """Create a list of sample compositions for testing.

    Returns:
        List of pymatgen Composition objects
    """
    return [
        Composition("Fe0.5Co0.5"),
        Composition("Ni0.7Cu0.3"),
        Composition("Zr0.6Ti0.4"),
    ]


@pytest.fixture
def sample_properties() -> np.ndarray:
    """Create sample property values for testing.

    Returns:
        Array of property values with a wide range (1-1000)
    """
    return np.array([1, 10, 100])


def test_basic_functionality(sample_compositions: list[Composition]) -> None:
    """Test basic functionality of chem_cluster_plot."""
    fig = cluster_compositions(
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


@pytest.mark.parametrize(
    "properties_input",
    [
        "array",
        "dict",
        "series",
    ],
)
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

    fig = cluster_compositions(
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
    fig_linear = cluster_compositions(
        compositions=sample_compositions,
        properties=sample_properties,
        prop_name=prop_name,
        embedding_method="one-hot",
        projection_method="pca",
    )
    fig_log = cluster_compositions(
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
    fig = cluster_compositions(
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
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
) -> None:
    """Test custom sorting function for property values."""

    def custom_sort(values: np.ndarray) -> np.ndarray:
        return np.argsort(values)[::-1]

    fig = cluster_compositions(
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
    ("projection_method", "projection_kwargs", "stats_text"),
    [
        ("pca", {}, ["PC1:", "PC2:"]),
        (
            "tsne",
            {"perplexity": 1.0, "learning_rate": "auto"},
            ["Perplexity:", "Learning rate:"],
        ),
        (
            "isomap",
            {"n_neighbors": 2, "metric": "euclidean"},
            ["n_neighbors:", "metric:"],
        ),
        ("kernel_pca", {"kernel": "rbf", "gamma": 0.1}, ["kernel:", "gamma:"]),
    ],
)
def test_projection_stats(
    sample_compositions: list[Composition],
    sample_properties: np.ndarray,
    projection_method: ProjectionMethod,
    projection_kwargs: dict[str, Any],
    stats_text: list[str],
) -> None:
    """Test projection statistics display for different methods."""
    fig = cluster_compositions(
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
        ann
        for ann in fig.layout.annotations
        if any(text in str(ann.text) for text in stats_text)
    ]
    assert len(stats_annotations) == 1, (
        f"{projection_method} stats annotation not found"
    )

    # For PCA, verify variance percentages are reasonable
    if projection_method == "pca":
        import re

        percentages = [
            float(re.search(r"(\d+\.?\d*)%", line)[1]) / 100  # type: ignore[index]
            for line in stats_annotations[0].text.split("<br>")
            if "PC" in line
        ]
        # Check that percentages are between 0 and 1
        assert all(0 <= p <= 1 for p in percentages), (
            "Variance percentages out of range"
        )
        # Check that percentages are in descending order
        assert all(p1 >= p2 for p1, p2 in itertools.pairwise(percentages)), (
            f"{percentages=}"
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
        cluster_compositions(
            compositions=sample_compositions,
            properties=sample_properties,
            prop_name="Test Property",
            embedding_method="one-hot",
            projection_method="pca",
            show_projection_stats=show_projection_stats,
        )


@pytest.mark.parametrize(
    ("embedding_method", "projection_method", "expected_shape"),
    [
        ("magpie", "pca", (3,)),
        ("one-hot", "tsne", (3,)),
        ("deml", "pca", (3,)),
        ("matminer", "isomap", (3,)),
        ("matscholar_el", "kernel_pca", (3,)),
    ],
)
def test_cluster_compositions_methods(
    sample_compositions: list[str],
    sample_properties: dict[str, float],
    embedding_method: str,
    projection_method: str,
    expected_shape: tuple[int, ...],
) -> None:
    """Test different combinations of embedding and projection methods."""
    # Skip UMAP test if not installed
    if projection_method == "umap":
        try:
            import umap  # noqa: F401
        except ImportError:
            pytest.skip("umap-learn is not installed")

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

    fig = cluster_compositions(
        sample_compositions,
        properties=sample_properties,
        prop_name="property",
        embedding_method=embedding_method,  # type: ignore[arg-type]
        projection_method=projection_method,  # type: ignore[arg-type]
        projection_kwargs=projection_kwargs,
    )

    # Check that we got a valid figure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == expected_shape
    assert fig.data[0].y.shape == expected_shape


def test_cluster_compositions_custom_embedding(
    sample_compositions: list[str],
    sample_properties: dict[str, float],
) -> None:
    """Test using a custom embedding function."""

    def custom_embedding(compositions: list[str], **_kwargs: Any) -> np.ndarray:
        """Custom embedding that just returns random values."""
        return np_rng.random((len(compositions), 10))

    fig = cluster_compositions(
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

    fig = cluster_compositions(
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

    fig = cluster_compositions(
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

    fig = cluster_compositions(
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
    fig = cluster_compositions(
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
    fig2 = cluster_compositions(
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
    fig = cluster_compositions(
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
    fig = cluster_compositions(
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
    fig = cluster_compositions(
        compositions=embeddings, show_chem_sys=True, projection_method="pca"
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)
    assert "chem_system" in fig.data[0].marker.color
