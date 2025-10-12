"""Tests for data projection functions."""

from typing import Any

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap

from pymatviz.cluster.composition import ProjectionMethod, project_vectors


@pytest.fixture
def sample_data() -> np.ndarray:
    """Generate sample data for testing projection methods."""
    xs, _ = make_blobs(  # Create synthetic data with 3 clusters in 10 dimensions
        n_samples=100, n_features=10, centers=3, random_state=0, cluster_std=1.0
    )
    return xs


def test_project_vectors_basic(sample_data: np.ndarray) -> None:
    """Test basic functionality of project_vectors."""
    # Test with default parameters (PCA)
    result, proj_obj = project_vectors(sample_data)

    # Check shape and type of projected data
    assert result.shape == (100, 2)
    assert isinstance(result, np.ndarray)

    # Check projection object
    assert isinstance(proj_obj, PCA)
    assert proj_obj.n_components_ == 2
    assert proj_obj.explained_variance_ratio_.shape == (2,)

    # Check that result is not degenerate
    assert not np.allclose(result, 0, atol=1e-10)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

    # Check that result has reasonable scale
    # PCA should roughly preserve scale
    assert np.std(result) == pytest.approx(1.0, rel=0.5)
    # PCA should center the data
    assert np.mean(result) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    ("method", "kwargs", "expected_obj_type", "obj_attrs"),
    [
        ("pca", {}, PCA, {"n_components_": 2}),
        (
            "tsne",
            {"perplexity": 10.0, "learning_rate": 100.0},
            TSNE,
            {"n_components": 2, "perplexity": 10.0},
        ),
        (
            "umap",
            {"n_neighbors": 5, "min_dist": 0.2},
            None,
            {"n_neighbors": 5, "min_dist": 0.2},
        ),
        (
            "isomap",
            {"n_neighbors": 5, "metric": "euclidean"},
            Isomap,
            {"n_neighbors": 5, "metric": "euclidean"},
        ),
        (
            "kernel_pca",
            {"kernel": "rbf", "gamma": 0.1},
            KernelPCA,
            {"n_components": 2, "kernel": "rbf", "gamma": 0.1},
        ),
    ],
)
def test_project_vectors_methods(
    sample_data: np.ndarray,
    method: ProjectionMethod,
    kwargs: dict[str, Any],
    expected_obj_type: type,
    obj_attrs: dict[str, Any],
) -> None:
    """Test different projection methods."""
    if method == "umap":
        pytest.importorskip("umap")
        from umap import UMAP

        kwargs = {"n_neighbors": 5, "min_dist": 0.2}
        expected_obj_type = UMAP
        obj_attrs = {"n_neighbors": 5, "min_dist": 0.2}

    result, proj_obj = project_vectors(sample_data, method=method, **kwargs)  # type: ignore[arg-type]

    # Check shape of projected data
    assert result.shape == (100, 2)

    # Check projection object type
    assert isinstance(proj_obj, expected_obj_type)

    # Check projection object attributes
    for attr, value in obj_attrs.items():
        assert getattr(proj_obj, attr) == value

    # Check that result is not degenerate
    assert not np.allclose(result, 0, atol=1e-10)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

    # Check that result has reasonable scale (not zero or infinite)
    assert 0.01 < np.std(result) < 100

    # Check that result preserves some structure
    if method in ("pca", "isomap", "kernel_pca"):
        # These methods should preserve some linear structure
        assert np.linalg.matrix_rank(result) == 2
    else:
        # Non-linear methods should spread out the points
        assert np.std(result) > 0.1


@pytest.mark.parametrize("n_components", [2, 3])
def test_project_vectors_components(sample_data: np.ndarray, n_components: int) -> None:
    """Test projecting to different numbers of components."""
    result, proj_obj = project_vectors(sample_data, n_components=n_components)

    # Check shape matches requested components
    assert result.shape == (100, n_components)
    assert isinstance(proj_obj, PCA)
    assert proj_obj.n_components_ == n_components

    # Check that result is not degenerate
    assert not np.allclose(result, 0, atol=1e-10)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

    # Check that result has reasonable scale
    assert np.std(result) == pytest.approx(1.0, rel=0.5)
    assert np.mean(result) == pytest.approx(0.0, abs=1e-10)

    # Check that result preserves some structure
    assert np.linalg.matrix_rank(result) == n_components


def test_project_vectors_tsne_params(sample_data: np.ndarray) -> None:
    """Test t-SNE with custom parameters."""
    result, tsne_obj = project_vectors(
        sample_data,
        method="tsne",
        perplexity=10.0,
        learning_rate=100.0,
        max_iter=500,
    )

    # Check shape and object type
    assert result.shape == (100, 2)
    assert isinstance(tsne_obj, TSNE)

    # Check t-SNE specific parameters
    assert tsne_obj.perplexity == 10.0
    assert tsne_obj.learning_rate == 100.0
    assert tsne_obj.max_iter == 500

    # Check that result is not degenerate
    assert not np.allclose(result, 0, atol=1e-10)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

    # Check that result has reasonable scale (not zero or infinite)
    assert 0.01 < np.std(result) < 100


def test_project_vectors_umap_params(sample_data: np.ndarray) -> None:
    """Test UMAP with custom parameters."""
    pytest.importorskip("umap")
    from umap import UMAP

    result, umap_obj = project_vectors(
        sample_data,
        method="umap",
        n_neighbors=10,
        min_dist=0.05,
    )

    # Check shape and object type
    assert result.shape == (100, 2)
    assert isinstance(umap_obj, UMAP)

    # Check UMAP specific parameters
    assert umap_obj.n_neighbors == 10
    assert umap_obj.min_dist == 0.05

    # Check that result is not degenerate
    assert not np.allclose(result, 0, atol=1e-10)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

    # Check that result has reasonable scale (not zero or infinite)
    assert 0.01 < np.std(result) < 100


def test_project_vectors_with_scaling(sample_data: np.ndarray) -> None:
    """Test projection with and without data scaling."""
    # With scaling (default)
    result_with_scaling, proj_obj1 = project_vectors(sample_data, scale_data=True)

    # Without scaling
    result_without_scaling, proj_obj2 = project_vectors(sample_data, scale_data=False)

    # Results should be different
    assert not np.allclose(result_with_scaling, result_without_scaling, atol=1e-10)

    # Both projection objects should be PCA instances
    assert isinstance(proj_obj1, PCA)
    assert isinstance(proj_obj2, PCA)
    assert proj_obj1.n_components_ == proj_obj2.n_components_ == 2
    assert proj_obj1.explained_variance_ratio_.mean() == pytest.approx(0.4348426)
    assert proj_obj2.explained_variance_ratio_.mean() == pytest.approx(0.4691580)

    # Check that both results have reasonable scale (not zero or infinite)
    assert 0.01 < np.std(result_with_scaling) < 100
    assert 0.01 < np.std(result_without_scaling) < 100


def test_project_vectors_random_state(sample_data: np.ndarray) -> None:
    """Test reproducibility with fixed random state."""
    pytest.importorskip("umap")
    from umap import UMAP

    # Run twice with same random state for UMAP
    result1, umap_obj1 = project_vectors(sample_data, method="umap", random_state=0)
    result2, umap_obj2 = project_vectors(sample_data, method="umap", random_state=0)
    assert isinstance(umap_obj1, UMAP)
    assert umap_obj1.random_state == 0
    assert umap_obj1 is not umap_obj2

    # Results should be identical with same random state
    assert np.allclose(result1, result2, atol=1e-10)

    # Run with different random state for UMAP
    result3, umap_obj3 = project_vectors(sample_data, method="umap", random_state=1)
    assert isinstance(umap_obj3, UMAP)
    assert umap_obj3.random_state == 1
    # Results should be different with different random states
    assert not np.allclose(result1, result3, atol=1e-10)

    # Check that both results have reasonable scale
    assert np.std(result1) == pytest.approx(1.0, rel=0.5)
    assert np.std(result3) == pytest.approx(1.0, rel=0.5)


def test_project_vectors_invalid_method(sample_data: np.ndarray) -> None:
    """Test error handling for invalid method."""
    with pytest.raises(ValueError, match="Unknown projection method="):
        project_vectors(sample_data, method="invalid_method")  # type: ignore[arg-type]


def test_project_vectors_invalid_components(sample_data: np.ndarray) -> None:
    """Test error handling for invalid number of components."""
    with pytest.raises(ValueError, match="n_components must be at least 2"):
        project_vectors(sample_data, n_components=1)


def test_project_vectors_not_enough_samples() -> None:
    """Test error handling for dataset with too few samples."""
    # Create a dataset with only 1 sample
    xs = np.random.default_rng(seed=0).random(size=(1, 5))

    # Trying to project to 2 components should fail
    with pytest.raises(ValueError, match="Not enough samples"):
        project_vectors(xs, n_components=2)


def test_project_vectors_tsne_max_components(sample_data: np.ndarray) -> None:
    """Test t-SNE with more than 3 components raises an error."""
    with pytest.raises(
        ValueError, match="t-SNE visualization typically uses 2 or 3 components"
    ):
        project_vectors(sample_data, method="tsne", n_components=4)


def test_project_vectors_pca_consistency(sample_data: np.ndarray) -> None:
    """Test PCA projection consistency with same random state."""
    result1, pca_obj1 = project_vectors(sample_data, method="pca", random_state=0)
    result2, pca_obj2 = project_vectors(sample_data, method="pca", random_state=0)

    # PCA should be deterministic with same random state
    assert np.allclose(result1, result2, atol=1e-10)

    # Also check with different random state (should still be the same for PCA)
    result3, pca_obj3 = project_vectors(sample_data, method="pca", random_state=1)
    assert np.allclose(result1, result3, atol=1e-10)

    # Check that results have reasonable scale
    assert np.std(result1) == pytest.approx(1.0, rel=0.5)
    assert np.mean(result1) == pytest.approx(0.0, abs=1e-10)

    assert isinstance(pca_obj1, PCA)
    assert isinstance(pca_obj2, PCA)
    assert isinstance(pca_obj3, PCA)
    # Check that all PCA objects have the same explained variance ratios
    assert np.allclose(
        pca_obj1.explained_variance_ratio_, pca_obj2.explained_variance_ratio_
    )
    assert np.allclose(
        pca_obj1.explained_variance_ratio_, pca_obj3.explained_variance_ratio_
    )


def test_project_vectors_small_dataset() -> None:
    """Test handling of small datasets."""
    # Create a very small dataset
    xs, _ = make_blobs(n_samples=20, n_features=5, centers=2, random_state=0)

    # Test with t-SNE (should adjust perplexity)
    result_tsne, tsne_obj = project_vectors(xs, method="tsne")
    assert result_tsne.shape == (len(xs), 2)
    assert isinstance(tsne_obj, TSNE)
    assert tsne_obj.perplexity == pytest.approx(min(30, len(xs) / 3))
    assert not np.allclose(result_tsne, 0, atol=1e-10)
    assert not np.any(np.isnan(result_tsne))
    assert not np.any(np.isinf(result_tsne))
    assert np.std(result_tsne) == pytest.approx(1.0, rel=0.5)

    pytest.importorskip("umap")
    from umap import UMAP

    result_umap, umap_obj = project_vectors(xs, method="umap")
    assert result_umap.shape == (len(xs), 2)
    assert isinstance(umap_obj, UMAP)
    assert umap_obj.n_neighbors == 15  # Default value
    assert not np.allclose(result_umap, 0, atol=1e-10)
    assert not np.any(np.isnan(result_umap))
    assert not np.any(np.isinf(result_umap))
    assert np.std(result_umap) == pytest.approx(1.0, rel=0.5)
