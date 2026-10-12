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


@pytest.mark.parametrize(
    ("method", "kwargs", "expected_obj_type", "obj_attrs"),
    [
        ("pca", {"whiten": True}, PCA, {"n_components_": 2, "whiten": True}),
        (
            "tsne",
            {"perplexity": 10.0, "learning_rate": 100.0, "max_iter": 500},
            TSNE,
            {
                "n_components": 2,
                "perplexity": 10.0,
                "learning_rate": 100.0,
                "max_iter": 500,
            },
        ),
        (
            "umap",
            {"n_neighbors": 5, "min_dist": 0.2, "spread": 2},
            None,
            {"n_neighbors": 5, "min_dist": 0.2, "spread": 2},
        ),
        (
            "isomap",
            {"n_neighbors": 5, "metric": "euclidean", "path_method": "D"},
            Isomap,
            {"n_neighbors": 5, "metric": "euclidean", "path_method": "D"},
        ),
        (
            "kernel_pca",
            {"kernel": "rbf", "gamma": 0.1, "fit_inverse_transform": True},
            KernelPCA,
            {
                "n_components": 2,
                "kernel": "rbf",
                "gamma": 0.1,
                "fit_inverse_transform": True,
            },
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
    if method == ProjectionMethod.umap:
        pytest.importorskip("umap")
        from umap import UMAP

        expected_obj_type = UMAP

    result, proj_obj = project_vectors(sample_data, method=method, **kwargs)

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
    """Default PCA projection to different numbers of components."""
    result, proj_obj = project_vectors(sample_data, n_components=n_components)

    # Check shape matches requested components
    assert result.shape == (100, n_components)
    assert isinstance(proj_obj, PCA)
    assert proj_obj.n_components_ == n_components
    assert proj_obj.explained_variance_ratio_.shape == (n_components,)

    # Check that result is not degenerate
    assert not np.allclose(result, 0, atol=1e-10)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

    # Check that result has reasonable scale
    assert np.std(result) == pytest.approx(1.0, rel=0.5)
    assert np.mean(result) == pytest.approx(0.0, abs=1e-10)

    # Check that result preserves some structure
    assert np.linalg.matrix_rank(result) == n_components


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
    result1, umap_obj1 = project_vectors(
        sample_data, method=ProjectionMethod.umap, random_state=0
    )
    result2, umap_obj2 = project_vectors(
        sample_data, method=ProjectionMethod.umap, random_state=0
    )
    assert isinstance(umap_obj1, UMAP)
    assert umap_obj1.random_state == 0
    assert umap_obj1 is not umap_obj2

    # Results should be identical with same random state
    assert np.allclose(result1, result2, atol=1e-10)

    # Run with different random state for UMAP
    result3, umap_obj3 = project_vectors(
        sample_data, method=ProjectionMethod.umap, random_state=1
    )
    assert isinstance(umap_obj3, UMAP)
    assert umap_obj3.random_state == 1
    # Results should be different with different random states
    assert not np.allclose(result1, result3, atol=1e-10)

    # Check that both results have reasonable scale
    assert np.std(result1) == pytest.approx(1.0, rel=0.5)
    assert np.std(result3) == pytest.approx(1.0, rel=0.5)


@pytest.mark.parametrize(
    ("n_samples", "kwargs", "match"),
    [
        (100, {"method": "invalid_method"}, "Unknown projection method="),
        (100, {"n_components": 1}, "n_components must be at least 2"),
        (1, {"n_components": 2}, "Not enough samples"),
        (
            100,
            {"method": ProjectionMethod.tsne, "n_components": 4},
            "t-SNE visualization typically uses 2 or 3 components",
        ),
        (3, {"method": ProjectionMethod.umap}, "UMAP needs at least n_components"),
    ],
)
def test_project_vectors_invalid_inputs(
    sample_data: np.ndarray, n_samples: int, kwargs: dict[str, Any], match: str
) -> None:
    """Invalid methods, component counts and too few samples raise ValueError."""
    with pytest.raises(ValueError, match=match):
        project_vectors(sample_data[:n_samples], **kwargs)


@pytest.mark.parametrize(
    ("method", "expected_attrs"),
    [(ProjectionMethod.tsne, {"perplexity": 20 / 3}), ("umap", {"n_neighbors": 15})],
)
def test_project_vectors_small_dataset(
    method: ProjectionMethod, expected_attrs: dict[str, float]
) -> None:
    """t-SNE adjusts perplexity to small datasets, UMAP keeps default n_neighbors."""
    if method == "umap":
        pytest.importorskip("umap")
    xs, _ = make_blobs(n_samples=20, n_features=5, centers=2, random_state=0)

    result, proj_obj = project_vectors(xs, method=method)
    assert result.shape == (len(xs), 2)
    for attr, value in expected_attrs.items():
        assert getattr(proj_obj, attr) == pytest.approx(value)
    assert not np.allclose(result, 0, atol=1e-10)
    assert np.isfinite(result).all()
    assert np.std(result) == pytest.approx(1.0, rel=0.5)


def test_project_vectors_zero_variance_no_nan() -> None:
    """Degenerate data must not yield NaN coords (regression: post-projection
    standardization divided by a zero std).
    """
    projected, _ = project_vectors(np.array([[1.0, 2.0, 3.0]] * 4), n_components=2)
    assert not np.isnan(projected).any()


def test_project_vectors_pca_is_exact_and_keeps_aspect_ratio() -> None:
    """PCA uses an exact SVD and one shared output scale for all axes."""
    rng = np.random.default_rng(seed=0)
    # 636x118 is a shape where sklearn's svd_solver="auto" picks randomized SVD.
    data = rng.normal(size=(636, 118)) * np.linspace(1, 3, 118)
    projected, pca = project_vectors(data, n_components=3)
    assert isinstance(pca, PCA)
    standardized = (data - data.mean(axis=0)) / data.std(axis=0)
    exact = PCA(n_components=3, svd_solver="full").fit_transform(standardized)
    exact /= np.sqrt(np.mean(exact**2))
    np.testing.assert_allclose(np.abs(projected), np.abs(exact), rtol=0, atol=1e-10)
    spread = projected.std(axis=0)
    variance_ratio = pca.explained_variance_ / pca.explained_variance_[0]
    np.testing.assert_allclose(spread / spread[0], np.sqrt(variance_ratio), rtol=1e-10)
    assert spread[0] > spread[1] > spread[2]


def test_project_vectors_isomap_two_samples() -> None:
    """Isomap clamps n_neighbors below n_samples for tiny inputs."""
    data = np.random.default_rng(seed=0).normal(size=(2, 4))
    with pytest.warns(UserWarning, match="adjusted to n_neighbors=1"):
        projected, _ = project_vectors(data, method=ProjectionMethod.isomap)
    assert projected.shape == (2, 2)
