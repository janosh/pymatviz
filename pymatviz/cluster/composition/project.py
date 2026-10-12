"""Chemical composition projection module.

This module provides a unified function for projecting high-dimensional composition
embeddings to lower dimensions for visualization, using various dimensionality reduction
techniques.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap


if TYPE_CHECKING:
    from pymatviz.cluster.composition.plot import ProjectionMethod


def project_vectors(
    data: np.ndarray,
    *,
    method: ProjectionMethod | None = None,
    n_components: int = 2,
    random_state: int | None = 42,
    scale_data: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, PCA | TSNE | Isomap | KernelPCA | Any]:
    """Project high-dimensional data to lower dimensions using various methods.

    Args:
        data (NDArray): Input data array of shape (n_samples, n_features)
        method ("pca" | "tsne" | "umap" | "isomap" | "kernel_pca"): Projection
            method to use (see ProjectionMethod enum)
        n_components (int): Projection dimensions (2 or 3) (default: 2)
        random_state (int | None): Random seed for reproducibility
        scale_data (bool): Whether to standardize input columns before projection
            and rescale the projected coordinates to zero mean and unit overall
            variance (one shared factor for all axes, so aspect ratios are kept)
        **kwargs: Additional arguments passed to the projection method

    Returns:
        tuple[np.array, PCA | TSNE | Isomap | KernelPCA | Any]: A tuple containing:
            - Projected data array of shape (n_samples, n_components)
            - The fitted projection object (PCA, TSNE, UMAP, Isomap, or KernelPCA)

    Raises:
        ValueError: If method is invalid or n_components is too small
        ImportError: If UMAP is requested but not installed
    """
    from pymatviz.cluster.composition.plot import ProjectionMethod

    if method is None:
        method = ProjectionMethod.pca

    if n_components < 2:
        raise ValueError("n_components must be at least 2")

    n_samples = data.shape[0]
    if n_samples < n_components:
        raise ValueError("Not enough samples")

    def standardize(arr: np.ndarray) -> np.ndarray:
        """Standardize columns to zero mean and unit variance, guarding std=0."""
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        # Replace 0s with 1s before division to avoid division by zero
        std = np.where(std == 0, 1, std)
        return (arr - mean) / std

    if scale_data:
        # Handle missing values in standardization, replace remaining NaNs with 0
        data = np.nan_to_num(standardize(data), nan=0.0)

    if method == "pca":
        # Exact SVD by default: svd_solver="auto" silently switches to randomized SVD
        # for e.g. 636x118 inputs, which rotates near-degenerate components.
        pca_kwargs = {"svd_solver": "full"} | kwargs
        reducer = PCA(
            n_components=n_components, random_state=random_state, **pca_kwargs
        )

    elif method == "tsne":
        if n_components > 3:
            raise ValueError("t-SNE visualization typically uses 2 or 3 components")
        # Adjust perplexity for small datasets
        default_perplexity = min(30, n_samples / 3)  # t-SNE default is 30
        tsne_kwargs = {
            "n_components": n_components,
            "perplexity": default_perplexity,
            "learning_rate": "auto",
            "init": "pca",
            "random_state": random_state,
        } | kwargs
        reducer = TSNE(**tsne_kwargs)

    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "UMAP requires the 'umap-learn' package: pip install umap-learn"
            ) from None
        if n_samples < n_components + 2:
            raise ValueError(
                f"UMAP needs at least n_components + 2 = {n_components + 2} samples, "
                f"got {n_samples}"
            )
        reducer = UMAP(
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )

    elif method == "isomap":
        # Adjust n_neighbors for small datasets
        n_neighbors = kwargs.pop("n_neighbors", 5)
        if n_samples < 10 and n_neighbors > n_samples / 2:
            # Isomap needs n_neighbors < n_samples
            n_neighbors = min(max(2, int(n_samples / 2)), n_samples - 1)
            warnings.warn(
                f"Isomap: adjusted to {n_neighbors=} for {n_samples=}",
                UserWarning,
                stacklevel=2,
            )
        reducer = Isomap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            **kwargs,
        )

    elif method == "kernel_pca":
        reducer = KernelPCA(
            n_components=n_components,
            kernel=kwargs.pop("kernel", "rbf"),
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown projection {method=}")

    projected_data = reducer.fit_transform(data)
    if scale_data:
        # One scale for all axes: per-axis standardization would stretch e.g. PC2 to
        # look as spread out as PC1.
        projected_data = projected_data - projected_data.mean(axis=0)
        scale = np.sqrt(np.mean(projected_data**2))
        projected_data = projected_data / (scale if scale > 0 else 1)
    return projected_data, reducer
