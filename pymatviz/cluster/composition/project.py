"""Chemical composition projection module.

This module provides a unified function for projecting high-dimensional composition
embeddings to lower dimensions for visualization, using various dimensionality reduction
techniques.
"""

from __future__ import annotations

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
        scale_data (bool): Whether to scale data before projection
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

    if data.shape[0] < n_components:
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
        reducer = PCA(n_components=n_components, random_state=random_state, **kwargs)

    elif method == "tsne":
        if n_components > 3:
            raise ValueError("t-SNE visualization typically uses 2 or 3 components")
        # Adjust perplexity for small datasets
        n_samples = data.shape[0]
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
        reducer = UMAP(
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )

    elif method == "isomap":
        # Adjust n_neighbors for small datasets
        n_samples = data.shape[0]
        n_neighbors = kwargs.pop("n_neighbors", 5)
        if n_samples < 10 and n_neighbors > n_samples / 2:
            n_neighbors = max(2, int(n_samples / 2))
            print(f"Warning: Adjusted to {n_neighbors=} due to small dataset size")  # noqa: T201
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
        projected_data = standardize(projected_data)
    return projected_data, reducer
