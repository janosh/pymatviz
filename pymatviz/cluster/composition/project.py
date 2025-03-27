"""Chemical composition projection module.

This module provides a unified function for projecting high-dimensional composition
embeddings to lower dimensions for visualization, using various dimensionality reduction
techniques.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap


if TYPE_CHECKING:
    from numpy.typing import NDArray


# Suppress sparse matrix efficiency warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.sparse._index")


def project_vectors(
    data: NDArray,
    *,
    method: Literal["pca", "tsne", "umap", "isomap", "kernel_pca"] = "pca",
    n_components: int = 2,
    random_state: int | None = 42,
    scale_data: bool = True,
    **kwargs: Any,
) -> tuple[NDArray, PCA | TSNE | Isomap | KernelPCA | Any]:
    """Project high-dimensional data to lower dimensions using various methods.

    Args:
        data (NDArray): Input data array of shape (n_samples, n_features)
        method ("pca" | "tsne" | "umap" | "isomap" | "kernel_pca"): Projection
            method to use
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
    if n_components < 2:
        raise ValueError("n_components must be at least 2")

    if data.shape[0] < n_components:
        raise ValueError("Not enough samples")

    if scale_data:
        # Handle missing values in standardization
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        # Replace 0s with 1s before division to avoid division by zero
        std = np.where(std == 0, 1, std)
        data = (data - mean) / std
        # Replace any remaining NaN values with 0
        data = np.nan_to_num(data, nan=0.0)

    if method == "pca":
        pca = PCA(n_components=n_components, random_state=random_state)
        projected_data = pca.fit_transform(data)
        if scale_data:
            projected_data = (
                projected_data - np.mean(projected_data, axis=0)
            ) / np.std(projected_data, axis=0)
        return projected_data, pca

    if method == "tsne":
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
        tsne = TSNE(**tsne_kwargs)
        projected_data = tsne.fit_transform(data)
        if scale_data:
            projected_data = (
                projected_data - np.mean(projected_data, axis=0)
            ) / np.std(projected_data, axis=0)
        return projected_data, tsne

    if method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "UMAP requires the 'umap-learn' package: pip install umap-learn"
            ) from None
        umap_reducer = UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            metric=kwargs.get("metric", "euclidean"),
            random_state=random_state,
        )
        projected_data = umap_reducer.fit_transform(data)
        if scale_data:
            projected_data = (
                projected_data - np.mean(projected_data, axis=0)
            ) / np.std(projected_data, axis=0)
        return projected_data, umap_reducer

    if method == "isomap":
        # Adjust n_neighbors for small datasets
        n_samples = data.shape[0]
        n_neighbors = kwargs.get("n_neighbors", 5)
        if n_samples < 10 and n_neighbors > n_samples / 2:
            n_neighbors = max(2, int(n_samples / 2))
            print(f"Warning: Adjusted to {n_neighbors=} due to small dataset size")  # noqa: T201
        isomap = Isomap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=kwargs.get("metric", "euclidean"),
        )
        projected_data = isomap.fit_transform(data)
        if scale_data:
            projected_data = (
                projected_data - np.mean(projected_data, axis=0)
            ) / np.std(projected_data, axis=0)
        return projected_data, isomap

    if method == "kernel_pca":
        kpca = KernelPCA(
            n_components=n_components,
            kernel=kwargs.get("kernel", "rbf"),
            gamma=kwargs.get("gamma"),
            degree=kwargs.get("degree", 3),
            coef0=kwargs.get("coef0", 1.0),
            random_state=random_state,
        )
        projected_data = kpca.fit_transform(data)
        if scale_data:
            projected_data = (
                projected_data - np.mean(projected_data, axis=0)
            ) / np.std(projected_data, axis=0)
        return projected_data, kpca

    raise ValueError(f"Unknown projection {method=}")
