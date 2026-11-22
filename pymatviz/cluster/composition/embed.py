"""Chemical embedding functions for compositions.

This module provides functions for encoding chemical formulas and compositions
into numerical vectors, which can then be used for clustering and visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pymatviz.composition import to_composition


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from kemio.interop import CompositionLike


def magpie_featurize(
    compositions: Sequence[CompositionLike] | pd.Series,
    *,
    normalize: bool = False,
    feature_subset: Sequence[str] | None = None,
) -> np.ndarray:
    """Magpie element-property features of chemical formulas or compositions
    (matminer's ElementProperty "magpie" preset, computed by
    kemio.composition_features). For other featurizations, pass a feature matrix
    or callable to cluster_compositions.

    Args:
        compositions (Sequence[CompositionLike] | pd.Series): chemical
            formulas as strings or kemio/pymatgen Composition objects.
        normalize (bool): If True, L2-normalize each feature vector (default: False).
            Magpie-style features mix units (e.g. kelvin and eV), so a row norm is
            dominated by the largest-magnitude ones (melting temperature) and divides
            every feature by it; column standardization in project_vectors already
            puts features on a common scale.
        feature_subset (Sequence[str] | None): Feature labels to keep, in this order
            (see kemio.composition_feature_labels). Defaults to all 132.

    Returns:
        np.ndarray: Array of shape (n_compositions, n_features), missing values as 0.

    Raises:
        ValueError: For unknown feature labels.
    """
    import kemio

    features = kemio.composition_features(
        [to_composition(comp) for comp in compositions]
    )
    if feature_subset is not None:
        labels = kemio.composition_feature_labels()
        if unknown := [label for label in feature_subset if label not in labels]:
            raise ValueError(f"Unknown Magpie features {unknown}, expected {labels}")
        features = features[:, [labels.index(label) for label in feature_subset]]
    features = np.nan_to_num(features)
    if normalize:
        import sklearn.preprocessing

        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)
    return features


def one_hot_encode(
    compositions: Sequence[CompositionLike] | pd.Series,
    *,
    normalize: bool = True,
    elements: Sequence[str] | None = None,
) -> np.ndarray:
    """One-hot encode chemical formulas or compositions.

    Args:
        compositions (Sequence[CompositionLike] | pd.Series): chemical
            formulas as strings or kemio/pymatgen Composition objects.
        normalize (bool): Whether to normalize the feature vectors to unit norm
            (default: True)
        elements (Sequence[str] | None): Element symbols to include in the one-hot
            encoding. If None (default), all elements in the periodic table will be
            used. Compositions with elements outside this list raise ValueError.

    Returns:
        np.ndarray: Array of shape (n_compositions, n_elements) with one-hot encoded
            compositions where each row corresponds to a composition and each column
            to an element's fraction in that composition.
    """
    if elements is None:
        from pymatviz.utils import df_ptable

        elements = list(df_ptable.index)
    elements = list(elements)
    column_of = {element: idx for idx, element in enumerate(elements)}
    if len(column_of) != len(elements):
        raise ValueError(f"duplicate entries in {elements=}")

    comp_objs = [to_composition(comp) for comp in compositions]
    features = np.zeros((len(comp_objs), len(elements)))

    for idx, comp in enumerate(comp_objs):
        comp_dict = comp.fractional_composition.as_dict()
        if missing := sorted(set(comp_dict) - set(column_of)):
            raise ValueError(
                f"{comp.formula} contains {missing}, which are not in {elements=}"
            )
        for element, fraction in comp_dict.items():
            features[idx, column_of[element]] = fraction

    if normalize:
        import sklearn.preprocessing

        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)

    return features
