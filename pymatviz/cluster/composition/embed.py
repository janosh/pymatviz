"""Chemical embedding functions for compositions.

This module provides functions for encoding chemical formulas and compositions
into numerical vectors, which can then be used for clustering and visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pymatviz.composition import to_composition


if TYPE_CHECKING:
    from collections.abc import Sequence

    from matscikit.interop import CompositionLike
    from pymatgen.core import Element


MatminerElementPropertyPreset = Literal[
    "magpie", "deml", "matminer", "matscholar_el", "megnet_el"
]


def matminer_featurize(
    compositions: Sequence[CompositionLike] | pd.Series,
    preset: MatminerElementPropertyPreset = "magpie",
    *,
    normalize: bool = False,
    feature_subset: list[str] | None = None,
    n_jobs: int | None = 1,  # Disable multiprocessing by default
) -> np.ndarray:
    """Featurize chemical formulas or compositions with matminer's ElementProperty
    presets (element-property statistics). "magpie" is computed by matscikit
    (bit-identical to matminer, no matminer or pymatgen needed), the other presets by
    matminer.

    Args:
        compositions (Sequence[CompositionLike] | pd.Series): chemical
            formulas as strings or matscikit/pymatgen Composition objects.
        preset (str): ElementProperty preset: "magpie" (default), "deml", "matminer",
            "matscholar_el" or "megnet_el".
        normalize (bool): If True, L2-normalize each feature vector (default: False).
            Magpie-style features mix units (e.g. kelvin and eV), so a row norm is
            dominated by the largest-magnitude ones (melting temperature) and divides
            every feature by it; column standardization in project_vectors already
            puts features on a common scale.
        feature_subset (list[str] | None): List of specific features to use from the
            full feature set. If None (default), all available features will be used.
        n_jobs (int | None): matminer worker processes for non-magpie presets. 1
            (default) disables multiprocessing, None uses all cores.

    Returns:
        np.ndarray: Array of shape (n_compositions, n_features) with featurized
            compositions, missing values as 0.
    """
    comps = [to_composition(comp) for comp in compositions]
    if preset == "magpie":
        import matscikit as mk

        features = mk.composition_features(comps)
        feature_cols = mk.composition_feature_labels()
    else:
        try:
            from matminer.featurizers.composition import ElementProperty
        except ImportError:
            raise ImportError(
                f"{preset} featurization requires pip install matminer"
            ) from None
        from pymatgen.core import Composition  # matminer featurizes pymatgen objects

        featurizer = ElementProperty.from_preset(preset)
        featurizer.set_n_jobs(n_jobs)
        df_comps = pd.DataFrame(
            {"composition": [Composition(c.as_dict()) for c in comps]}
        )
        feature_df = featurizer.featurize_dataframe(df_comps, "composition")
        feature_cols = [col for col in feature_df if col != "composition"]
        features = feature_df[feature_cols].to_numpy()

    if feature_subset is not None:
        if not (valid_cols := [col for col in feature_subset if col in feature_cols]):
            raise ValueError(
                f"None of the requested features {feature_subset} are available. "
                f"{feature_cols=}"
            )
        features = features[:, [feature_cols.index(col) for col in valid_cols]]
    features = np.nan_to_num(features)

    if normalize:
        import sklearn.preprocessing

        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)

    return features


def one_hot_encode(
    compositions: Sequence[CompositionLike] | pd.Series,
    *,
    normalize: bool = True,
    elements: Sequence[str | Element] | None = None,
) -> np.ndarray:
    """One-hot encode chemical formulas or compositions.

    Args:
        compositions (Sequence[CompositionLike] | pd.Series): chemical
            formulas as strings or matscikit/pymatgen Composition objects.
        normalize (bool): Whether to normalize the feature vectors to unit norm
            (default: True)
        elements (Sequence[str | Element] | None): Elements to include in the one-hot
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
    elements = list(map(str, elements))  # also pymatgen Element objects
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
