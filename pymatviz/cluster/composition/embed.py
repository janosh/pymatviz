"""Chemical embedding functions for compositions.

This module provides functions for encoding chemical formulas and compositions
into numerical vectors, which can then be used for clustering and visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import sklearn.preprocessing
from pymatgen.core import Composition, Element


if TYPE_CHECKING:
    from collections.abc import Sequence


MatminerElementPropertyPreset = Literal[
    "magpie", "deml", "matminer", "matscholar_el", "megnet_el"
]


def try_composition(composition: str | Composition) -> Composition:
    """Try to convert a string or Composition to a Composition."""
    try:
        return Composition(composition)
    except Exception:  # noqa: BLE001
        raise ValueError(f"Invalid {composition=}") from None


def matminer_featurize(
    compositions: Sequence[str | Composition] | pd.Series,
    preset: MatminerElementPropertyPreset = "magpie",
    *,
    normalize: bool = False,
    feature_subset: list[str] | None = None,
    n_jobs: int | None = 1,  # Disable multiprocessing by default
) -> np.ndarray:
    """Apply matminer featurization to chemical formulas or compositions.

    This uses matminer's ElementProperty featurizer with the specified preset,
    which includes physical and chemical properties of elements.

    Args:
        compositions (Sequence[str] | Sequence[Composition] | pd.Series): chemical
            formulas as strings or pymatgen Composition objects.
        preset (str): Matminer preset to use. Options: "magpie", "deml", "matminer",
            "matscholar_el", "megnet_el" (default: "magpie").
        normalize (bool): If True, L2-normalize each feature vector (default: False).
            Magpie-style features mix units (e.g. kelvin and eV), so a row norm is
            dominated by the largest-magnitude ones (melting temperature) and divides
            every feature by it; column standardization in project_vectors already
            puts features on a common scale.
        feature_subset (list[str] | None): List of specific features to use from the
            full feature set. If None (default), all available features will be used.
        n_jobs (int | None): Number of jobs to run in parallel. Set to 1 to disable
            multiprocessing (default). Set to None to use all available cores.

    Returns:
        np.ndarray: Array of shape (n_compositions, n_features) with featurized
            compositions.
    """
    try:
        from matminer.featurizers.composition import ElementProperty
    except ImportError:
        raise ImportError(
            f"{preset} featurization requires pip install matminer"
        ) from None

    comp_objs = [try_composition(comp) for comp in compositions]

    featurizer = ElementProperty.from_preset(preset)
    featurizer.set_n_jobs(n_jobs)

    df_comps = pd.DataFrame({"composition": comp_objs})

    feature_df = featurizer.featurize_dataframe(df_comps, "composition")

    feature_cols = [col for col in feature_df if col != "composition"]

    if feature_subset is not None:
        valid_cols = [col for col in feature_subset if col in feature_cols]
        if not valid_cols:
            raise ValueError(
                f"None of the requested features {feature_subset} are available. "
                f"{feature_cols=}"
            )
        feature_cols = valid_cols

    features = np.nan_to_num(feature_df[feature_cols].to_numpy())

    if normalize:
        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)

    return features


def one_hot_encode(
    compositions: Sequence[str | Composition] | pd.Series,
    *,
    normalize: bool = True,
    elements: Sequence[str | Element] | None = None,
) -> np.ndarray:
    """One-hot encode chemical formulas or compositions.

    Args:
        compositions (Sequence[str] | Sequence[Composition] | pd.Series): chemical
            formulas as strings or pymatgen Composition objects.
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
    elements = list(map(str, Element if elements is None else elements))
    column_of = {element: idx for idx, element in enumerate(elements)}
    if len(column_of) != len(elements):
        raise ValueError(f"duplicate entries in {elements=}")

    comp_objs = [try_composition(comp) for comp in compositions]
    features = np.zeros((len(comp_objs), len(elements)))

    for idx, comp in enumerate(comp_objs):
        comp_dict = comp.element_composition.fractional_composition.as_dict()
        if missing := sorted(set(comp_dict) - set(column_of)):
            raise ValueError(
                f"{comp.formula} contains {missing}, which are not in {elements=}"
            )
        for element, fraction in comp_dict.items():
            features[idx, column_of[element]] = fraction

    if normalize:
        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)

    return features
