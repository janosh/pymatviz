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
    compositions: Sequence[str] | Sequence[Composition] | pd.Series,
    preset: MatminerElementPropertyPreset = "magpie",
    *,
    normalize: bool = True,
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
        normalize (bool): If True (default), normalize the feature vectors to unit norm.
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

    # Convert compositions to pymatgen Composition objects
    comp_objs = [try_composition(comp) for comp in compositions]

    # Initialize featurizer with the specified preset
    featurizer = ElementProperty.from_preset(preset)
    featurizer.set_n_jobs(n_jobs)

    # Create a DataFrame for matminer featurizers
    df_comps = pd.DataFrame({"composition": comp_objs})

    # Apply featurization
    feature_df = featurizer.featurize_dataframe(df_comps, "composition")

    # Get the feature columns (skip the composition column)
    feature_cols = [col for col in feature_df if col != "composition"]

    # Filter to subset if requested
    if feature_subset is not None:
        valid_cols = [col for col in feature_subset if col in feature_cols]
        if not valid_cols:
            raise ValueError(
                f"None of the requested features {feature_subset} are available. "
                f"{feature_cols=}"
            )
        feature_cols = valid_cols

    # Extract features as numpy array
    features = feature_df[feature_cols].to_numpy()

    # Handle NaN values that might be present
    features = np.nan_to_num(features)

    # Normalize vectors if requested
    if normalize:
        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)

    return features


def one_hot_encode(
    compositions: Sequence[str] | Sequence[Composition] | pd.Series,
    *,
    normalize: bool = True,
    elements: Sequence[str] | None = None,
) -> np.ndarray:
    """One-hot encode chemical formulas or compositions.

    Args:
        compositions (Sequence[str] | Sequence[Composition] | pd.Series): chemical
            formulas as strings or pymatgen Composition objects.
        normalize (bool): Whether to normalize the feature vectors to unit norm
            (default: True)
        elements (Sequence[str] | None): Elements to include in the one-hot encoding.
            If None (default), all elements in the periodic table will be used.

    Returns:
        np.ndarray: Array of shape (n_compositions, n_elements) with one-hot encoded
            compositions where each row corresponds to a composition and each column
            to an element's fraction in that composition.
    """
    if elements is None:
        # All elements in the periodic table
        elements = list(map(str, Element))

    # Convert compositions to pymatgen Composition objects
    comp_objs = [try_composition(comp) for comp in compositions]

    # Initialize feature matrix of zeros
    n_compositions = len(comp_objs)
    n_elements = len(elements)
    features = np.zeros((n_compositions, n_elements))

    # Fill the feature matrix with element fractions
    for idx, comp in enumerate(comp_objs):
        # Get fractional amounts for each element
        comp_dict = comp.fractional_composition.as_dict()
        for element, fraction in comp_dict.items():
            if element in elements:
                # Get the index of the element in our elements list
                element_idx = elements.index(element)
                # Use the fractional composition as features
                features[idx, element_idx] = fraction

    # Normalize vectors if requested
    if normalize:
        features = sklearn.preprocessing.normalize(features, norm="l2", axis=1)

    return features
