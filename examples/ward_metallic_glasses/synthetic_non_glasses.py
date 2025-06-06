"""Helper functions for generating negative examples of non-glass-forming compositions
to turn the Ward et al. dataset into a more balanced classification task.
"""

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from ward_metallic_glasses import formula_features

import pymatviz as pmv
from pymatviz.enums import Key


def make_rand_binaries(
    n_samples: int,
    elements: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Generate random binary compositions with integer percentages summing to 100.

    Args:
        n_samples (int): Number of compositions to generate
        elements (list[str] | None): element set to choose from. If None, uses all
            elements in pmv.df_ptable.
        rng (np.random.Generator | None): Random number generator. If None, creates a
            new one with fixed seed.

    Returns:
        List of composition strings like ["Cu35Au65", "Fe70Ni30"]
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)
    if elements is None:
        elements = list(pmv.df_ptable.index)

    compositions = []
    for _ in range(n_samples):
        # Choose two random elements
        el1, el2 = rng.choice(elements, size=2, replace=False)
        # Random integer percentage for first element between 1 and 99
        pct1 = rng.integers(1, 100)
        pct2 = 100 - pct1
        compositions.append(f"{el1}{pct1}{el2}{pct2}")

    return compositions


def make_rand_ternaries_similar_radii(
    n_samples: int,
    elements: list[str] | None = None,
    max_radius_diff_percent: float = 5.0,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Generate random ternary compositions with similar atomic radii.

    Args:
        n_samples (int): Number of compositions to generate
        elements (list[str] | None): Element set to choose from. If None, uses all
            elements in pmv.df_ptable.
        max_radius_diff_percent (float): Max allowed difference in atomic radii between
            any 2 elements in the ternary composition, as percentage of smaller radius.
        rng (np.random.Generator | None): Random number generator. If None, creates a
            new one with fixed seed.

    Returns:
        List of composition strings like ["Cu33Fe33Ni34", "Al30Ti35V35"]
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)
    if elements is None:
        elements = list(pmv.df_ptable.index)

    # Get atomic radii for all elements
    radii = pmv.df_ptable[Key.covalent_radius].fillna(0.2)  # fallback value of 0.2 nm

    compositions: list[str] = []
    max_attempts = n_samples * 100  # avoid infinite loop if constraints too strict
    attempts = 0

    while len(compositions) < n_samples and attempts < max_attempts:
        # Choose three random elements
        el1, el2, el3 = rng.choice(elements, size=3, replace=False)

        # Check if atomic radii are similar enough
        r1, r2, r3 = radii[el1], radii[el2], radii[el3]
        min_radius = min(r1, r2, r3)

        # Calculate percentage differences
        diffs = [
            abs(r1 - r2) / min_radius * 100,
            abs(r2 - r3) / min_radius * 100,
            abs(r3 - r1) / min_radius * 100,
        ]

        if max(diffs) <= max_radius_diff_percent:
            # Generate random percentages that sum to 100
            pct1 = rng.integers(20, 40)  # ensure each element is at least 20%
            pct2 = rng.integers(20, 60 - pct1)
            pct3 = 100 - pct1 - pct2
            compositions.append(f"{el1}{pct1}{el2}{pct2}{el3}{pct3}")

        attempts += 1

    if len(compositions) < n_samples:
        print(
            f"Warning: Could only generate {len(compositions)} compositions with "
            f"atomic radius differences <= {max_radius_diff_percent}% after "
            f"{attempts} attempts"
        )

    return compositions


def add_random_negative_examples(
    data_df: pd.DataFrame,
    df_feat: pd.DataFrame,
    n_binary_samples: int,
    n_ternary_samples: int,
    max_radius_diff_percent: float = 5.0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add random binary + ternary compositions as negative examples for classification.

    Args:
        data_df (pd.DataFrame): Must have a 'composition' column.
        df_feat (pd.DataFrame): Must have a 'composition' column.
        n_binary_samples (int): Number of random binary compositions to add.
        n_ternary_samples (int): Number of random ternary compositions with similar
            atomic radii to add.
        max_radius_diff_percent (float): Max allowed difference in atomic radii between
            any 2 elements in the ternary composition, as a percentage of the smaller
            radius.
        rng (np.random.Generator | None): Random number generator. If None, creates a
            new one with fixed seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Augmented data_df and df_feat.
    """
    # Get list of elements present in the training set
    training_elements = []
    for comp in data_df[Key.composition]:
        training_elements.extend(Composition(comp).chemical_system_set)
    training_elements = sorted(set(training_elements))

    # Generate random binary compositions
    binary_comps = make_rand_binaries(
        n_binary_samples, elements=training_elements, rng=rng
    )

    # Generate random ternary compositions with similar atomic radii
    ternary_comps = make_rand_ternaries_similar_radii(
        n_ternary_samples,
        elements=training_elements,
        max_radius_diff_percent=max_radius_diff_percent,
        rng=rng,
    )

    # Create DataFrame with new compositions
    df_random = pd.DataFrame(  # all random compositions are assumed non-glass-forming
        {Key.composition: binary_comps + ternary_comps, "is_glass": "crystal"}
    )

    # Featurize new compositions
    df_random_feat = formula_features.one_hot_encode(df_random)

    # Create a DataFrame with all columns from df_feat, filled with zeros
    df_random_feat_full = pd.DataFrame(
        0.0, index=df_random_feat.index, columns=df_feat.columns
    )

    # Fill in the element composition columns we have
    common_cols = df_random_feat.columns.intersection(df_feat.columns)
    df_random_feat_full[common_cols] = df_random_feat[common_cols]

    # Create new indices with 'neg-' prefix
    n_random = len(df_random)
    df_random.index = [f"synthetic-non-gfa-{idx + 1}" for idx in range(n_random)]
    df_random_feat_full.index = df_random.index

    # Combine with original data
    data_df_aug = pd.concat([data_df, df_random])
    df_feat_aug = pd.concat([df_feat, df_random_feat_full])

    return data_df_aug, df_feat_aug
