"""Functions for calculating compositional features from chemical formulas.

This module provides functions to calculate various compositional features used in
materials science, particularly for predicting glass-forming ability. Features include:

1. Atomic size difference
2. Mixing enthalpy
3. Binary liquidus temperature

The features are based on Liu et al. (2023) https://doi.org/10.1016/j.actamat.2022.118497
and other literature sources.
"""

import os
import pathlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from matminer.featurizers.composition.alloy import MixingEnthalpy
from pymatgen.core import Composition, Element
from scipy.constants import convert_temperature
from scipy.interpolate import interp1d

from pymatviz.enums import Key


def load_binary_liquidus_data(zip_path: str) -> dict[str, interp1d]:
    """Load binary liquidus temperature data from a ZIP archive of CSV files.

    Args:
        zip_path (str): Path to ZIP archive containing binary liquidus temperature data.
            Each CSV file should be named after the chemical system (e.g., "Al-Cu.csv")
            and contain columns for composition and temperature.

    Returns:
        dict[str, interp1d]: Map of chemical system (e.g., "Al-Cu") to interpolation
            function that takes composition fraction and returns liquidus temperature.
    """
    from zipfile import ZipFile

    amended_chemsys: dict[str, dict[str, float]] = defaultdict(dict)
    binary_interpolations = {}

    with ZipFile(zip_path) as zip_file:
        for filename in zip_file.namelist():
            if not (
                filename.startswith("binary_liquidus/") and filename.endswith(".csv")
            ):
                continue

            chemsys = Composition(pathlib.Path(filename).stem).chemical_system
            # read CSV directly from zip using pandas
            with zip_file.open(filename) as csv_file:
                df_binary = pd.read_csv(csv_file, header=None)

            # drop rows with na values
            df_binary = df_binary.dropna()
            df_binary[7] = [
                convert_temperature(temp, "Celsius", "Kelvin")
                for temp in df_binary[7].astype(float)
            ]

            if not np.isclose(df_binary[5] + df_binary[6], 1.0).all():
                raise ValueError(f"Composition sum is not 1.0 for {chemsys}")

            # Some of the tabulated data is missing the terminal values for the wt
            # fraction of the binary component. We add these here using the melting
            # points of the unary elements retrieved from pymatgen. NOTE that this
            # may cause the reporting of unphysical temperatures if the missing regions
            # are large or contain non-trivial structures (e.g. a eutectic point).
            if not np.isclose(df_binary.loc[0, 6], 0.0):
                new_row = df_binary.iloc[0].copy()
                new_row[5] = 1.0
                new_row[6] = 0.0
                unary = Element(new_row[1])
                new_row[7] = unary.melting_point
                print(f"Adding unary {unary.name} to {chemsys}")
                df_binary = pd.concat(
                    [new_row.to_frame().T, df_binary], ignore_index=True
                )
                amended_chemsys[chemsys][unary.name] = unary.melting_point
            if not np.isclose(df_binary.loc[df_binary.index[-1], 6], 1.0):
                new_row = df_binary.iloc[df_binary.index[-1]].copy()
                new_row[5] = 0.0
                new_row[6] = 1.0
                unary = Element(new_row[2])
                new_row[7] = unary.melting_point
                print(f"Adding unary {unary.name} to {chemsys}")
                df_binary = pd.concat(
                    [df_binary, new_row.to_frame().T], ignore_index=True
                )
                amended_chemsys[chemsys][unary.name] = unary.melting_point
            if chemsys in amended_chemsys:
                df_binary[5] = pd.to_numeric(df_binary[5])
                df_binary[6] = pd.to_numeric(df_binary[6])
                df_binary[7] = pd.to_numeric(df_binary[7])

            binary_interpolations[chemsys] = interp1d(
                df_binary[6], df_binary[7], kind="linear"
            )

    return binary_interpolations


def calc_reduced_binary_liquidus_temp(
    composition: Composition,
    binary_interpolations: dict[str, interp1d],
    *,
    on_key_err: Literal["raise", "set-none"] = "set-none",
) -> float:
    """Calculate the reduced average binary liquidus temperature for a general alloy.

    NOTE the unary melting points from the tabulated data are not used here as
    they are not consistent through the tabulated data. Instead we make use of
    the tabulated values available in pymatgen for the unary elements melting
    points.

    Args:
        composition (Composition): For which to calculate the reduced binary
            liquidus temperature. This composition should be expressed in %wt.
        binary_interpolations (dict[str, interp1d]): The binary liquidus temperature
            interpolations.
        on_key_err ("raise" | "set-none"): How to handle missing binary
            systems.
            If "raise", raises KeyError. If "set-none", returns None.
            Defaults to "raise".

    Returns:
        float: The reduced binary liquidus temperature or None if on_key_err="set-none"
            and a binary system is missing.
    """
    if len(composition) < 2:
        return 1.0

    temp_alloy = 0.0
    temp_alloy_norm = 0.0
    for binary_pair in combinations(composition, 2):
        comp_dict = {el: composition[el] for el in binary_pair}
        binary_composition = Composition(comp_dict).fractional_composition
        chemsys = binary_composition.chemical_system
        try:
            temp_binary = binary_interpolations[chemsys](
                binary_composition[chemsys.split("-")[0]]
            )
        except KeyError:
            if on_key_err == "raise":
                raise
            return None  # type: ignore[return-value]
        binary_weight = sum(comp_dict.values())
        temp_alloy += temp_binary * binary_weight
        temp_alloy_norm += binary_weight

    temp_alloy /= temp_alloy_norm

    # Calculate the mean liquidus temperature among the constituent elements
    temp_mean = sum(
        amt * el.melting_point for el, amt in composition.fractional_composition.items()
    )

    # Calculate the reduced binary liquidus
    return temp_alloy / temp_mean


def calc_atomic_size_difference(composition: Composition) -> float:
    """Calculate the atomic size difference for a general alloy.

    NOTE: the paper doesn't describe the use of the absolute value but our
    implementation uses the absolute value based on observation that we would
    otherwise see cancellation leading to very small features and potentially
    imaginary features if the mean radius deviation is negative.

    Args:
        composition (Composition): For which to calculate the atomic size difference.
            This composition should be expressed in %at.

    Returns:
        float: The atomic size difference.
    """
    if composition.is_element:
        return 0

    amounts = np.array(list(composition.fractional_composition.values()))
    radii = np.array([el.atomic_radius for el in composition.elements])
    mean_radius = np.average(radii, weights=amounts)

    return ((np.average((1 - radii / mean_radius) ** 2, weights=amounts)) ** 0.5) * 100


def calc_miedema_maximum_heat_of_mixing(composition: Composition) -> float | None:
    """Calculate the maximum heat of mixing feature for a general alloy.

    The assumption behind this feature is that the largest magnitude heat of
    mixing is the most relevant for working out the behavior of the alloy.
    This value is then scaled by a weighting factor to account for the fraction
    of the bonding expected to be from the pairing considered.

    NOTE this feature is fairly questionable as it's not robust to adversarial
    perturbations.
    """
    if len(composition) < 2:
        return 0

    mixing_enthalpy = MixingEnthalpy(impute_nan=True)

    delta_h_max = delta_h_best = 0
    binary_fracs_max = [0, 0]  # Initialize with zeros
    for binary_pair in combinations(composition, 2):
        binary_fracs = [composition[el] for el in binary_pair]
        delta_h = mixing_enthalpy.get_mixing_enthalpy(*binary_pair)
        if np.abs(delta_h) > delta_h_max:
            delta_h_max = np.abs(delta_h)
            delta_h_best = delta_h
            binary_fracs_max = binary_fracs

    if np.sum(binary_fracs_max) == 0:
        return None
    return delta_h_best * 2 * np.prod(binary_fracs_max) / np.sum(binary_fracs_max)


def calc_liu_features(
    formulas: str | Composition | Sequence[str | Composition],
    include: Sequence[str] = (),
    binary_liquidus_data: dict[str, interp1d] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Calculate Liu et al.'s (2023) compositional features for metallic glasses.

    Args:
        formulas (Sequence[str] | Composition | Sequence[Composition]):
            Chemical formulas or pymatgen Compositions to calculate features for.
            If strings, they should be valid chemical formulas like "Fe2O3" or
            "LiPO4".
        include (tuple[str, ...]): The features to include. If empty, all features
            are included.
        binary_liquidus_data: Dictionary mapping chemical system (e.g., "Al-Cu") to
            interpolation function for liquidus temperature. If None, liquidus
            temperature feature will not be calculated.

    Returns:
        Nested dict mapping formula to feature dict with keys:
        - "atomic_size_diff": Atomic size difference (%)
        - "mixing_enthalpy": Maximum mixing enthalpy (kJ/mol)
        - "liquidus_temp": Reduced binary liquidus temperature (dimensionless)

    Example:
        >>> calc_liu_features("Fe2O3")
        {'Fe2O3': {'atomic_size_diff': 23.45, 'mixing_enthalpy': -12.34,
                   'liquidus_temp': 0.89}}
    """
    if isinstance(formulas, str | Composition):
        formulas = [formulas]

    results = {}
    feature_funcs: dict[str, Callable[[Any], float | None]] = {
        "mixing_enthalpy": calc_miedema_maximum_heat_of_mixing,
        "atomic_size_diff": calc_atomic_size_difference,
    }
    if binary_liquidus_data is not None:
        feature_funcs["liquidus_temp"] = lambda comp: calc_reduced_binary_liquidus_temp(
            comp, binary_interpolations=binary_liquidus_data
        )

    for key, func in feature_funcs.items():
        if include and key not in include:
            continue
        results[key] = {comp: func(Composition(comp)) for comp in formulas}

    return results


def liu_featurize(
    df_in: pd.DataFrame,
    binary_liquidus_data: dict[str, interp1d] | None = None,
) -> pd.DataFrame:
    """Calculate Liu et al.'s (2023) features for a DataFrame of compositions.

    Args:
        df_in (pd.DataFrame): Must have a 'composition' column with chemical formulas.
        binary_liquidus_data (dict[str, interp1d]): Map of chemical system
            (e.g. "Al-Cu") to interpolation function for liquidus temperature. If None,
            liquidus temperature feature will not be calculated.

    Returns:
        pd.DataFrame: with original columns plus Liu features
    """
    df_out = df_in.copy()
    df_out[Key.composition] = df_out[Key.composition].map(Composition)

    for col_name, func in {
        "mixing_enthalpy": calc_miedema_maximum_heat_of_mixing,
        "atomic_size_diff": calc_atomic_size_difference,
    }.items():
        df_out[col_name] = df_out[Key.composition].map(func)

    if binary_liquidus_data is not None:
        df_out["liquidus_temp"] = df_out[Key.composition].map(
            lambda comp: calc_reduced_binary_liquidus_temp(
                comp, binary_interpolations=binary_liquidus_data
            )
        )

    return df_out


def one_hot_encode(df_in: pd.DataFrame) -> pd.DataFrame:
    """Convert composition strings to one-hot encoded element weights.

    Creates a feature matrix where each column represents an element and each row
    contains the fractional amount of that element in the composition. Elements not
    present in a composition get 0.0.

    Args:
        df_in (pd.DataFrame): Must have a 'composition' column with strings like 'Fe2O3'

    Returns:
        pd.DataFrame: with the original columns plus one column per element found across
        all compositions. The element columns contain the fractional amounts.
    """
    comp_dicts = df_in[Key.composition].map(lambda comp: Composition(comp).as_dict())
    all_elements = sorted({el for comp in comp_dicts for el in comp})
    comp_features = pd.DataFrame(0.0, index=df_in.index, columns=all_elements)
    for idx, comp in comp_dicts.items():
        for el, wt in comp.items():
            comp_features.loc[idx, el] = wt
    return pd.concat([df_in, comp_features], axis=1)


if __name__ == "__main__":
    # Example usage and validation
    module_dir = os.path.dirname(__file__)
    zip_path = f"{module_dir}/binary-liquidus-temperatures.zip"

    # Load binary liquidus data
    binary_interpolations = load_binary_liquidus_data(zip_path)

    # Test with a simple binary composition
    test_comp = Composition("Pt50P50")
    features = calc_liu_features(test_comp, binary_liquidus_data=binary_interpolations)
    print("\nFeatures for Pt50P50:")
    for feature, values in features.items():
        print(f"{feature}: {values[test_comp]:.2f}")

    # Test with a more complex composition
    test_comp2 = "Zr6.2Ti45.8Cu39.9Ni5.1Sn3"
    features2 = calc_liu_features(
        test_comp2, binary_liquidus_data=binary_interpolations
    )
    print(f"\nFeatures for {test_comp2}:")
    for feature, values in features2.items():
        print(f"{feature}: {values[test_comp2]:.2f}")

    # Example of batch processing with a DataFrame
    df_test = pd.DataFrame(
        {
            Key.composition: [
                "Zr6.2Ti45.8Cu39.9Ni5.1Sn3",
                "Zr6.2Ti45.8Cu39.9Ni5.1Sn2Si1",
                "Fe70Ga30",
            ]
        }
    )
    df_features = liu_featurize(df_test, binary_liquidus_data=binary_interpolations)
    print("\nBatch processing results:")
    print(df_features)
