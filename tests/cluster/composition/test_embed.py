"""Tests for chemical composition embedding functions."""

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Composition

from pymatviz.cluster.composition import matminer_featurize, one_hot_encode
from pymatviz.cluster.composition.embed import MatminerElementPropertyPreset


@pytest.mark.parametrize(
    ("compositions", "expected_shape"),
    [
        (["H2O", "CO2", "NaCl"], (3, 118)),  # Test with formula strings
        # Test with Composition objects
        ([Composition(comp) for comp in ["H2O", "CO2", "NaCl"]], (3, 118)),
    ],
)
def test_one_hot_encode(
    compositions: list[str | Composition], expected_shape: tuple[int, int]
) -> None:
    """Test one-hot encoding of chemical formulas."""
    # Default elements list contains all elements in the periodic table
    result = one_hot_encode(compositions)

    # Check shape
    assert result.shape == expected_shape

    # Check if values are normalized
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    # Test with a custom elements list
    elements = ["H", "C", "O", "Na", "Cl"]
    result_custom = one_hot_encode(compositions, elements=elements)

    # Check custom shape
    assert result_custom.shape == (len(compositions), len(elements))

    # Basic check for H2O: should have non-zero values for H and O
    # Find the index of H2O
    h2o_idx = (
        compositions.index("H2O")
        if isinstance(compositions[0], str)
        else compositions.index(Composition("H2O"))
    )

    # Get element indices
    if elements is None:
        # Get from periodic table - H is element 1, O is element 8
        h_idx = 0  # 0-indexed
        o_idx = 7  # 0-indexed
    else:
        h_idx = elements.index("H")
        o_idx = elements.index("O")

    # Check if H and O have non-zero values for H2O
    assert result_custom[h2o_idx, h_idx] > 0
    assert result_custom[h2o_idx, o_idx] > 0


def test_one_hot_encode_log_transform() -> None:
    """Test log transform option in one-hot encoding."""
    compositions = ["H2O", "CO2", "NaCl"]

    # Test with log transform (without normalization)
    result_log = one_hot_encode(compositions, log_transform=True, normalize=False)

    # Test without log transform (without normalization)
    result_no_log = one_hot_encode(compositions, log_transform=False, normalize=False)

    # Check that the two results are different
    assert not np.allclose(result_log, result_no_log)

    # Check that log transform values are smaller (since log(1+x) < x for x > 0)
    assert np.all(result_log <= result_no_log)

    # Check that log transform preserves zeros
    assert np.all((result_no_log == 0) == (result_log == 0))


def test_one_hot_encode_pandas_input() -> None:
    """Test one-hot encoding with pandas Series input."""
    compositions = pd.Series(["H2O", "CO2", "NaCl"])
    result = one_hot_encode(compositions)

    # Check shape and normalization
    assert result.shape == (3, 118)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_one_hot_encode_invalid_input() -> None:
    """Test one-hot encoding with invalid input."""
    with pytest.raises(ValueError, match="Invalid composition="):
        one_hot_encode([1, 2, 3])

    with pytest.raises(ValueError, match="Invalid composition="):
        one_hot_encode([["H2O"]])


@pytest.mark.parametrize(
    "preset", ["magpie", "deml", "matminer", "matscholar_el", "megnet_el"]
)
def test_matminer_featurize(preset: MatminerElementPropertyPreset) -> None:
    """Test matminer featurization with different presets."""
    pytest.importorskip("matminer")
    from matminer.featurizers.composition import ElementProperty

    compositions = ("H2O", "CO2", "NaCl")

    # Test basic functionality
    result = matminer_featurize(compositions, preset=preset, normalize=True)
    first_fea_col = ElementProperty.from_preset(preset).feature_labels()[0]
    subset = [first_fea_col]

    # Check that we got valid output
    assert result.shape[0] == len(compositions)
    assert not np.isnan(result).any()

    # Check normalization
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    # Test with a feature subset
    result_subset = matminer_featurize(
        compositions, preset=preset, feature_subset=subset
    )

    # Check that the subset has the right shape
    assert result_subset.shape == (len(compositions), len(subset))


def test_matminer_featurize_pandas_input() -> None:
    """Test matminer featurization with pandas Series input."""
    pytest.importorskip("matminer")
    compositions = pd.Series(["H2O", "CO2", "NaCl"])
    result = matminer_featurize(compositions)

    # Check shape and no NaN values
    assert result.shape[0] == len(compositions)
    assert not np.isnan(result).any()


def test_matminer_featurize_invalid_input() -> None:
    """Test matminer featurization with invalid input."""
    pytest.importorskip("matminer")
    with pytest.raises(ValueError, match="Invalid composition="):
        matminer_featurize([1, 2, 3])

    with pytest.raises(ValueError, match="Invalid composition="):
        matminer_featurize([["H2O"]])


def test_matminer_featurize_invalid_feature_subset() -> None:
    """Test matminer featurization with invalid feature subset."""
    pytest.importorskip("matminer")
    compositions = ["H2O", "CO2", "NaCl"]

    with pytest.raises(ValueError, match="None of the requested features"):
        matminer_featurize(
            compositions, preset="magpie", feature_subset=["NonexistentFeature"]
        )


def test_matminer_featurize_n_jobs() -> None:
    """Test matminer featurization with different n_jobs values."""
    pytest.importorskip("matminer")
    compositions = ["H2O", "CO2", "NaCl"]

    # Test with n_jobs=1 (default)
    result1 = matminer_featurize(compositions, n_jobs=1)

    # Test with n_jobs=2
    result2 = matminer_featurize(compositions, n_jobs=2)

    # Results should be the same regardless of n_jobs
    assert np.allclose(result1, result2)


def test_magpie_preset() -> None:
    """Test using matminer_featurize with magpie preset."""
    pytest.importorskip("matminer")

    compositions = ["H2O", "CO2", "NaCl"]

    # Test using preset="magpie"
    result = matminer_featurize(compositions, preset="magpie")

    # Check that we got valid output
    assert result.shape[0] == len(compositions)
    assert not np.isnan(result).any()
