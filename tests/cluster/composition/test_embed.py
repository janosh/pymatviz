"""Tests for chemical composition embedding functions."""

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Composition

from pymatviz.cluster.composition import matminer_featurize, one_hot_encode
from pymatviz.cluster.composition.embed import MatminerElementPropertyPreset


@pytest.mark.parametrize(
    "compositions",
    [
        pytest.param(["H2O", "CO2", "NaCl"], id="strings"),
        pytest.param(
            [Composition(comp) for comp in ["H2O", "CO2", "NaCl"]],
            id="compositions",
        ),
        pytest.param(pd.Series(["H2O", "CO2", "NaCl"]), id="series"),
    ],
)
def test_one_hot_encode(compositions: list[str | Composition] | pd.Series) -> None:
    """Encode supported inputs with all elements or a selected subset."""
    result = one_hot_encode(compositions)
    assert result.shape == (3, 118)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, rtol=1e-14, atol=0)

    elements = ["H", "C", "O", "Na", "Cl"]
    result_custom = one_hot_encode(compositions, elements=elements)
    assert result_custom.shape == (len(compositions), len(elements))
    assert result_custom[0, elements.index("H")] > 0
    assert result_custom[0, elements.index("O")] > 0


@pytest.mark.parametrize("normalize", [False, True])
def test_one_hot_encode_oxidation_states(normalize: bool) -> None:
    """Oxidation states do not change elemental fractions or normalized embeddings."""
    result = one_hot_encode(
        ["Fe2O3", Composition({"Fe3+": 2, "O2-": 3})],
        elements=["Fe", "O"],
        normalize=normalize,
    )
    np.testing.assert_array_equal(result[0], result[1])
    if not normalize:
        np.testing.assert_array_equal(result, [[0.4, 0.6], [0.4, 0.6]])


def test_one_hot_encode_invalid_input() -> None:
    """Test one-hot encoding with invalid input."""
    with pytest.raises(ValueError, match="Invalid composition="):
        one_hot_encode([1, 2, 3])  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="Invalid composition="):
        one_hot_encode([["H2O"]])  # ty: ignore[invalid-argument-type]


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
        matminer_featurize([1, 2, 3])  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="Invalid composition="):
        matminer_featurize([["H2O"]])  # ty: ignore[invalid-argument-type]


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
