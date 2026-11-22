"""Tests for chemical composition embedding functions."""

from collections.abc import Callable
from typing import Any

import kemio
import numpy as np
import pandas as pd
import pytest

from pymatviz.cluster.composition import magpie_featurize, one_hot_encode


@pytest.mark.parametrize(
    "compositions",
    [
        pytest.param(["H2O", "CO2", "NaCl"], id="strings"),
        pytest.param(
            [kemio.Composition(comp) for comp in ["H2O", "CO2", "NaCl"]],
            id="compositions",
        ),
        pytest.param(pd.Series(["H2O", "CO2", "NaCl"]), id="series"),
    ],
)
def test_one_hot_encode(
    compositions: list[str | kemio.Composition] | pd.Series,
) -> None:
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
        ["Fe2O3", kemio.Composition({"Fe3+": 2, "O2-": 3})],
        elements=["Fe", "O"],
        normalize=normalize,
    )
    np.testing.assert_array_equal(result[0], result[1])
    if not normalize:
        np.testing.assert_array_equal(result, [[0.4, 0.6], [0.4, 0.6]])
    with pytest.raises(ValueError, match=r"contains \['Al'\]"):
        one_hot_encode(["Fe2O3Al"], elements=["Fe", "O"])
    with pytest.raises(ValueError, match="duplicate entries"):
        one_hot_encode(["FeO"], elements=["Fe", "O", "Fe"])


@pytest.mark.parametrize("embed_func", [one_hot_encode, magpie_featurize])
@pytest.mark.parametrize("compositions", [[1, 2, 3], [["H2O"]]])
def test_embed_invalid_input(
    embed_func: Callable[..., np.ndarray], compositions: list[Any]
) -> None:
    """Non-composition inputs raise a TypeError naming the expected types."""
    with pytest.raises(TypeError, match="Expected a formula string, mapping"):
        embed_func(compositions)


def test_magpie_featurize() -> None:
    """Featurize a Series, normalized or not, with and without a feature subset."""
    compositions = pd.Series(["H2O", "CO2", "NaCl", "Fe2O3"])
    labels = kemio.composition_feature_labels()

    raw = magpie_featurize(compositions)
    assert raw.shape == (len(compositions), len(labels))
    assert not np.isnan(raw).any()
    result = magpie_featurize(compositions, normalize=True)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        raw / np.linalg.norm(raw, axis=1)[:, None], result, rtol=1e-14, atol=0
    )
    subset = magpie_featurize(compositions, feature_subset=labels[1::50])
    np.testing.assert_array_equal(subset, raw[:, 1::50])

    with pytest.raises(ValueError, match=r"Unknown Magpie features \['Nonexistent"):
        magpie_featurize(compositions, feature_subset=["NonexistentFeature"])
