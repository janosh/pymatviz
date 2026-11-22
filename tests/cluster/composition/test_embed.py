"""Tests for chemical composition embedding functions."""

import importlib.util
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Composition, Element

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
        elements=[Element.Fe, "O"],  # Element objects and strings
        normalize=normalize,
    )
    np.testing.assert_array_equal(result[0], result[1])
    if not normalize:
        np.testing.assert_array_equal(result, [[0.4, 0.6], [0.4, 0.6]])
    with pytest.raises(ValueError, match=r"contains \['Al'\]"):
        one_hot_encode(["Fe2O3Al"], elements=["Fe", "O"])
    with pytest.raises(ValueError, match="duplicate entries"):
        one_hot_encode(["FeO"], elements=["Fe", "O", "Fe"])


@pytest.mark.parametrize("embed_func", [one_hot_encode, matminer_featurize])
@pytest.mark.parametrize("compositions", [[1, 2, 3], [["H2O"]]])
def test_embed_invalid_input(
    embed_func: Callable[..., np.ndarray], compositions: list[Any]
) -> None:
    """Non-composition inputs raise a TypeError naming the expected types."""
    with pytest.raises(TypeError, match="Expected a formula string, mapping"):
        embed_func(compositions)


@pytest.mark.parametrize(
    "preset", ["magpie", "deml", "matminer", "matscholar_el", "megnet_el"]
)
def test_matminer_featurize(preset: MatminerElementPropertyPreset) -> None:
    """Featurize a Series with each preset (magpie by matscikit, bit-identical to
    matminer), normalized or not, with and without a feature subset.
    """
    compositions = pd.Series(["H2O", "CO2", "NaCl", "Fe2O3"])
    if preset == "magpie":
        import matscikit as mk

        labels = mk.composition_feature_labels()
    else:
        pytest.importorskip("matminer")
        from matminer.featurizers.composition import ElementProperty

        labels = ElementProperty.from_preset(preset).feature_labels()

    raw = matminer_featurize(compositions, preset=preset)
    assert raw.shape == (len(compositions), len(labels))
    assert not np.isnan(raw).any()
    result = matminer_featurize(compositions, preset=preset, normalize=True)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0)
    np.testing.assert_allclose(raw / np.linalg.norm(raw, axis=1)[:, None], result)
    subset = matminer_featurize(
        compositions, preset=preset, feature_subset=labels[1::50]
    )
    np.testing.assert_array_equal(subset, raw[:, 1::50])

    if preset == "magpie" and importlib.util.find_spec("matminer"):
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition

        featurizer = ElementProperty.from_preset("magpie")
        expected = [featurizer.featurize(Composition(comp)) for comp in compositions]
        np.testing.assert_array_equal(raw, np.nan_to_num(expected))


def test_matminer_featurize_invalid_feature_subset() -> None:
    """Unknown feature names raise."""
    compositions = ["H2O", "CO2", "NaCl"]

    with pytest.raises(ValueError, match="None of the requested features"):
        matminer_featurize(
            compositions, preset="magpie", feature_subset=["NonexistentFeature"]
        )


def test_matminer_featurize_n_jobs() -> None:
    """Matminer presets give the same features with 1 or 2 worker processes."""
    pytest.importorskip("matminer")
    compositions = ["H2O", "CO2", "NaCl"]
    result1 = matminer_featurize(compositions, preset="deml", n_jobs=1)
    result2 = matminer_featurize(compositions, preset="deml", n_jobs=2)

    # Results should be the same regardless of n_jobs
    assert np.allclose(result1, result2)
