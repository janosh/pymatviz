"""Test chemical environment analysis utilities."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from pymatviz import chem_env


@pytest.mark.parametrize(
    ("symbol", "mapping", "expected_cn"),
    [
        ("T:4", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 4),
        ("PP:5", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 5),
        ("X:Y:7", {"CUSTOM:3": 3, "X:Y:7": 7}, 7),
        ("S:1", {}, 1),
        ("S:1", {"other": 10}, 1),
        ("M:1", {}, 1),
        ("M:8", {"T:4": 4}, 8),
        ("M:100", {}, 100),
        ("NULL", {"T:4": 4}, 0),
        ("UNKNOWN", {}, 0),
        ("O:6", {"T:4": 4}, 0),
        ("M:", {}, 0),
        ("M:abc", {}, 0),
        ("M:3.5", {}, 0),
        ("M:-5", {}, 0),
        ("", {}, 0),
        (":", {}, 0),
        ("T:", {}, 0),
        ("123", {}, 0),
        ("T:4:extra", {"T:4": 4}, 0),
    ],
)
def test_get_cn_from_symbol(
    symbol: str, mapping: dict[str, int], expected_cn: int
) -> None:
    """Coordination numbers come from the mapping, S:1, or M:<n> symbols."""
    assert chem_env.get_cn_from_symbol(symbol, mapping) == expected_cn


@pytest.mark.parametrize(
    ("site_idx", "cn_val", "expected"),
    [
        (0, 0, "CN:0"),
        (0, -1, "CN:-1"),
        (0, 99, "CN:99"),
        (999, 6, IndexError),
        (-1, 4, IndexError),
    ],
)
def test_classify_local_env_with_order_params_cubic(
    structures: tuple[Structure, Structure],
    site_idx: int,
    cn_val: int,
    expected: str | type[Exception],
) -> None:
    """Unsupported CNs get a generic CN label, out-of-range sites raise."""
    if isinstance(expected, str):
        result = chem_env.classify_local_env_with_order_params(
            structures[0], site_idx, cn_val
        )
        assert result == expected
    else:
        with pytest.raises(expected, match=rf"site_idx={site_idx} out of range"):
            chem_env.classify_local_env_with_order_params(
                structures[0], site_idx, cn_val
            )


@pytest.mark.parametrize(
    ("order_params", "expected"),
    [([0.0, 1.0], "T:4"), ([0.0, 0.5], "CN:4")],
    ids=["best-match", "weak-match-fallback"],
)
def test_classify_local_env_order_params(
    monkeypatch: pytest.MonkeyPatch, order_params: list[float], expected: str
) -> None:
    """Order params use the requested site's neighbors; weak matches fall back."""
    from pymatgen.analysis import local_env

    structure = Structure(
        Lattice.cubic(4), ["Li", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]]
    )

    class FakeCrystalNN:
        """Nearest-neighbor finder that only answers for site 1."""

        def get_nn_info(self, structure: Structure, n: int) -> list[dict[str, object]]:
            """Return the two other sites as neighbors of site 1."""
            assert n == 1
            return [{"site": structure[0]}, {"site": structure[2]}]

    class FakeOrderParams:
        """Order-parameter calculator returning fixed values."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Accept the same constructor shape as LocalStructOrderParams."""

        def get_order_parameters(
            self,
            structure: Structure,
            n: int,
            indices_neighs: list[int],
        ) -> np.ndarray:
            """Check the local structure is centered on site 1."""
            assert isinstance(structure, Structure)
            assert structure[0].species_string == "O"
            assert n == 0
            assert indices_neighs == [1, 2]
            return np.array(order_params)

    monkeypatch.setitem(local_env.CN_OPT_PARAMS, 4, {"Z": ("zero",), "T": ("tet",)})
    monkeypatch.setattr(local_env, "CrystalNN", FakeCrystalNN)
    monkeypatch.setattr(local_env, "LocalStructOrderParams", FakeOrderParams)

    assert chem_env.classify_local_env_with_order_params(structure, 1, 4) == expected


def test_collect_coord_envs_crystal_nn_normalizes_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalized CrystalNN counts weight sites equally; failed structures add none."""
    from pymatgen.analysis import local_env

    structure = Structure(Lattice.cubic(4), ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    # fails on its 2nd site, so its 1st site must not leak a partial row
    bad_structure = Structure(
        Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )

    class FakeCrystalNN:
        """Nearest-neighbor finder with one neighbor per site."""

        def get_nn_info(
            self, structure: Structure, site_idx: int
        ) -> list[dict[str, object]]:
            """Return one neighbor so the coordination number is one."""
            if structure is bad_structure and site_idx == 1:
                raise ValueError("No valid targets for site")
            return [{"site": structure[1 - site_idx]}]

    def fake_classify_local_env(
        _structure: Structure, site_idx: int, cn_val: int
    ) -> str:
        """Return a site-specific label to keep normalized rows distinct."""
        return f"site-{site_idx}:CN{cn_val}"

    monkeypatch.setattr(local_env, "CrystalNN", FakeCrystalNN)
    monkeypatch.setattr(
        chem_env,
        "classify_local_env_with_order_params",
        fake_classify_local_env,
    )

    with pytest.warns(
        UserWarning, match="CrystalNN analysis failed for structure 0, skipping it"
    ):
        result = chem_env.collect_coord_envs_crystal_nn(
            [bad_structure, structure], normalize=True
        )

    assert result == [
        {"coord_num": 1, "chem_env_symbol": "site-0:CN1", "count": 0.5},
        {"coord_num": 1, "chem_env_symbol": "site-1:CN1", "count": 0.5},
    ]
