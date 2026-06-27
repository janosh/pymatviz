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
        ("O:6", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 6),
        ("C:8", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 8),
        ("PP:5", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 5),
        ("CUSTOM:3", {"CUSTOM:3": 3, "X:Y:7": 7}, 3),
        ("X:Y:7", {"CUSTOM:3": 3, "X:Y:7": 7}, 7),
        ("S:1", {}, 1),
        ("S:1", {"other": 10}, 1),
        ("M:1", {}, 1),
        ("M:6", {}, 6),
        ("M:7", {}, 7),
        ("M:8", {"T:4": 4}, 8),
        ("M:10", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 10),
        ("M:12", {}, 12),
        ("M:15", {}, 15),
        ("M:24", {}, 24),
        ("M:100", {}, 100),
        ("NULL", {}, 0),
        ("NULL", {"T:4": 4}, 0),
        ("UNKNOWN", {}, 0),
        ("UNKNOWN", {"T:4": 4}, 0),
        ("O:6", {"T:4": 4}, 0),
        ("M:", {}, 0),
        ("M:abc", {}, 0),
        ("M:3.5", {}, 0),
        ("M:-5", {}, 0),
        ("INVALID_SYMBOL", {}, 0),
        ("UNKNOWN_SYMBOL", {}, 0),
        ("", {}, 0),
        (":", {}, 0),
        ("T:", {}, 0),
        ("invalid", {}, 0),
        ("random_text", {}, 0),
        ("123", {}, 0),
        ("T:4:extra", {"T:4": 4}, 0),
    ],
)
def test_get_cn_from_symbol(
    symbol: str, mapping: dict[str, int], expected_cn: int
) -> None:
    assert chem_env.get_cn_from_symbol(symbol, mapping) == expected_cn


@pytest.mark.parametrize(
    ("site_idx", "cn_val", "exact"),
    [
        (0, 0, "CN:0"),
        (0, -1, "CN:-1"),
        (0, 99, "CN:99"),
        (0, 1000, "CN:1000"),
        (999, 6, "CN:6"),
        (-1, 4, "CN:4"),
        (100, 8, "CN:8"),
    ],
)
def test_classify_local_env_with_order_params_cubic(
    structures: tuple[Structure, Structure],
    site_idx: int,
    cn_val: int,
    exact: str | None,
) -> None:
    result = chem_env.classify_local_env_with_order_params(
        structures[0], site_idx, cn_val
    )

    assert result == exact


def test_classify_local_env_uses_requested_site_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for using site 0/order-param neighbors for every site."""
    from pymatgen.analysis import local_env

    structure = Structure(
        Lattice.cubic(4), ["Li", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]]
    )

    class FakeCrystalNN:
        def get_nn_info(self, structure: Structure, n: int) -> list[dict[str, object]]:
            assert n == 1
            return [{"site": structure[0]}, {"site": structure[2]}]

    class FakeOrderParams:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_order_parameters(
            self,
            structure: Structure,
            n: int,
            indices_neighs: list[int],
        ) -> np.ndarray:
            assert isinstance(structure, Structure)
            assert structure[0].species_string == "O"
            assert n == 0
            assert indices_neighs == [1, 2]
            return np.array([0.0, 1.0])

    monkeypatch.setitem(local_env.CN_OPT_PARAMS, 4, {"Z": ("zero",), "T": ("tet",)})
    monkeypatch.setattr(local_env, "CrystalNN", FakeCrystalNN)
    monkeypatch.setattr(local_env, "LocalStructOrderParams", FakeOrderParams)

    assert chem_env.classify_local_env_with_order_params(structure, 1, 4) == "T:4"


def test_classify_local_env_falls_back_for_weak_order_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weak local order matches fall back to the generic CN label."""
    from pymatgen.analysis import local_env

    structure = Structure(Lattice.cubic(4), ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    class FakeCrystalNN:
        """Nearest-neighbor finder returning a single neighbor."""

        def get_nn_info(self, structure: Structure, n: int) -> list[dict[str, object]]:
            """Return one neighbor so order parameters can be evaluated."""
            return [{"site": structure[1 - n]}]

    class FakeOrderParams:
        """Order-parameter calculator returning a low-confidence match."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Accept the same constructor shape as LocalStructOrderParams."""

        def get_order_parameters(
            self,
            structure: Structure,
            n: int,
            indices_neighs: list[int],
        ) -> np.ndarray:
            """Return a value below the production confidence threshold."""
            assert isinstance(structure, Structure)
            assert n == 0
            assert indices_neighs == [1]
            return np.array([0.5])

    monkeypatch.setitem(local_env.CN_OPT_PARAMS, 1, {"L": ("linear",)})
    monkeypatch.setattr(local_env, "CrystalNN", FakeCrystalNN)
    monkeypatch.setattr(local_env, "LocalStructOrderParams", FakeOrderParams)

    assert chem_env.classify_local_env_with_order_params(structure, 0, 1) == "CN:1"


def test_collect_coord_envs_crystal_nn_normalizes_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalized CrystalNN counts assign each site equal structure weight."""
    from pymatgen.analysis import local_env

    structure = Structure(Lattice.cubic(4), ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    class FakeCrystalNN:
        """Nearest-neighbor finder with one neighbor per site."""

        def get_nn_info(
            self, structure: Structure, site_idx: int
        ) -> list[dict[str, object]]:
            """Return one neighbor so the coordination number is one."""
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

    result = chem_env.collect_coord_envs_crystal_nn([structure], normalize=True)

    assert result == [
        {"coord_num": 1, "chem_env_symbol": "site-0:CN1", "count": 0.5},
        {"coord_num": 1, "chem_env_symbol": "site-1:CN1", "count": 0.5},
    ]
