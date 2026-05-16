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
    result = chem_env.get_cn_from_symbol(symbol, mapping)
    assert result == expected_cn
    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.parametrize(
    ("site_idx", "cn_val", "exact"),
    [
        *[(0, cn, None) for cn in range(1, 25)],
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

    assert isinstance(result, str)
    assert len(result) > 0
    if exact:
        assert result == exact
    else:
        assert ":" in result or result.startswith("CN")
    if ":" in result:
        parts = result.split(":")
        assert len(parts) == 2
        try:
            int(parts[1])
        except ValueError:
            pytest.fail(f"Second part of '{result}' is not numeric")
        if not result.startswith("CN:"):
            assert parts[1] == str(cn_val)


@pytest.mark.parametrize("struct_idx", [0, 1])
@pytest.mark.parametrize("cn_val", [0, 1, 2, 4, 6, 8, 12, 13, 20, 50, 100])
def test_classify_local_env_fixture_structures(
    structures: tuple[Structure, Structure], struct_idx: int, cn_val: int
) -> None:
    result = chem_env.classify_local_env_with_order_params(
        structures[struct_idx], 0, cn_val
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert ":" in result or result.startswith("CN")


def test_classify_local_env_uses_requested_site_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for using site 0/order-param neighbors for every site."""
    from pymatgen.analysis import local_env

    structure = Structure(
        Lattice.cubic(4),
        ["Li", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]],
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


@pytest.mark.parametrize(
    ("lattice_param", "species", "coords"),
    [
        (2.0, ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        (6.0, ["K", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        (4.0, ["Fe"], [[0, 0, 0]]),
        (3.0, ["Al"], [[0, 0, 0]]),
        (
            4.0,
            ["Ca", "Ti", "O", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        ),
    ],
)
def test_classify_local_env_various_structures(
    lattice_param: float, species: list[str], coords: list[list[float]]
) -> None:
    structure = Structure(Lattice.cubic(lattice_param), species, coords)
    for cn_val in [0, 1, 2, 4, 6, 8, 12, 13, 20, 50, 100]:
        for site_idx in range(len(species)):
            result = chem_env.classify_local_env_with_order_params(
                structure, site_idx, cn_val
            )
            assert isinstance(result, str)
            assert len(result) > 0
            assert ":" in result or result.startswith("CN")


def test_chem_env_functions_integration(
    structures: tuple[Structure, Structure],
) -> None:
    cn = chem_env.get_cn_from_symbol("T:4", {"T:4": 4, "O:6": 6, "C:8": 8})
    assert cn == 4
    result = chem_env.classify_local_env_with_order_params(structures[0], 0, cn)
    assert isinstance(result, str)
    assert len(result) > 0
