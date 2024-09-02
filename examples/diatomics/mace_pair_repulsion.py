"""Predict pair-repulsion curves for diatomic molecules with MACE-MP.

All credit for this code to Tamas Stenczel. Authored in https://github.com/stenczelt/MACE-MP-work
for MACE-MP paper https://arxiv.org/abs/2401.00096
"""

# %%
from __future__ import annotations

import json
import lzma
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from mace.calculators import MACECalculator, mace_mp


if TYPE_CHECKING:
    from collections.abc import Generator

__date__ = "2024-03-31"


# %%
@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager for timing code execution."""
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label} took {time.perf_counter() - start:.3}s.")


def generate_diatomics(
    symbol0: str,
    symbol1: str,
    distances: list[float] | np.ndarray,
) -> list[Atoms]:
    """Build diatomic molecules in vacuum for given distances.

    Args:
        symbol0: Chemical symbol of the first element.
        symbol1: Chemical symbol of the second element.
        distances: Distances to sample at.

    Returns:
        list[Atoms]: List of diatomic molecules.
    """
    return [
        Atoms(f"{symbol0}{symbol1}", positions=[[0, 0, 0], [dist, 0, 0]])
        for dist in distances
    ]


def calc_one_pair(
    z0: int,
    z1: int,
    calc: MACECalculator,
    distances: list[float] | np.ndarray,
) -> list[float]:
    """Calculate potential energy for a pair of elements at given distances.

    Args:
        z0: Atomic number of the first element.
        z1: Atomic number of the second element.
        calc: MACECalculator instance.
        distances: Distances to calculate potential energy at.

    Returns:
        list[float]: Potential energies at each distance.
    """
    return [
        calc.get_potential_energy(at)
        for at in generate_diatomics(
            chemical_symbols[z0],
            chemical_symbols[z1],
            distances,
        )
    ]


def generate_homonuclear(calculator: MACECalculator, label: str) -> None:
    """Generate potential energy data for homonuclear diatomic molecules.

    Args:
        calculator: MACECalculator instance.
        label: Label for the output file.
    """
    distances = np.linspace(0.1, 6.0, 119)
    allowed_atomic_numbers = calculator.z_table.zs
    # saving the results in a dict: "z0-z1" -> [energy] & saved the distances
    results = {"distances": list(distances)}
    # homo-nuclear diatomics
    for z0 in allowed_atomic_numbers:
        z1 = z0
        formula = f"{chemical_symbols[z0]}{chemical_symbols[z1]}"
        with timer(formula):
            results[f"{z0}-{z1}"] = calc_one_pair(z0, z1, calculator, distances)
    with lzma.open(f"homo-nuclear-{label}.json.lzma", "wt") as file:
        json.dump(results, file)


def generate_fixed_any(z0: int, calculator: MACECalculator, label: str) -> None:
    """Generate potential energy data for hetero-nuclear diatomic molecules with a
    fixed first element.

    Args:
        z0: Atomic number of the fixed first element.
        calculator: MACECalculator instance.
        label: Label for the output file.
    """
    distances = np.linspace(0.1, 6.0, 119)
    allowed_atomic_numbers = calculator.z_table.zs
    # saving the results in a dict: "z0-z1" -> [energy] & saved the distances
    results = {"distances": list(distances)}
    # hetero-nuclear diatomics
    for z1 in allowed_atomic_numbers:
        formula = f"{chemical_symbols[z0]}{chemical_symbols[z1]}"
        with timer(formula):
            results[f"{z0}-{z1}"] = calc_one_pair(z0, z1, calculator, distances)
    with lzma.open(f"{label}-{z0}-X.json.lzma", "wt") as file:
        json.dump(results, file)


if __name__ == "__main__":
    # first homo-nuclear diatomics
    for label in ("small", "medium", "large"):
        calculator = mace_mp(model=label)
        generate_homonuclear(calculator, f"mace-{label}")

    # then all hetero-nuclear diatomics
    for label in ("small", "medium", "large"):
        calculator = mace_mp(model=label)
        for z0 in calculator.z_table.zs:
            generate_fixed_any(z0, calculator, f"mace-{label}")
