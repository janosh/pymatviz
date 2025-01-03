"""Predict pair-repulsion curves for diatomic molecules with MACE-MP.

Thanks to Tamas Stenczel who first did this type of PES smoothness and physicality
analysis in https://github.com/stenczelt/MACE-MP-work for the MACE-MP paper
https://arxiv.org/abs/2401.00096 (see fig. 56).
"""

# %%
from __future__ import annotations

import json
import lzma
import os
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
    elem1: str, elem2: str, distances: list[float] | np.ndarray
) -> list[Atoms]:
    """Build diatomic molecules in vacuum for given distances.

    Args:
        elem1: Chemical symbol of the first element.
        elem2: Chemical symbol of the second element.
        distances: Distances to sample at.

    Returns:
        list[Atoms]: List of diatomic molecules.
    """
    return [
        Atoms(f"{elem1}{elem2}", positions=[[0, 0, 0], [dist, 0, 0]], pbc=False)
        for dist in distances
    ]


def calc_one_pair(
    elem1: str, elem2: str, calc: MACECalculator, distances: list[float] | np.ndarray
) -> list[float]:
    """Calculate potential energy for a pair of elements at given distances.

    Args:
        elem1: Chemical symbol of the first element.
        elem2: Chemical symbol of the second element.
        calc: MACECalculator instance.
        distances: Distances to calculate potential energy at.

    Returns:
        list[float]: Potential energies at each distance.
    """
    return [
        calc.get_potential_energy(at)
        for at in generate_diatomics(elem1, elem2, distances)
    ]


def generate_homo_nuclear(calculator: MACECalculator, label: str) -> None:
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
        elem1, elem2 = chemical_symbols[z0], chemical_symbols[z0]
        formula = f"{elem1}-{elem2}"
        with timer(formula):
            results[formula] = calc_one_pair(elem1, elem2, calculator, distances)
    with lzma.open(f"homo-nuclear-{label}.json.xz", mode="wt") as file:
        json.dump(results, file)


def generate_hetero_nuclear(z0: int, calculator: MACECalculator, label: str) -> None:
    """Generate potential energy data for hetero-nuclear diatomic molecules with a
    fixed first element.

    Args:
        z0: Atomic number of the fixed first element.
        calculator: MACECalculator instance.
        label: Label for the output file.
    """
    out_path = f"hetero-nuclear-diatomics-{z0}-{label}.json.xz"
    if os.path.isfile(out_path):
        print(f"Skipping {z0} because {out_path} already exists")
        return
    print(f"Starting {z0}")
    distances = np.linspace(0.1, 6.0, 119)
    allowed_atomic_numbers = calculator.z_table.zs
    # saving the results in a dict: "z0-z1" -> [energy] & saved the distances
    results = {"distances": list(distances)}
    # hetero-nuclear diatomics
    for z1 in allowed_atomic_numbers:
        elem1, elem2 = chemical_symbols[z0], chemical_symbols[z1]
        formula = f"{elem1}-{elem2}"
        with timer(formula):
            results[formula] = calc_one_pair(elem1, elem2, calculator, distances)
    with lzma.open(out_path, mode="wt") as file:
        json.dump(results, file)


if __name__ == "__main__":
    # first homo-nuclear diatomics
    # for label in ("small", "medium", "large"):
    #     calculator = mace_mp(model=label)
    #     generate_homo_nuclear(calculator, f"mace-{label}")

    # then all hetero-nuclear diatomics
    for label in ("small", "medium", "large"):
        calculator = mace_mp(model=label)
        for z0 in calculator.z_table.zs:
            generate_hetero_nuclear(z0, calculator, f"mace-{label}")
