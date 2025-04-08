"""Calculate MLFF pair-repulsion curves for diatomic molecules.

Thanks to Tamas Stenczel who first did this type of PES smoothness and physicality
analysis in https://github.com/stenczelt/MACE-MP-work for the MACE-MP paper
https://arxiv.org/abs/2401.00096 (see fig. 56).
"""

# %%
from __future__ import annotations

import json
import lzma
import os
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from mace.calculators import mace_mp
from matbench_discovery import today
from tqdm import tqdm


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.calculators.calculator import Calculator

__date__ = "2024-03-31"
module_dir = os.path.dirname(__file__)
DiatomicsType = Literal["homo-nuclear", "hetero-nuclear"]
homo_nuc, hetero_nuc = get_args(DiatomicsType)
atom_num_symbol_map = dict(enumerate(chemical_symbols, start=0))


# %%
def generate_diatomics(
    elem1: str, elem2: str, distances: Sequence[float]
) -> list[Atoms]:
    """Build diatomic molecules in vacuum for given distances.

    Args:
        elem1 (str): Chemical symbol of the first element.
        elem2 (str): Chemical symbol of the second element.
        distances (Sequence[float]): Distances to sample at.

    Returns:
        list[Atoms]: Diatomic molecules with elements at given distances.
    """
    return [
        Atoms(f"{elem1}{elem2}", positions=[[0, 0, 0], [dist, 0, 0]], pbc=False)
        for dist in distances
    ]


def calc_diatomic_curve(
    pairs: list[tuple[str | int, str | int]],
    calculator: Calculator,
    model_name: str,
    distances: Sequence[float],
    results: dict[str, dict[str, list[float | list[list[float]]]]],
) -> dict[str, dict[str, list[float | list[list[float]]]]]:
    """Generate potential energy and forces data for diatomic molecules.

    Args:
        pairs (list[tuple[str | int, str | int]]): List of element pairs to calculate.
            Each pair can be specified as element symbols or atomic numbers.
        calculator (Calculator): ASE calculator instance.
        model_name (str): Name of the model for the output file.
        distances (list[float]): Distances to calculate potential energy at.
        results (dict[str, dict[str, list[float | list[list[float]]]]]): Results dict
            to collect energies and forces at given distances for all diatomic curves.
            Will be updated in-place.

    Returns:
        dict[str, dict[str, list[float | list[list[float]]]]]: Potential energy and
            forces data for diatomic molecules.
    """
    # saving results in dict: {"symbol-symbol": {"energies": [...], "forces": [...]}}
    for idx, (z1, z2) in (pbar := tqdm(enumerate(pairs, start=1))):
        # Convert atomic numbers to symbols if needed
        elem1 = atom_num_symbol_map.get(z1, z1)
        elem2 = atom_num_symbol_map.get(z2, z2)
        formula = f"{elem1}-{elem2}"
        prior_res = results.get(formula)

        if prior_res and len(prior_res.get("energies", [])) == len(distances):
            continue

        pbar.set_description(
            f"{idx}/{len(pairs)} {formula} diatomic curve with {model_name}"
        )

        results[formula] |= {"energies": [], "forces": []}
        for atoms in generate_diatomics(elem1, elem2, distances):
            results[formula]["energies"] += [calculator.get_potential_energy(atoms)]
            results[formula]["forces"] += [calculator.get_forces(atoms).tolist()]

    return results


if __name__ == "__main__":
    mace_chkpt_url = "https://github.com/ACEsuit/mace-foundations/releases/download"
    checkpoints = {
        # "mace-mpa-0-medium": f"{mace_chkpt_url}/mace_mpa_0/mace-mpa-0-medium.model",
        "mace-omat-0-medium": f"{mace_chkpt_url}/mace_omat_0/mace-omat-0-medium.model",
        # "GN-S-OMat-r6": "/lambdafs/assets/radsim_checkpoints/radsim-s-v4/"
        # "GN-S-OMat-cutoff6-grad-noquadcont-forces.pt",
    }
    distances = np.logspace(np.log10(0.1), np.log10(6.0), 119)

    for model_name, checkpoint in checkpoints.items():
        out_path = f"{module_dir}/{today}-{model_name}-diatomics.json.xz"
        results: dict[
            DiatomicsType, dict[str, dict[str, list[float | list[list[float]]]]]
        ] = {homo_nuc: {}, hetero_nuc: {}}
        if os.path.isfile(out_path):
            with lzma.open(out_path, mode="rt") as file:
                results |= json.load(file)

        if model_name.startswith("mace"):
            calculator = mace_mp(model=checkpoint, default_dtype="float64")
            atomic_numbers = calculator.z_table.zs

        elif model_name.startswith("GN-"):
            from fairchem.experimental.umlipv1.calculator import RadCalculator

            calculator = RadCalculator(
                checkpoint_path=checkpoint, cpu=True, dtype="float64", seed=0
            )
            atomic_numbers = [*range(1, 85)]
            # atomic_numbers = [*range(1, 85), *range(89, 95)]
        else:
            raise ValueError(f"Unknown {model_name=}")

        kwargs = dict(
            calculator=calculator,
            model_name=model_name,
            distances=distances,
        )
        # Generate homo-nuclear pairs (same element with itself)
        homo_pairs = [(z, z) for z in atomic_numbers]
        calc_diatomic_curve(pairs=homo_pairs, **kwargs, results=results[homo_nuc])

        # write results in case run is killed
        with lzma.open(out_path, mode="wt") as file:
            json.dump(results | {"distances": list(distances)}, file)
        print(f"Saved results to {out_path}")

        # Generate all hetero-nuclear pairs (different elements)
        for z1 in calculator.z_table.zs:
            hetero_pairs = [(z1, z2) for z2 in atomic_numbers if z2 != z1]
            calc_diatomic_curve(
                pairs=hetero_pairs, **kwargs, results=results[hetero_nuc]
            )

            # write results in case run is killed
            with lzma.open(out_path, mode="wt") as file:
                json.dump(results | {"distances": list(distances)}, file)
            print(f"Saved results to {out_path}")
