"""Chemical environment analysis utilities.

Functions for analyzing coordination numbers and chemical environments
using pymatgen's ChemEnv and local environment modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pymatgen.core import IStructure, Structure


def get_cn_from_symbol(ce_symbol: str, symbol_cn_mapping: dict[str, int]) -> int:
    """Extract coordination number from ChemEnv symbol.

    Args:
        ce_symbol (str): ChemEnv symbol (e.g., 'T:4', 'O:6', 'M:8')
        symbol_cn_mapping (dict[str, int]): Mapping from symbols to coordination numbers

    Returns:
        int: Coordination number
    """
    if ce_symbol in symbol_cn_mapping:
        return symbol_cn_mapping[ce_symbol]

    if ce_symbol == "S:1":
        return 1

    if ce_symbol.startswith("M:"):
        try:
            cn_val = int(ce_symbol.split(":")[1])
        except (ValueError, IndexError):
            return 0
        return max(0, cn_val)

    if ce_symbol in ("NULL", "UNKNOWN"):
        return 0

    return 0


def classify_local_env_with_order_params(
    structure: Structure | IStructure, site_idx: int, cn_val: int
) -> str:
    """Classify local coordination environment using LocalStructOrderParams.

    Based on Crystal Toolkit's approach using pymatgen's LocalStructOrderParams
    to calculate order parameters for different coordination geometries.

    Args:
        structure (Structure | IStructure): The crystal structure
        site_idx (int): Index of the site to analyze
        cn_val (int): Coordination number of the site

    Returns:
        str: String describing the coordination environment (e.g. "T:4", "O:6", "CN:8")
    """
    from pymatgen.analysis import local_env
    from pymatgen.core import Structure

    try:
        analysis_structure = Structure.from_sites(structure)
        if not 0 <= site_idx < len(analysis_structure):
            return f"CN:{cn_val}"

        # Check if we have order parameters for this coordination number
        cn_params = local_env.CN_OPT_PARAMS.get(cn_val)
        if cn_params is None:
            return f"CN:{cn_val}"

        # Get the parameter names and settings for this CN
        names = list(cn_params)
        order_params = list(cn_params.values())
        types = [str(order_param[0]) for order_param in order_params]
        params = [
            (
                order_param[1]
                if len(order_param) > 1 and isinstance(order_param[1], dict)
                else None
            )
            for order_param in order_params
        ]

        # Create LocalStructOrderParams instance
        local_ops = local_env.LocalStructOrderParams(types, parameters=params)

        # Get neighboring sites using CrystalNN
        nn_info = local_env.CrystalNN().get_nn_info(
            structure=analysis_structure, n=site_idx
        )

        # Create local structure: central site + neighbors
        sites = [analysis_structure[site_idx]] + [info["site"] for info in nn_info]
        local_structure = Structure.from_sites(sites)

        # Calculate order parameters
        local_order_params = local_ops.get_order_parameters(
            structure=local_structure,
            n=0,
            indices_neighs=list(range(1, len(sites))),
        )

        # Find the geometry with the highest order parameter
        best_match_idx, best_value = max(
            (
                (idx, val)
                for idx, val in enumerate(local_order_params)
                if val is not None
            ),
            key=lambda item: item[1],
            default=(0, 0),
        )
        if best_value > 0.5:  # Only use if reasonably good match
            return f"{names[best_match_idx]}:{cn_val}"

    except (ImportError, RuntimeError, ValueError, IndexError):
        # Fallback to generic CN-based label
        pass
    return f"CN:{cn_val}"  # Fallback to generic CN label
