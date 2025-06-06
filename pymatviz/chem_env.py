"""Chemical environment analysis utilities.

Functions for analyzing coordination numbers and chemical environments
using pymatgen's ChemEnv and local environment modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pymatgen.core import Structure


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
            return int(ce_symbol.split(":")[1])
        except (ValueError, IndexError):
            return 0

    if ce_symbol in ("NULL", "UNKNOWN"):
        return 0

    return 0


def classify_local_env_with_order_params(
    structure: Structure, site_idx: int, cn_val: int
) -> str:
    """Classify local coordination environment using LocalStructOrderParams.

    Based on Crystal Toolkit's approach using pymatgen's LocalStructOrderParams
    to calculate order parameters for different coordination geometries.

    Args:
        structure (Structure): The crystal structure
        site_idx (int): Index of the site to analyze
        cn_val (int): Coordination number of the site

    Returns:
        str: String describing the coordination environment (e.g. "T:4", "O:6", "CN:8")
    """
    from pymatgen.analysis import local_env

    try:
        # Check if we have order parameters for this coordination number
        if cn_val not in [int(k_cn) for k_cn in local_env.CN_OPT_PARAMS]:
            return f"CN:{cn_val}"

        # Get the parameter names and settings for this CN
        names = list(local_env.CN_OPT_PARAMS[cn_val])
        types = []
        params = []
        for name in names:
            types.append(local_env.CN_OPT_PARAMS[cn_val][name][0])
            tmp = (
                local_env.CN_OPT_PARAMS[cn_val][name][1]
                if len(local_env.CN_OPT_PARAMS[cn_val][name]) > 1
                else None
            )
            params.append(tmp)

        # Create LocalStructOrderParams instance
        lost_ops = local_env.LocalStructOrderParams(types, parameters=params)

        # Get neighboring sites using CrystalNN
        crystal_nn = local_env.CrystalNN()
        nn_info = crystal_nn.get_nn_info(structure, site_idx)

        # Create sites list: central site + neighbors
        sites = [structure[site_idx]] + [info["site"] for info in nn_info]

        # Calculate order parameters
        neighbor_indices = list(range(1, len(sites)))
        lost_op_vals = lost_ops.get_order_parameters(
            sites, 0, indices_neighs=neighbor_indices
        )

        # Find the geometry with the highest order parameter
        best_match_idx = 0
        best_value = 0.0

        for idx, op_val in enumerate(lost_op_vals):
            if op_val is not None and op_val > best_value:
                best_value = op_val
                best_match_idx = idx

        # Return the best matching geometry name with CN
        if best_value > 0.5:  # Only use if reasonably good match
            return f"{names[best_match_idx]}:{cn_val}"

    except (ImportError, RuntimeError, ValueError, IndexError):
        # Fallback to generic CN-based label
        return f"CN:{cn_val}"
    else:
        return f"CN:{cn_val}"  # Fallback to generic CN label
