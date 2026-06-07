"""Chemical environment analysis utilities.

Functions for analyzing coordination numbers and chemical environments
using pymatgen's ChemEnv and local environment modules.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable

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


def collect_coord_envs_chemenv(
    structures: Iterable[Any],
    *,
    chem_env_settings: dict[str, Any] | None = None,
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """Collect coordination environment counts using pymatgen's ChemEnv module.

    Args:
        structures (Iterable[Structure]): Structures to analyze.
        chem_env_settings (dict | None): Passed to LocalGeometryFinder
            .setup_parameters. Defaults to None.
        normalize (bool): Normalize counts per structure. Defaults to False.

    Returns:
        list[dict]: One dict per (coord_num, chem_env_symbol) pair per structure
            with keys coord_num, chem_env_symbol, count.
    """
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometries as coord_geoms  # noqa: E501
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder as coord_finder  # noqa: E501
    import pymatgen.analysis.chemenv.coordination_environments.structure_environments as struct_envs  # noqa: E501
    from pymatgen.analysis.chemenv.coordination_environments import chemenv_strategies

    chem_env_data: list[dict[str, Any]] = []

    try:
        lgf = coord_finder.LocalGeometryFinder()
        lgf.setup_parameters(**(chem_env_settings or {}))
        strategy = chemenv_strategies.SimplestChemenvStrategy()
        all_coord_geoms = coord_geoms.AllCoordinationGeometries()
        symbol_cn_mapping = all_coord_geoms.get_symbol_cn_mapping()

        for structure in structures:
            try:
                lgf.setup_structure(structure=structure)
                structure_environments = lgf.compute_structure_environments()
                lse = (
                    struct_envs.LightStructureEnvironments.from_structure_environments(
                        strategy, structure_environments
                    )
                )

                coord_envs_dict: dict[tuple[int, str], float] = {}

                for env_list in lse.coordination_environments or []:
                    for coord_env in env_list or []:
                        ce_symbol = coord_env["ce_symbol"]
                        cn_val = get_cn_from_symbol(ce_symbol, symbol_cn_mapping)
                        key = (cn_val, ce_symbol)
                        coord_envs_dict[key] = coord_envs_dict.get(key, 0) + 1

                total = sum(coord_envs_dict.values())
                for (cn_val, ce_symbol), env_count in coord_envs_dict.items():
                    final_count = env_count
                    if normalize and total > 0:
                        final_count = env_count / total

                    chem_env_data += [
                        dict(
                            coord_num=cn_val,
                            chem_env_symbol=ce_symbol,
                            count=final_count,
                        )
                    ]

            except (ImportError, RuntimeError, KeyError) as exc:
                warnings.warn(
                    f"ChemEnv analysis failed for structure: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

    except (ImportError, RuntimeError) as exc:
        warnings.warn(f"ChemEnv setup failed: {exc}", UserWarning, stacklevel=2)

    return chem_env_data


def collect_coord_envs_crystal_nn(
    structures: Iterable[Any],
    *,
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """Collect coordination environment counts using CrystalNN + order params
    (faster than ChemEnv but less detailed geometric classification).

    Args:
        structures (Iterable[Structure]): Structures to analyze.
        normalize (bool): Normalize counts per structure. Defaults to False.

    Returns:
        list[dict]: One dict per site per structure with keys coord_num,
            chem_env_symbol, count.
    """
    from pymatgen.analysis.local_env import CrystalNN

    chem_env_data: list[dict[str, Any]] = []
    crystal_nn = CrystalNN()

    try:
        for structure in structures:
            try:
                # Get coordination info for each site
                for site_idx in range(len(structure)):
                    # Get coordination number
                    nn_info = crystal_nn.get_nn_info(structure, site_idx)
                    cn_val = len(nn_info)

                    # Get best matching coordination environment using order parameters
                    ce_symbol = classify_local_env_with_order_params(
                        structure, site_idx, cn_val
                    )

                    final_count = 1.0
                    if normalize:
                        final_count = 1.0 / len(structure)  # Normalize per structure

                    chem_env_data += [
                        dict(
                            coord_num=cn_val,
                            chem_env_symbol=ce_symbol,
                            count=final_count,
                        )
                    ]

            except (ImportError, RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"CrystalNN analysis failed for structure: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

    except (ImportError, RuntimeError) as exc:
        warnings.warn(f"CrystalNN setup failed: {exc}", UserWarning, stacklevel=2)

    return chem_env_data
