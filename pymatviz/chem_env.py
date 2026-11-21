"""Chemical environment analysis utilities.

Functions for analyzing coordination numbers and chemical environments
using pymatgen's ChemEnv and local environment modules (optional pymatgen
dependency: pip install 'pymatviz[pymatgen]').
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from math import fsum
from typing import TYPE_CHECKING, Any, Literal

from pymatviz.hierarchy import _limit_nodes, _validate_overflow, hierarchy_widget
from pymatviz.structures import normalize_periodic_structures, to_pymatgen


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pymatgen.core import PeriodicSite, Structure

    from pymatviz.typing import PeriodicStructure
    from pymatviz.widgets.sunburst import SunburstWidget
    from pymatviz.widgets.treemap import TreemapWidget

ChemEnvSettings = dict[str, Any] | Literal["chemenv", "crystal_nn"]


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
        except ValueError:
            return 0
        return max(0, cn_val)

    return 0


def classify_local_env_with_order_params(
    center: PeriodicSite, neighbors: Sequence[PeriodicSite]
) -> str:
    """Classify a site from its image-resolved CrystalNN neighbors."""
    from pymatgen.analysis import local_env
    from pymatgen.core import Structure

    cn_val = len(neighbors)
    if (cn_params := local_env.CN_OPT_PARAMS.get(cn_val)) is None:
        return f"CN:{cn_val}"

    try:
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

        # Preserve each periodic image's coordinates, including repeated site indices.
        local_structure = Structure.from_sites([center, *neighbors])

        # Calculate order parameters
        local_order_params = local_ops.get_order_parameters(
            structure=local_structure,
            n=0,
            indices_neighs=list(range(1, cn_val + 1)),
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

    except (RuntimeError, ValueError, IndexError):
        pass  # order params unavailable for this site, use generic CN label
    return f"CN:{cn_val}"  # no confident geometry match, use generic CN label


def collect_coord_envs_chemenv(
    structures: Iterable[Structure],
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

    # setup failures (e.g. invalid chem_env_settings) propagate to the caller
    lgf = coord_finder.LocalGeometryFinder()
    lgf.setup_parameters(**(chem_env_settings or {}))
    strategy = chemenv_strategies.SimplestChemenvStrategy()
    symbol_cn_mapping = coord_geoms.AllCoordinationGeometries().get_symbol_cn_mapping()

    chem_env_data: list[dict[str, Any]] = []
    for struct_idx, structure in enumerate(structures):
        try:
            lgf.setup_structure(structure=structure)
            structure_environments = lgf.compute_structure_environments()
            lse = struct_envs.LightStructureEnvironments.from_structure_environments(
                strategy, structure_environments
            )
        except (RuntimeError, KeyError) as exc:
            warnings.warn(
                f"ChemEnv analysis failed for structure {struct_idx}, skipping it: "
                f"{exc!r}",
                UserWarning,
                stacklevel=2,
            )
            continue

        coord_envs_dict: dict[tuple[int, str], float] = {}
        for env_list in lse.coordination_environments or []:
            for coord_env in env_list or []:
                ce_symbol = coord_env["ce_symbol"]
                cn_val = get_cn_from_symbol(ce_symbol, symbol_cn_mapping)
                key = (cn_val, ce_symbol)
                coord_envs_dict[key] = coord_envs_dict.get(key, 0) + 1

        total = sum(coord_envs_dict.values())
        chem_env_data += [
            dict(
                coord_num=cn_val,
                chem_env_symbol=ce_symbol,
                count=env_count / total if normalize else env_count,
            )
            for (cn_val, ce_symbol), env_count in coord_envs_dict.items()
        ]

    return chem_env_data


def collect_coord_envs_crystal_nn(
    structures: Iterable[Structure],
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

    for struct_idx, structure in enumerate(structures):
        if not structure:
            continue
        site_count = 1.0 / len(structure) if normalize else 1.0
        # collect per structure so a failure midway doesn't leave partial rows
        struct_rows: list[dict[str, Any]] = []
        try:
            for site_idx, site in enumerate(structure):
                neighbors = [
                    info["site"] for info in crystal_nn.get_nn_info(structure, site_idx)
                ]
                struct_rows.append(
                    dict(
                        coord_num=len(neighbors),
                        chem_env_symbol=classify_local_env_with_order_params(
                            site, neighbors
                        ),
                        count=site_count,
                    )
                )
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"CrystalNN analysis failed for structure {struct_idx}, skipping it: "
                f"{exc!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        chem_env_data += struct_rows

    return chem_env_data


def chem_env_nodes(
    structures: PeriodicStructure | Sequence[PeriodicStructure],
    *,
    chem_env_settings: ChemEnvSettings,
    normalize: bool,
    max_coordination_numbers: int | None,
    max_environments: int | None,
    overflow: Literal["other", "drop"],
) -> list[dict[str, Any]]:
    """Build coordination-number branches with environment counts as leaves.

    Rank coordination numbers by their full counts before limiting environments
    per branch. Ties use coordination number and environment symbol respectively.
    """
    _validate_overflow(overflow)

    if not isinstance(chem_env_settings, dict) and chem_env_settings not in (
        "chemenv",
        "crystal_nn",
    ):
        raise ValueError(
            f"Invalid {chem_env_settings=}, expected 'chemenv', 'crystal_nn' or a dict"
        )
    structs = (
        to_pymatgen(struct)
        for struct in normalize_periodic_structures(structures).values()
        if len(struct)
    )
    if chem_env_settings == "crystal_nn":
        rows = collect_coord_envs_crystal_nn(structs, normalize=normalize)
    else:
        settings = {} if chem_env_settings == "chemenv" else chem_env_settings
        rows = collect_coord_envs_chemenv(
            structs, chem_env_settings=settings, normalize=normalize
        )

    counts: defaultdict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        counts[row["coord_num"], row["chem_env_symbol"]].append(row["count"])

    by_cn: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for (coord_num, symbol), weights in sorted(counts.items()):
        by_cn[coord_num].append(
            dict(id=f"CN {coord_num}/{symbol}", label=symbol, value=fsum(weights))
        )
    nodes = [
        dict(
            id=f"CN {coord_num}",
            label=f"CN {coord_num}",
            children=children,
            value=fsum(child["value"] for child in children),
        )
        for coord_num, children in by_cn.items()
    ]
    roots = _limit_nodes(nodes, max_coordination_numbers, overflow)
    for root in roots:
        if children := root.get("children"):
            root["children"] = _limit_nodes(
                children, max_environments, overflow, parent_id=root["id"]
            )
            del root["value"]  # The renderer derives branch totals from visible leaves.
    return roots


def chem_env_hierarchy(
    structures: PeriodicStructure | Sequence[PeriodicStructure],
    *,
    kind: Literal["sunburst", "treemap"] = "sunburst",
    chem_env_settings: ChemEnvSettings = "crystal_nn",
    normalize: bool = False,
    max_coordination_numbers: int | None = None,
    max_environments: int | None = None,
    overflow: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> SunburstWidget | TreemapWidget:
    """Show coordination numbers and their chemical environments as a sunburst or
    treemap.

    Use CrystalNN (default), "chemenv", or a dict of ChemEnv settings.
    With normalize=True, each structure with at least one classified environment
    contributes one total count before pruning (structures without environments
    contribute zero). Limit the most frequent coordination numbers and
    environments per coordination number with max_coordination_numbers and
    max_environments. Mode "other" combines omitted counts into one leaf per
    group; "drop" omits them. Additional options go to the widget.
    """
    nodes = chem_env_nodes(
        structures,
        chem_env_settings=chem_env_settings,
        normalize=normalize,
        max_coordination_numbers=max_coordination_numbers,
        max_environments=max_environments,
        overflow=overflow,
    )
    return hierarchy_widget(kind, nodes, **kwargs)
