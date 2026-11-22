"""Coordination numbers and chemical environments of structures as hierarchies,
classified with matscikit's ChemEnv or CrystalNN order parameters.
"""

from __future__ import annotations

from collections import defaultdict
from math import fsum
from typing import TYPE_CHECKING, Any, Literal

import matscikit as mk

from pymatviz.hierarchy import _limit_nodes, _validate_overflow, hierarchy_widget
from pymatviz.structures import normalize_periodic_structures


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatviz.typing import PeriodicStructure
    from pymatviz.widgets.sunburst import SunburstWidget
    from pymatviz.widgets.treemap import TreemapWidget

type ChemEnvMethod = Literal["chemenv", "crystal_nn"]

# Minimum order parameter for a CrystalNN geometry label, else the site is "CN:<n>"
MIN_ORDER_PARAM = 0.5


def site_environments(
    structures: Sequence[mk.Structure], method: ChemEnvMethod
) -> list[list[tuple[int, str]]]:
    """(coordination number, environment symbol) of each site of each structure.

    "chemenv" gives ChemEnv symbols such as "O:6" (pymatgen's LocalGeometryFinder
    with SimplestChemenvStrategy, on the input cell). "crystal_nn" gives the
    best-matching CrystalNN order-parameter geometry such as "octahedral:6" when it
    scores above MIN_ORDER_PARAM. Sites without a match are "CN:<n>".

    Raises:
        ValueError: For an invalid method, or structures matscikit can't classify
            (e.g. disordered ones with "crystal_nn").
    """
    if method == "chemenv":
        return [
            [
                (cn, symbol or f"CN:{cn}")
                for cn, symbol in zip(
                    result.coordination_numbers.tolist(), result.symbols, strict=True
                )
            ]
            for result in mk.chemenv(structures)
        ]
    if method == "crystal_nn":
        return [
            [
                (cn, f"{geometry}:{cn}")
                if geometry is not None and score > MIN_ORDER_PARAM
                else (cn, f"CN:{cn}")
                for cn, geometry, score in zip(
                    result.coordination_numbers.tolist(),
                    result.geometries,
                    result.scores.tolist(),
                    strict=True,
                )
            ]
            for result in mk.local_environments(structures)
        ]
    raise ValueError(f"Invalid {method=}, expected 'chemenv' or 'crystal_nn'")


def chem_env_nodes(
    structures: PeriodicStructure | Sequence[PeriodicStructure],
    *,
    method: ChemEnvMethod,
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
    if method not in ("chemenv", "crystal_nn"):
        raise ValueError(f"Invalid {method=}, expected 'chemenv' or 'crystal_nn'")
    structs = [
        struct
        for struct in normalize_periodic_structures(structures).values()
        if len(struct)
    ]
    counts: defaultdict[tuple[int, str], list[float]] = defaultdict(list)
    for envs in site_environments(structs, method):
        for env in envs:
            counts[env].append(1 / len(envs) if normalize else 1)

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
    method: ChemEnvMethod = "crystal_nn",
    normalize: bool = False,
    max_coordination_numbers: int | None = None,
    max_environments: int | None = None,
    overflow: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> SunburstWidget | TreemapWidget:
    """Show coordination numbers and their chemical environments as a sunburst or
    treemap.

    method "crystal_nn" (default) labels sites by CrystalNN order-parameter
    geometry, "chemenv" by ChemEnv symbol (see site_environments). With
    normalize=True, each non-empty structure contributes one total count before
    pruning. Limit the most frequent coordination numbers and
    environments per coordination number with max_coordination_numbers and
    max_environments. Mode "other" combines omitted counts into one leaf per
    group; "drop" omits them. Additional options go to the widget.
    """
    nodes = chem_env_nodes(
        structures,
        method=method,
        normalize=normalize,
        max_coordination_numbers=max_coordination_numbers,
        max_environments=max_environments,
        overflow=overflow,
    )
    return hierarchy_widget(kind, nodes, **kwargs)
