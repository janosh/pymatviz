"""Chemical system, chemical environment and space group hierarchies as sunburst or
treemap widgets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import fsum
from typing import TYPE_CHECKING, Any, Literal

import kemio
import pandas as pd

from pymatviz.compositions import count_formulas
from pymatviz.structures import normalize_periodic_structures, normalize_spacegroups
from pymatviz.widgets.sunburst import SunburstWidget
from pymatviz.widgets.treemap import TreemapWidget


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatviz.typing import (
        FormulaGroupBy,
        FormulaLike,
        PeriodicStructure,
        SpacegroupLike,
    )


_WIDGETS = {"sunburst": SunburstWidget, "treemap": TreemapWidget}


def _check_options(overflow: str, kind: str = "sunburst") -> None:
    """Reject invalid overflow modes and widget kinds before preparing plot data."""
    if overflow not in ("other", "drop"):
        raise ValueError(f"Invalid {overflow=}, must be 'other' or 'drop'")
    if kind not in _WIDGETS:
        raise ValueError(f"Invalid {kind=}, must be 'sunburst' or 'treemap'")


def _limit_nodes(
    nodes: list[dict[str, Any]],
    limit: int | None,
    overflow: Literal["other", "drop"],
    *,
    parent_id: str = "",
) -> list[dict[str, Any]]:
    """Keep the largest siblings and optionally sum omitted values into Other.

    Raises:
        ValueError: If limit is below 1 (None means no limit).
    """
    if limit is not None and limit < 1:
        raise ValueError(f"Slice limits must be at least 1 or None, got {limit=}")
    if limit is None or len(nodes) <= limit:
        return nodes
    ranked = sorted(nodes, key=lambda node: node["value"], reverse=True)
    kept = ranked[:limit]
    if overflow == "other":
        omitted = ranked[limit:]
        kept.append(
            {
                # Native max_children and min_fraction reserve the sibling ID "Other".
                "id": f"{parent_id}/__omitted__" if parent_id else "__omitted__",
                "label": f"Other ({len(omitted)} more not shown)",
                "value": sum(node["value"] for node in omitted),
            }
        )
    return kept


def chem_sys_hierarchy(
    data: Sequence[FormulaLike],
    *,
    kind: Literal["sunburst", "treemap"] = "sunburst",
    group_by: FormulaGroupBy = "chem_sys",
    max_systems: int | None = None,
    overflow: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> SunburstWidget | TreemapWidget:
    """Show chemical systems within each arity as an interactive sunburst or treemap.

    Input can contain formulas, chemical systems such as ``"Fe-O"``, compositions,
    and structures. Set ``group_by="formula"`` or ``"reduced_formula"`` to add formula
    leaves. A positive ``max_systems`` keeps the most frequent systems per arity
    (alphabetical ties) and all their formulas. ``overflow="other"`` combines the rest
    in one leaf per arity; ``"drop"`` excludes their counts. Additional options,
    such as ``label_text`` and ``max_depth``, go to the widget.
    """
    _check_options(overflow, kind)
    counts = count_formulas(data, group_by=group_by)
    roots = []
    for arity, group in counts.groupby("arity_name", sort=False):
        # count_formulas orders systems alphabetically, providing a stable tiebreaker.
        totals = group.groupby("chem_sys", sort=False)["count"].sum()
        systems = [
            {"id": f"{arity}/{system}", "label": system, "value": count}
            for system, count in totals.items()
        ]
        children = _limit_nodes(systems, max_systems, overflow, parent_id=str(arity))
        if group_by != "chem_sys":
            by_system = {node["label"]: node for node in children}
            for system, formula, count in zip(
                group["chem_sys"], group["formula"], group["count"], strict=True
            ):
                if (node := by_system.get(system)) is not None:
                    node.setdefault("children", []).append(
                        {
                            "id": f"{node['id']}/{formula}",
                            "label": formula,
                            "value": count,
                        }
                    )
            # All native value modes sum leaves. An explicit branch value would
            # double count its formulas when the widget uses remainder mode.
            for node in children:
                if "children" in node:
                    del node["value"]
        roots.append({"id": arity, "label": arity, "children": children})
    return _WIDGETS[kind](roots, **kwargs)


def site_environments(
    structures: Sequence[kemio.Structure],
) -> list[list[tuple[int, str]]]:
    """(coordination number, ChemEnv symbol such as "O:6") of each site of each
    structure: pymatgen's LocalGeometryFinder with SimplestChemenvStrategy on the input
    cell. Sites whose coordination number has no reference geometry are "CN:<n>".

    Raises:
        ValueError: For structures kemio can't classify (e.g. coincident sites).
    """
    return [
        [
            (cn, symbol or f"CN:{cn}")
            for cn, symbol in zip(
                result.coordination_numbers.tolist(), result.symbols, strict=True
            )
        ]
        for result in kemio.chemenv(structures)
    ]


def chem_env_hierarchy(
    structures: PeriodicStructure | Sequence[PeriodicStructure],
    *,
    kind: Literal["sunburst", "treemap"] = "sunburst",
    normalize: bool = False,
    max_coordination_numbers: int | None = None,
    max_environments: int | None = None,
    overflow: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> SunburstWidget | TreemapWidget:
    """Show coordination numbers and their chemical environments as a sunburst or
    treemap.

    Sites are labeled by ChemEnv symbol (see site_environments). With
    normalize=True, each non-empty structure contributes one total count before
    pruning. Limit the most frequent coordination numbers and
    environments per coordination number with max_coordination_numbers and
    max_environments (coordination numbers ranked by their full counts, ties by
    coordination number and environment symbol). Mode "other" combines omitted counts
    into one leaf per group; "drop" omits them. Additional options go to the widget.
    """
    _check_options(overflow, kind)
    structs = [
        struct
        for struct in normalize_periodic_structures(structures).values()
        if len(struct)
    ]
    counts: defaultdict[tuple[int, str], list[float]] = defaultdict(list)
    for envs in site_environments(structs):
        for env in envs:
            counts[env].append(1 / len(envs) if normalize else 1)

    by_coord_num: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for (coord_num, symbol), weights in sorted(counts.items()):
        by_coord_num[coord_num].append(
            dict(id=f"CN {coord_num}/{symbol}", label=symbol, value=fsum(weights))
        )
    nodes = [
        dict(
            id=f"CN {coord_num}",
            label=f"CN {coord_num}",
            children=children,
            value=fsum(child["value"] for child in children),
        )
        for coord_num, children in by_coord_num.items()
    ]
    roots = _limit_nodes(nodes, max_coordination_numbers, overflow)
    for root in roots:
        if children := root.get("children"):
            root["children"] = _limit_nodes(
                children, max_environments, overflow, parent_id=root["id"]
            )
            del root["value"]  # The renderer derives branch totals from visible leaves.
    return _WIDGETS[kind](roots, **kwargs)


def spacegroup_sunburst(
    data: Sequence[SpacegroupLike] | pd.Series,
    *,
    max_spacegroups: int | None = None,
    overflow: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> SunburstWidget:
    """Show space group counts with crystal systems as the inner ring.

    Args:
        data (Sequence[SpacegroupLike] | pd.Series): Space group numbers (1 -
            230), international symbols or structures (kemio, pymatgen, ASE,
            phonopy).
        max_spacegroups (int | None): Maximum number of space groups to show for each
            crystal system. If None (default), all space groups are shown. If a
            positive integer, only the top N space groups by count are shown (ties
            keep the first seen).
        overflow ("other" | "drop"): How to handle space groups beyond
            max_spacegroups:
            - "other": Combine remaining space groups into an "Other" slice (default)
            - "drop": Discard remaining space groups entirely
        **kwargs: Passed to SunburstWidget, e.g. ``label_text`` (default
            ``"label+value"``), ``max_depth`` or ``style``.

    Returns:
        SunburstWidget: The MatterViz sunburst.

    Raises:
        ValueError: If data is empty, has missing values or invalid space groups, or
            overflow is invalid.
        TypeError: If the space group of a structure can't be determined.
    """
    _check_options(overflow)
    numbers = normalize_spacegroups(data)  # merges spellings, settings, int/float
    # symbol inputs are labeled with their canonical short symbol, others by number
    as_symbols = isinstance(pd.Series(data).iloc[0], str)

    by_system: dict[str, list[dict[str, Any]]] = {}
    for number, count in Counter(numbers.tolist()).most_common():
        crystal_sys = kemio.crystal_system(number)
        label = kemio.space_group_symbol(number) if as_symbols else str(number)
        by_system.setdefault(crystal_sys, []).append(
            {"id": f"{crystal_sys}/{label}", "label": label, "value": count}
        )
    nodes = [
        {
            "id": crystal_sys,
            "label": crystal_sys,
            "children": _limit_nodes(
                children, max_spacegroups, overflow, parent_id=crystal_sys
            ),
        }
        for crystal_sys, children in by_system.items()
    ]
    return SunburstWidget(nodes, **{"label_text": "label+value"} | kwargs)
