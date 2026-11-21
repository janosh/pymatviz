"""Hierarchy preparation for sunburst and treemap plots."""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal

from pymatviz.composition import count_formulas
from pymatviz.enums import Key


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Composition, Structure

    from pymatviz.typing import FormulaGroupBy


def _validate_overflow(overflow: Literal["other", "drop"]) -> None:
    """Reject invalid aggregation modes before preparing plot data."""
    if overflow not in ("other", "drop"):
        raise ValueError(f"Invalid {overflow=}, must be 'other' or 'drop'")


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
    ranked = sorted(nodes, key=itemgetter("value"), reverse=True)
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


def _chem_sys_hierarchy(
    data: Sequence[str | Composition | Structure],
    group_by: FormulaGroupBy,
    max_systems: int | None,
    overflow: Literal["other", "drop"],
) -> list[dict[str, Any]]:
    """Group systems and formulas by arity with stable path IDs and counts."""
    _validate_overflow(overflow)
    counts = count_formulas(data, group_by=group_by)
    roots = []
    for arity, group in counts.groupby("arity_name", sort=False):
        # count_formulas orders systems alphabetically, providing a stable tiebreaker.
        totals = group.groupby(Key.chem_sys, sort=False)[Key.count].sum()
        systems = [
            {"id": f"{arity}/{system}", "label": system, "value": count}
            for system, count in totals.items()
        ]
        children = _limit_nodes(systems, max_systems, overflow, parent_id=str(arity))
        if group_by != "chem_sys":
            by_system = {node["label"]: node for node in children}
            for system, formula, count in zip(
                group[Key.chem_sys], group[Key.formula], group[Key.count], strict=True
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
    return roots
