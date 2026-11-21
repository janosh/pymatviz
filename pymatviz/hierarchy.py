"""Chemical systems as sunburst or treemap hierarchies, and the node pruning they share
with chemical environment hierarchies.
"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal

from pymatviz.composition import count_formulas
from pymatviz.widgets.sunburst import SunburstWidget
from pymatviz.widgets.treemap import TreemapWidget


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


def _chem_sys_nodes(
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
    return roots


def hierarchy_widget(
    kind: Literal["sunburst", "treemap"], nodes: list[dict[str, Any]], **kwargs: Any
) -> SunburstWidget | TreemapWidget:
    """Sunburst or treemap of hierarchy nodes.

    Raises:
        ValueError: For any other kind.
    """
    if kind not in ("sunburst", "treemap"):
        raise ValueError(f"Invalid {kind=}, must be 'sunburst' or 'treemap'")
    widget_cls = SunburstWidget if kind == "sunburst" else TreemapWidget
    return widget_cls(nodes, **kwargs)


def chem_sys_hierarchy(
    data: Sequence[str | Composition | Structure],
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
    nodes = _chem_sys_nodes(data, group_by, max_systems, overflow)
    return hierarchy_widget(kind, nodes, **kwargs)
