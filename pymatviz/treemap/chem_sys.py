"""Treemap of chemical systems grouped by arity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pymatviz._hierarchy import _chem_sys_hierarchy
from pymatviz.widgets.treemap import TreemapWidget


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Composition, Structure

    from pymatviz.typing import FormulaGroupBy


def chem_sys_treemap(
    data: Sequence[str | Composition | Structure],
    *,
    group_by: FormulaGroupBy = "chem_sys",
    max_systems: int | None = None,
    mode: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> TreemapWidget:
    """Show chemical systems within each arity as an interactive treemap.

    Input can contain formulas, chemical systems such as ``"Fe-O"``, compositions,
    and structures. Set ``group_by="formula"`` or ``"reduced_formula"`` to add formula
    leaves. A positive ``max_systems`` keeps the most frequent systems per arity
    (alphabetical ties) and all their formulas. ``mode="other"`` combines the rest
    in one leaf per arity; ``"drop"`` excludes their counts. Additional options,
    such as ``label_text`` and ``max_depth``, go to ``TreemapWidget``.
    """
    return TreemapWidget(
        _chem_sys_hierarchy(data, group_by, max_systems, mode), **kwargs
    )
