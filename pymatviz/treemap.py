"""Hierarchical treemap visualizations.

E.g. for chemical system distributions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import plotly.express as px

from pymatviz.enums import Key
from pymatviz.process_data import count_formulas


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import plotly.graph_objects as go
    from pymatgen.core import Composition, Structure

    from pymatviz.typing import FormulaGroupBy

ShowCounts = Literal["value", "percent", "value+percent", False]
ArityFormatter = Callable[[str, int, int], str]


def default_arity_formatter(arity: str, count: int, total: int) -> str:
    """Default formatter for arity with counts and percentages of total dataset."""
    return f"{arity} (N={count:,}, {count / total:.1%})"


def chem_sys_treemap(
    data: Sequence[str | Composition | Structure],
    *,
    show_counts: ShowCounts = "value",
    show_arity_counts: ArityFormatter | bool = default_arity_formatter,
    group_by: FormulaGroupBy = "chem_sys",
    **kwargs: Any,
) -> go.Figure:
    """Generate a treemap plot showing the distribution of chemical systems by arity.

    The first level shows the number of samples with each arity (unary, binary,
    etc.), and the second level shows the counts for each unique chemical system within
    that arity.

    Args:
        data (Sequence[str | Composition | Structure]): Chemical systems. Can be:
            - Chemical system strings like ["Fe-O", "Li-P-O"]
            - Formula strings like ["Fe2O3", "LiPO4"]
            - Pymatgen Composition objects
            - Pymatgen Structure objects
        show_counts ("value" | "percent" | "value+percent" | False): Whether to display
            values below each labels on the treemap.
        show_arity_counts (Callable[[str, int, int], str]): How to display arity names
            and their counts. A function that takes arity name, count, and total count
            and returns a string to show atop each top-level treemap node.
            Default: lambda arity, cnt, total: f"{arity} (N={cnt:,}, {cnt/total:.1%})"
        group_by ("formula" | "reduced_formula" | "chem_sys"): How to group formulas:
            - "formula": Each unique formula is counted separately.
            - "reduced_formula": Formulas are reduced to simplest ratios (e.g. Fe2O3
                and Fe4O6 count as same).
            - "chem_sys": All formulas with same elements are grouped (e.g. FeO and
                Fe2O3 count as Fe-O). Defaults to "chem_sys".
        **kwargs: Additional keyword arguments passed to plotly.express.treemap

    Returns:
        Figure: The Plotly figure.

    Example:
        >>> import pymatviz as pmv
        >>> formulas = [
        ...     "Fe2O3",  # binary
        ...     "Fe4O6",  # same as Fe2O3 when group_by="reduced_formula"
        ...     "FeO",  # different formula but same system when group_by="chem_sys"
        ...     "Li2O",  # binary
        ...     "LiFeO2",  # ternary
        ... ]
        >>> # Count each formula separately
        >>> fig1 = pmv.chem_sys_treemap(formulas, group_by="formula")
        >>> # Group by reduced formulas
        >>> fig2 = pmv.chem_sys_treemap(formulas, group_by="reduced_formula")
        >>> # Group by chemical systems (default)
        >>> fig3 = pmv.chem_sys_treemap(formulas)  # group_by="chem_sys"
    """
    df_counts = count_formulas(data, group_by=group_by)

    # Add counts and percentages to arity labels if requested
    if show_arity_counts is True:
        show_arity_counts = default_arity_formatter
    if show_arity_counts is not False:
        arity_totals = df_counts.groupby("arity_name")[Key.count].sum()
        total_count = arity_totals.sum()
        df_counts["arity_name"] = df_counts["arity_name"].map(
            lambda x: show_arity_counts(x, arity_totals[x], total_count)
        )

    treemap_defaults = dict(
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    path = ["arity_name", Key.chem_sys]
    if group_by != Key.chem_sys:
        path += [Key.formula]

    fig = px.treemap(
        df_counts, path=path, values=Key.count, **treemap_defaults | kwargs
    )

    # Remove Parent and ID from hover tooltips
    hovertemplate = (
        "%{label}<br>Count: %{value}<br>%{percentEntry:,.1%}<br>of this "
        "arity<extra></extra>"
    )
    fig.data[0].hovertemplate = hovertemplate

    if show_counts == "percent":
        fig.data[0].textinfo = "label+percent entry"
    elif show_counts == "value":
        fig.data[0].textinfo = "label+value"
        fig.data[0].texttemplate = "%{label}<br>N=%{value}"
    elif show_counts == "value+percent":
        fig.data[0].textinfo = "label+value+percent entry"
        fig.data[0].texttemplate = "%{label}<br>N=%{value}<br>%{percentEntry}"
    elif show_counts is not False:
        raise ValueError(
            f"Invalid {show_counts=}, must be 'value', 'percent', 'value+percent', "
            "or False"
        )

    fig.layout.margin = dict(l=0, r=0, b=0, pad=0)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
