"""Sunburst plot of chemical systems."""

from collections.abc import Sequence
from typing import Any, Literal, get_args

import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core import Composition, Structure

from pymatviz.enums import Key
from pymatviz.process_data import count_formulas
from pymatviz.sunburst.helpers import _limit_slices
from pymatviz.typing import FormulaGroupBy, ShowCounts


def chem_sys_sunburst(
    data: Sequence[str | Composition | Structure],
    *,
    show_counts: ShowCounts = "value",
    group_by: FormulaGroupBy = "chem_sys",
    max_slices: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> go.Figure:
    """Generate a sunburst plot showing the distribution of chemical systems by arity.

    The innermost ring shows the number of samples with each arity (unary, binary,
    etc.), and the outer ring shows the counts for each unique chemical system within
    that arity.

    Args:
        data (Sequence[str | Composition | Structure]): Chemical systems. Can be:
            - Chemical system strings like ["Fe-O", "Li-P-O"]
            - Formula strings like ["Fe2O3", "LiPO4"]
            - Pymatgen Composition objects
            - Pymatgen Structure objects
        show_counts ("value" | "percent" | "value+percent" | False): Whether to display
            values below each labels on the sunburst.
        group_by ("formula" | "reduced_formula" | "chem_sys"): How to group formulas:
            - "formula": Each unique formula is counted separately.
            - "reduced_formula": Formulas are reduced to simplest ratios (e.g. Fe2O3
                and Fe4O6 count as same).
            - "chem_sys": All formulas with same elements are grouped (e.g. FeO and
                Fe2O3 count as Fe-O). Defaults to "chem_sys".
        max_slices (int | None): Maximum number of chemical systems to show
            for each arity level. If None (default), all systems are shown. If positive
            integer, only the top N systems by count are shown.
        max_slices_mode ("other" | "drop"): How to handle systems beyond max_slices:
            - "other": Combine remaining systems into an "Other" slice (default)
            - "drop": Discard remaining systems entirely
        **kwargs: Additional keyword arguments passed to plotly.express.sunburst.

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
        >>> # Show top 5 systems per arity, combine rest into "Other"
        >>> fig1 = pmv.chem_sys_sunburst(formulas, max_slices=5)
        >>> # Show only top 5 systems per arity, drop the rest
        >>> fig2 = pmv.chem_sys_sunburst(formulas, max_slices=5, max_slices_mode="drop")
    """
    df_counts = count_formulas(data, group_by=group_by)

    # Limit the number of systems per arity if requested
    df_counts = _limit_slices(
        df_counts,
        group_col="arity_name",
        count_col=Key.count,
        max_slices=max_slices,
        max_slices_mode=max_slices_mode,
        child_col_for_other_label=Key.chem_sys,
    )

    sunburst_defaults = dict(
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    path = ["arity_name", Key.chem_sys]
    if group_by != Key.chem_sys:
        path += [Key.formula]

    fig = px.sunburst(
        df_counts, path=path, values=Key.count, **sunburst_defaults | kwargs
    )

    if show_counts == "percent":
        fig.data[0].textinfo = "label+percent entry"
    elif show_counts == "value":
        fig.data[0].textinfo = "label+value"
        fig.data[0].texttemplate = "%{label}<br>N=%{value:.2f}"
    elif show_counts == "value+percent":
        fig.data[0].textinfo = "label+value+percent entry"
        fig.data[0].texttemplate = "%{label}<br>N=%{value:.2f}<br>%{percentEntry}"
    elif show_counts is not False:
        raise ValueError(
            f"Invalid {show_counts=}, must be one of {get_args(ShowCounts)}"
        )

    fig.layout.margin = dict(l=10, r=10, b=10, pad=10)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
