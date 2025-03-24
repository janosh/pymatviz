"""Hierarchical multi-level pie charts (i.e. sunbursts).

E.g. for crystal symmetry distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import plotly.express as px

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.process_data import count_formulas


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import plotly.graph_objects as go
    from pymatgen.core import Composition, Structure

    from pymatviz.typing import FormulaGroupBy

ShowCounts = Literal["value", "percent", "value+percent", False]


def _limit_slices(
    df: pd.DataFrame,
    group_col: str,
    count_col: str,
    max_slices: int | None,
    max_slices_mode: Literal["other", "drop"],
    other_label: str | None = None,
) -> pd.DataFrame:
    """Limit the number of slices shown in each group of a sunburst plot.

    Args:
        df (pd.DataFrame): The data to limit.
        group_col (str): Column name to group by.
        count_col (str): Column name containing the count values.
        max_slices (int | None): Maximum number of slices to show per group. If None,
            all slices are shown. Defaults to None.
        max_slices_mode ("other" | "drop"): How to handle slices beyond max_slices:
            - "other": Combine remaining slices into an "Other" slice
            - "drop": Discard remaining slices entirely
        other_label (str | None): Label to use for the "Other" slice. If None, defaults
            to "Other (N more not shown)" where N is the number of omitted slices.

    Returns:
        DataFrame with limited slices per group
    """
    if max_slices is None or max_slices <= 0:
        return df

    # Validate max_slices_mode
    if max_slices_mode not in ("other", "drop"):
        raise ValueError(f"Invalid {max_slices_mode=}, must be 'other' or 'drop'")

    grouped = df.groupby(group_col)
    limited_dfs: list[pd.DataFrame] = []

    for group_name, group in grouped:
        # Sort by count in descending order
        sorted_group = group.sort_values(count_col, ascending=False)

        # If the group has more slices than the limit
        if len(sorted_group) > max_slices:
            # Take the top N slices
            top_slices = sorted_group.iloc[:max_slices]

            if max_slices_mode == "other":
                # Create an "Other" entry for the remaining slices
                other_slices = sorted_group.iloc[max_slices:]
                other_count = other_slices[count_col].sum()
                n_omitted = len(other_slices)

                # Create "Other" entry that combines the count of the omitted cells
                other_slice = pd.DataFrame(
                    {group_col: [group_name], count_col: [other_count]}
                )

                # Add any additional columns from the original DataFrame
                for col in df.columns:
                    if col not in (group_col, count_col):
                        if col in (Key.chem_sys, Key.formula):
                            other_slice[col] = [
                                other_label or f"Other ({n_omitted} more not shown)"
                            ]
                        elif col == Key.spg_num:
                            other_slice[col] = [f"Other ({n_omitted} more not shown)"]
                        else:
                            other_slice[col] = [""]  # Empty string for other columns

                # Combine top slices with the "Other" entry
                limited_group = pd.concat([top_slices, other_slice])
            else:  # max_slices_mode == "drop"
                limited_group = top_slices

            limited_dfs += [limited_group]
        else:
            # If the group has fewer slices than the limit, keep all of them
            limited_dfs += [sorted_group]

    # Combine all the limited groups
    return pd.concat(limited_dfs)


def spacegroup_sunburst(
    data: Sequence[int | str] | pd.Series,
    *,
    show_counts: ShowCounts = "value",
    max_slices: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    **kwargs: Any,
) -> go.Figure:
    """Generate a sunburst plot with crystal systems as the inner ring for a list of
    international space group numbers.

    Hint: To hide very small labels, set a uniformtext minsize and mode='hide'.
    fig.update_layout(uniformtext=dict(minsize=9, mode="hide"))

    Args:
        data (list[int] | pd.Series): A sequence (list, tuple, pd.Series) of
            space group strings or numbers (from 1 - 230) or pymatgen structures.
        show_counts ("value" | "percent" | "value+percent" | False): Whether to display
            values below each labels on the sunburst.
        max_slices (int | None): Maximum number of space groups to show
            for each crystal system. If None (default), all space groups are shown. If
            positive integer, only the top N space groups by count are shown.
        max_slices_mode ("other" | "drop"): How to handle spacegroups beyond max_slices:
            - "other": Combine remaining space groups into an "Other" slice (default)
            - "drop": Discard remaining space groups entirely
        color_discrete_sequence (list[str]): A list of 7 colors, one for each crystal
            system. Defaults to plotly.express.colors.qualitative.G10.
        **kwargs: Additional keyword arguments passed to plotly.express.sunburst.

    Returns:
        Figure: The Plotly figure.
    """
    if type(next(iter(data))).__qualname__ in ("Structure", "Atoms"):
        # if 1st sequence item is pymatgen structure or ASE Atoms, assume all are
        from moyopy import MoyoDataset
        from moyopy.interface import MoyoAdapter

        series = pd.Series(
            MoyoDataset(MoyoAdapter.from_py_obj(struct)).number for struct in data
        )
    else:
        series = pd.Series(data)

    df_spg_counts = pd.DataFrame(series.value_counts().reset_index())
    df_spg_counts.columns = [Key.spg_num, "count"]

    try:  # assume column contains integers as space group numbers
        df_spg_counts[Key.crystal_system] = df_spg_counts[Key.spg_num].map(
            pmv.utils.spg_to_crystal_sys
        )

    except (ValueError, TypeError):  # assume column is strings of space group symbols
        df_spg_counts[Key.crystal_system] = df_spg_counts[Key.spg_num].map(
            pmv.utils.spg_num_to_from_symbol
        )

    # Limit the number of space groups per crystal system if requested
    df_spg_counts = _limit_slices(
        df_spg_counts,
        group_col=Key.crystal_system,
        count_col="count",
        max_slices=max_slices,
        max_slices_mode=max_slices_mode,
        other_label=f"Other ({max_slices} more not shown)" if max_slices else None,
    )

    sunburst_defaults = dict(color_discrete_sequence=px.colors.qualitative.G10)

    fig = px.sunburst(
        df_spg_counts,
        path=[Key.crystal_system, Key.spg_num],
        values="count",
        **sunburst_defaults | kwargs,
    )

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

    fig.layout.margin = dict(l=10, r=10, b=10, pad=10)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig


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
    )

    sunburst_defaults = dict(
        color_discrete_sequence=px.colors.qualitative.Set3,
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
        fig.data[0].texttemplate = "%{label}<br>N=%{value}"
    elif show_counts == "value+percent":
        fig.data[0].textinfo = "label+value+percent entry"
        fig.data[0].texttemplate = "%{label}<br>N=%{value}<br>%{percentEntry}"
    elif show_counts is not False:
        raise ValueError(
            f"Invalid {show_counts=}, must be 'value', 'percent', 'value+percent', "
            "or False"
        )

    fig.layout.margin = dict(l=10, r=10, b=10, pad=10)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
