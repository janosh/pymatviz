"""Hierarchical multi-level pie charts (i.e. sunbursts).

E.g. for crystal symmetry distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import pandas as pd
import plotly.express as px
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.groups import SpaceGroup

import pymatviz as pmv
from pymatviz.enums import Key


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import plotly.graph_objects as go

ShowCounts = Literal["value", "percent", "value+percent", False]
GroupBy = Literal["formula", "reduced_formula", "chem_sys"]


def spacegroup_sunburst(
    data: Sequence[int | str] | pd.Series,
    *,
    show_counts: ShowCounts = "value",
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
        color_discrete_sequence (list[str]): A list of 7 colors, one for each crystal
            system. Defaults to plotly.express.colors.qualitative.G10.
        **kwargs: Additional keyword arguments passed to plotly.express.sunburst.

    Returns:
        Figure: The Plotly figure.
    """
    if isinstance(next(iter(data)), Structure):
        # if 1st sequence item is structure, assume all are
        series = pd.Series(
            struct.get_space_group_info()[1]  # type: ignore[union-attr]
            for struct in data
        )
    else:
        series = pd.Series(data)

    df_spg_counts = pd.DataFrame(series.value_counts().reset_index())
    df_spg_counts.columns = [Key.spg_num, "count"]

    try:  # assume column contains integers as space group numbers
        df_spg_counts[Key.crystal_system] = df_spg_counts[Key.spg_num].map(
            pmv.utils.crystal_sys_from_spg_num
        )

    except (ValueError, TypeError):  # column must be strings of space group symbols
        df_spg_counts[Key.crystal_system] = [
            SpaceGroup(x).crystal_system for x in df_spg_counts[Key.spg_num]
        ]

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
    group_by: GroupBy = "chem_sys",
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
        >>> # Count each formula separately
        >>> fig1 = pmv.arity_sunburst(formulas, group_by="formula")
        >>> # Group by reduced formulas
        >>> fig2 = pmv.arity_sunburst(formulas, group_by="reduced_formula")
        >>> # Group by chemical systems (default)
        >>> fig3 = pmv.arity_sunburst(formulas)  # group_by="chem_sys"
    """
    if len(data) == 0:
        raise ValueError("Empty input: data sequence is empty")

    # Map from number of elements to arity name
    arity_names: Final[dict[int, str]] = {
        1: "unary",
        2: "binary",
        3: "ternary",
        4: "quaternary",
        5: "quinary",
        6: "senary",
        7: "septenary",
        8: "octonary",
        9: "nonary",
        10: "denary",
    }

    # Convert all inputs to chemical systems (sorted tuples of element strings)
    systems: list[tuple[str, ...]] = []
    formulas: list[str | None] = []  # store formulas if not grouping by chem_sys
    for item in data:
        if isinstance(item, Structure):
            elems = sorted(item.composition.chemical_system.split("-"))
            if group_by == "formula":
                formula = str(item.composition)
            elif group_by == "reduced_formula":
                formula = item.composition.reduced_formula
            else:  # chem_sys
                formula = None
        elif isinstance(item, Composition):
            elems = sorted(item.chemical_system.split("-"))
            if group_by == "formula":
                formula = str(item)
            elif group_by == "reduced_formula":
                formula = item.reduced_formula
            else:  # chem_sys
                formula = None
        elif isinstance(item, str):
            if "-" in item:  # already a chemical system string
                elems = sorted(item.split("-"))
                formula = item if group_by != Key.chem_sys else None
            else:  # assume it's a formula string
                try:
                    comp = Composition(item)
                    elems = sorted(comp.chemical_system.split("-"))
                    if group_by == "formula":
                        formula = item  # preserve original formula string
                    elif group_by == "reduced_formula":
                        formula = comp.reduced_formula
                    else:  # chem_sys
                        formula = None
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid formula: {item}") from e
        else:
            raise TypeError(
                f"Expected str, Composition or Structure, got {type(item)} instead"
            )

        # Remove duplicates and sort elements
        elems = sorted(set(elems))
        if not all(elem.isalpha() for elem in elems):
            raise ValueError(f"Invalid elements in system: {item}")
        systems += [tuple(elems)]
        if group_by != Key.chem_sys:
            formulas += [formula]

    # Create a DataFrame with arity and chemical system columns
    df_systems = pd.DataFrame({"system": systems})
    if group_by != Key.chem_sys:
        df_systems[Key.formula] = formulas

    df_systems[Key.arity] = df_systems["system"].map(len)
    df_systems["arity_name"] = df_systems[Key.arity].map(
        lambda n_elems: arity_names.get(n_elems, f"{n_elems}-component")
    )
    df_systems[Key.chem_sys] = df_systems["system"].str.join("-")

    # Count occurrences of each system
    group_cols = ["arity_name", Key.chem_sys]
    if group_by != Key.chem_sys:
        group_cols += [Key.formula]

    df_counts = df_systems.value_counts(group_cols).reset_index()
    df_counts.columns = [*group_cols, Key.count]  # preserve original column names

    # Sort by arity and chemical system for consistent ordering
    df_counts = df_counts.sort_values(["arity_name", Key.chem_sys])

    sunburst_defaults = dict(
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    path = ["arity_name", Key.chem_sys]
    if group_by != Key.chem_sys:
        path += [Key.formula]

    fig = px.sunburst(
        df_counts,
        path=path,
        values=Key.count,
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
