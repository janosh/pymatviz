"""Hierarchical multi-level pie charts (i.e. sunbursts).

E.g. for crystal symmetry distributions.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, get_args

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.process_data import count_formulas, normalize_structures
from pymatviz.typing import ShowCounts


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from pymatgen.core import Composition, Structure

    from pymatviz.typing import FormulaGroupBy


def _limit_slices(
    df_grouped: pd.DataFrame,
    group_col: str,
    count_col: str,
    max_slices: int | None,
    max_slices_mode: Literal["other", "drop"],
    *,
    other_label: str = "Other",
    child_col_for_other_label: str | None = None,
) -> pd.DataFrame:
    """Limit slices in sunburst plot with other/drop modes.

    Args:
        df_grouped (pd.DataFrame): DataFrame with grouped data
        group_col (str): Column name for grouping
        count_col (str): Column name for counts
        max_slices (int | None): Maximum number of slices to show
        max_slices_mode ("other" | "drop"): How to handle excess slices.
            - "other": Combine remaining slices into an "Other" slice (default)
            - "drop": Discard remaining slices entirely
        other_label (str): Label for grouped excess slices
        child_col_for_other_label (str | None): Column to use for other label

    Returns:
        pd.DataFrame: with limited slices
    """
    if max_slices_mode not in ("other", "drop"):
        raise ValueError(f"Invalid {max_slices_mode=}, must be 'other' or 'drop'")

    if not max_slices or max_slices <= 0:
        return df_grouped

    df_grouped = df_grouped.sort_values(count_col, ascending=False)

    if len(df_grouped) <= max_slices:
        return df_grouped

    if max_slices_mode == "drop":
        return df_grouped[:max_slices]

    # max_slices_mode == "other"
    top_slices = df_grouped[:max_slices]
    remaining_slices = df_grouped[max_slices:]

    other_row = {group_col: top_slices.iloc[0][group_col]}
    other_count = remaining_slices[count_col].sum()
    other_row[count_col] = other_count

    n_hidden = len(remaining_slices)
    other_text = f"{other_label} ({n_hidden} more not shown)"

    # Set the label for the other entry
    for col in df_grouped.columns:
        if col in (group_col, count_col):
            continue
        if child_col_for_other_label and col == child_col_for_other_label:
            other_row[col] = other_text
        elif child_col_for_other_label:
            # For child_col mode, set other columns to empty string
            other_row[col] = ""
        # Legacy mode: try formula first, then other columns
        elif col == Key.formula:
            other_row[col] = other_text
        else:
            other_row[col] = ""

    return pd.concat([top_slices, pd.DataFrame([other_row])], ignore_index=True)


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
        other_label="Other",
        child_col_for_other_label=Key.spg_num,
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
        fig.data[0].texttemplate = "%{label}<br>N=%{value}"
    elif show_counts == "value+percent":
        fig.data[0].textinfo = "label+value+percent entry"
        fig.data[0].texttemplate = "%{label}<br>N=%{value}<br>%{percentEntry}"
    elif show_counts is not False:
        raise ValueError(
            f"Invalid {show_counts=}, must be one of {get_args(ShowCounts)}"
        )

    fig.layout.margin = dict(l=10, r=10, b=10, pad=10)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig


def _get_cn_from_symbol(ce_symbol: str, symbol_cn_mapping: dict[str, int]) -> int:
    """Extract coordination number from ChemEnv symbol.

    Args:
        ce_symbol: ChemEnv symbol (e.g., 'T:4', 'O:6', 'M:8')
        symbol_cn_mapping: Mapping from symbols to coordination numbers

    Returns:
        Coordination number as integer
    """
    if ce_symbol in symbol_cn_mapping:
        return symbol_cn_mapping[ce_symbol]

    if ce_symbol == "S:1":
        return 1

    if ce_symbol.startswith("M:"):
        try:
            return int(ce_symbol.split(":")[1])
        except (ValueError, IndexError):
            return 0

    if ce_symbol in ("NULL", "UNKNOWN"):
        return 0

    return 0


def cn_ce_sunburst(
    structures: Structure | Sequence[Structure],
    *,
    chemenv_settings: dict[str, Any] | None = None,
    max_slices_cn: int | None = None,
    max_slices_ce: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    show_counts: ShowCounts = "value",
    normalize: bool = False,
) -> go.Figure:
    """Create sunburst plot of coordination numbers and environments.

    Args:
        structures (Structure | Sequence[Structure]): Structures to analyze.
        chemenv_settings (dict[str, Any] | None): Settings for ChemEnv analysis.
        max_slices_cn (int | None): Maximum CN slices to show. Defaults to None.
        max_slices_ce (int | None): Maximum CE slices per CN to show. Defaults to None.
        max_slices_mode ("other" | "drop"): How to handle excess slices. Defaults to
            "other".
        show_counts ("value" | "percent" | "value+percent" | False): How to display
            counts. Defaults to "value".
        normalize (bool): Whether to normalize counts per structure. Defaults to False.

    Returns:
        Plotly Figure with sunburst plot

    Raises:
        ValueError: For invalid inputs
    """
    import pymatgen.analysis.chemenv.coordination_environments as coord_envs
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometries as coord_geoms  # noqa: E501
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder as coord_finder  # noqa: E501
    import pymatgen.analysis.chemenv.coordination_environments.structure_environments as struct_envs  # noqa: E501
    from pymatgen.analysis.chemenv.coordination_environments import chemenv_strategies

    structs = normalize_structures(structures).values()

    if show_counts not in get_args(ShowCounts):
        raise ValueError(f"Invalid {show_counts=}")

    settings = chemenv_settings or {}
    cn_ce_data: list[dict[str, Any]] = []

    try:
        lgf = coord_finder.LocalGeometryFinder()
        lgf.setup_parameters(**settings)
        strategy = chemenv_strategies.SimplestChemenvStrategy()
        all_coord_geoms = coord_geoms.AllCoordinationGeometries()
        symbol_cn_mapping = all_coord_geoms.get_symbol_cn_mapping()

        for structure in structs:
            try:
                lgf.setup_structure(structure=structure)
                structure_environments = lgf.compute_structure_environments()
                lse = (
                    struct_envs.LightStructureEnvironments.from_structure_environments(
                        strategy, structure_environments
                    )
                )

                coord_envs_dict: dict[tuple[int, str], float] = {}

                for coord_envs in lse.coordination_environments or []:
                    for coord_env in coord_envs or []:
                        ce_symbol = coord_env["ce_symbol"]
                        cn_val = _get_cn_from_symbol(ce_symbol, symbol_cn_mapping)
                        key = (cn_val, ce_symbol)
                        coord_envs_dict[key] = coord_envs_dict.get(key, 0) + 1

                for (cn_val, ce_symbol), env_count in coord_envs_dict.items():
                    final_count = env_count
                    if normalize:
                        total = sum(coord_envs_dict.values())
                        if total > 0:
                            final_count = env_count / total

                    cn_ce_data.append(
                        {
                            "coord_num": cn_val,
                            "chem_env_symbol": ce_symbol,
                            "count": final_count,
                        }
                    )

            except (ImportError, RuntimeError) as exc:
                warnings.warn(
                    f"ChemEnv analysis failed for structure: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

    except (ImportError, RuntimeError) as exc:
        warnings.warn(f"ChemEnv setup failed: {exc}", UserWarning, stacklevel=2)

    if not cn_ce_data:
        fig = go.Figure()
        fig.layout.title = dict(
            text="No CN/CE data to display",
            x=0.5,
            font_size=16,
        )
        return fig

    df_cn_ce = pd.DataFrame(cn_ce_data)
    df_cn_ce = df_cn_ce.groupby(["coord_num", "chem_env_symbol"], as_index=False)[
        "count"
    ].sum()

    # Limit CN slices
    if max_slices_cn:
        df_cn_grouped = df_cn_ce.groupby("coord_num", as_index=False)["count"].sum()
        df_cn_grouped = _limit_slices(
            df_cn_grouped,
            group_col="coord_num",
            count_col="count",
            max_slices=max_slices_cn,
            max_slices_mode=max_slices_mode,
            other_label="Other CNs",
        )

        # Filter original data to keep only selected CNs
        if max_slices_mode == "drop":
            selected_cns = set(df_cn_grouped["coord_num"])
            df_cn_ce = df_cn_ce[df_cn_ce["coord_num"].isin(selected_cns)]
        else:
            # For "other" mode, combine excluded CNs
            selected_cns = {
                cn
                for cn in df_cn_grouped["coord_num"]
                if not str(cn).startswith("Other")
            }

            excluded_data = df_cn_ce[~df_cn_ce["coord_num"].isin(selected_cns)]
            if not excluded_data.empty:
                other_cn_count = excluded_data["count"].sum()
                other_entry = pd.DataFrame(
                    [
                        {
                            "coord_num": "Other CNs",
                            "chem_env_symbol": "Other CNs",
                            "count": other_cn_count,
                        }
                    ]
                )
                df_cn_ce = pd.concat(
                    [df_cn_ce[df_cn_ce["coord_num"].isin(selected_cns)], other_entry],
                    ignore_index=True,
                )

    # Limit CE slices within each CN
    if max_slices_ce:
        df_ce_limited = []
        for cn_val in df_cn_ce["coord_num"].unique():
            df_cn_subset = df_cn_ce[df_cn_ce["coord_num"] == cn_val]
            df_cn_subset = _limit_slices(
                df_cn_subset,
                group_col="coord_num",
                count_col="count",
                max_slices=max_slices_ce,
                max_slices_mode=max_slices_mode,
                other_label="Other CEs",
                child_col_for_other_label="chem_env_symbol",
            )
            df_ce_limited.append(df_cn_subset)
        df_cn_ce = pd.concat(df_ce_limited, ignore_index=True)

    fig = px.sunburst(df_cn_ce, path=["coord_num", "chem_env_symbol"], values="count")

    # Apply text formatting
    text_templates = {
        "value": "%{label}: %{value}",
        "percent": "%{label}: %{percentParent:.1%}",
        "value+percent": "%{label}: %{value} (%{percentParent:.1%})",
        False: "%{label}",
    }

    fig.data[0].update(
        texttemplate=text_templates[show_counts],
        textinfo="none",
        marker=dict(line=dict(color="white")),
    )

    return fig
