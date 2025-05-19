"""Hierarchical multi-level pie charts (i.e. sunbursts).

E.g. for crystal symmetry distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import pandas as pd
import plotly.express as px
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
    AllCoordinationGeometries,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
    StructureEnvironments,
)

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.process_data import count_formulas, normalize_structures


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
    child_col_for_other_label: str | None = None,
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
        child_col_for_other_label (str | None): Column name where the 'Other' label
            should be placed. If None, uses legacy behavior for specific known keys.
            Defaults to None.

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
                label_text = other_label or f"Other ({n_omitted} more not shown)"

                # Create "Other" entry that combines the count of the omitted cells
                # Initialize with group_col and count_col
                other_slice_data = {group_col: [group_name], count_col: [other_count]}

                # Populate other columns
                for col in df.columns:
                    if col == group_col or col == count_col:
                        continue  # Already handled

                    if child_col_for_other_label == col or (
                        child_col_for_other_label is None
                        and col
                        in (
                            Key.chem_sys,
                            Key.formula,
                            Key.spg_num,
                        )
                    ):
                        other_slice_data[col] = [label_text]
                    # If child_col_for_other_label is set but doesn't match current col,
                    # and this col is one of the legacy keys, it should NOT get the label.
                    # It should get "" unless it was explicitly set.
                    elif child_col_for_other_label is not None and col in (
                        Key.chem_sys,
                        Key.formula,
                        Key.spg_num,
                    ):
                        if (
                            col not in other_slice_data
                        ):  # ensure it wasn't set by child_col logic
                            other_slice_data[col] = [""]
                    elif col not in other_slice_data:  # Default for any other columns
                        other_slice_data[col] = [""]

                other_slice = pd.DataFrame(other_slice_data)
                # Ensure all original columns are present in other_slice
                other_slice = other_slice.reindex(columns=df.columns, fill_value="")
                # Re-assign the known values after reindexing to be safe
                other_slice[group_col] = group_name
                other_slice[count_col] = other_count
                if (
                    child_col_for_other_label
                    and child_col_for_other_label in other_slice.columns
                ):
                    other_slice[child_col_for_other_label] = label_text
                elif (
                    child_col_for_other_label is None
                ):  # re-apply legacy if child_col not used
                    if Key.chem_sys in other_slice.columns and Key.chem_sys not in (
                        group_col,
                        count_col,
                    ):
                        other_slice[Key.chem_sys] = label_text
                    elif Key.formula in other_slice.columns and Key.formula not in (
                        group_col,
                        count_col,
                    ):
                        other_slice[Key.formula] = (
                            label_text  # this could overwrite chem_sys if both present
                        )
                    if Key.spg_num in other_slice.columns and Key.spg_num not in (
                        group_col,
                        count_col,
                    ):
                        other_slice[Key.spg_num] = label_text

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
        other_label=None,
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


def cn_ce_sunburst(
    structures: Sequence[Structure],
    *,
    show_counts: ShowCounts = "value",
    normalize: bool = False,
    max_slices_cn: int | None = None,
    max_slices_ce: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    chemenv_settings: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Generate a sunburst plot of coordination number (CN) and coordination
    environment (CE) distributions.

    The inner ring represents CNs, and the outer ring represents CEs for each CN.
    For a large dataset, you'd likely want to use max_slices options and possibly
    normalize=True to make the plot interpretable.

    Args:
        structures (Sequence[Structure]): A list or sequence of pymatgen Structure
            objects.
        show_counts ("value" | "percent" | "value+percent" | False): Whether to display
            values on the sunburst slices. Defaults to "value".
        normalize (bool): Whether to normalize CE counts within each structure to sum
            to 1. If False (default), absolute counts of CEs are used. This helps
            avoid overrepresentation of larger structures.
        max_slices_cn (int | None): Maximum number of CN slices to show. If None, all
            CNs are shown. Defaults to None.
        max_slices_ce (int | None): Maximum number of CE slices to show for each CN.
            If None, all CEs are shown. Defaults to None.
        max_slices_mode ("other" | "drop"): How to handle slices beyond max_slices_cn
            or max_slices_ce:
            - "other": Combine remaining slices into an "Other" slice (default).
            - "drop": Discard remaining slices entirely.
        chemenv_settings (dict[str, Any] | None): Settings for ChemEnv analysis.
            Keys like 'maximum_distance_factor', 'minimum_angle_factor',
            'voronoi_normalized_distance_tolerance', 'voronoi_normalized_angle_tolerance'
            can be passed to `LocalGeometryFinder.compute_structure_environments`.
            The example script for this function used `distance_cutoff` (maps to
            `maximum_distance_factor`) and `angle_cutoff` (maps to
            `minimum_angle_factor`). Defaults to None (using ChemEnv defaults).
        **kwargs: Additional keyword arguments passed to plotly.express.sunburst.

    Returns:
        go.Figure: The Plotly sunburst figure.

    Example:
        >>> from pymatgen.core import Structure
        >>> from pymatgen.ext.matproj import MPRester  # doctest: +SKIP
        >>> # Fetch a few structures from Materials Project for demonstration
        >>> with MPRester() as mpr:  # doctest: +SKIP
        ...     structures = mpr.get_structures("mp-149")  # Silicon # doctest: +SKIP
        ...     structures += mpr.get_structures("mp-22862")  # Diamond # doctest: +SKIP
        ...     structures += mpr.get_structures(
        ...         "mp-568347"
        ...     )  # Lonsdaleite # doctest: +SKIP
        >>> if not structures:  # doctest: +SKIP
        ...     raise SystemExit(
        ...         "MPRester query returned no structures."
        ...     )  # doctest: +SKIP
        >>> fig = pmv.cn_ce_sunburst(structures, max_slices_ce=5)  # doctest: +SKIP
        >>> # fig.show() # Uncomment to display
    """
    _chemenv_settings = chemenv_settings or {}

    all_cn_ce_counts: list[dict[str, Any]] = []
    all_coord_geoms = AllCoordinationGeometries()
    symbol_cn_map = all_coord_geoms.get_symbol_cn_mapping()

    structures = normalize_structures(structures)

    for struct_idx, struct in enumerate(structures.values()):
        structure_ce_counts: dict[tuple[int, str], float] = {}
        try:
            lgf = LocalGeometryFinder()
            lgf.setup_structure(struct)

            compute_se_kwargs: dict[str, Any] = {}
            if "maximum_distance_factor" in _chemenv_settings:
                compute_se_kwargs["maximum_distance_factor"] = _chemenv_settings[
                    "maximum_distance_factor"
                ]
            elif "distance_cutoff" in _chemenv_settings:  # alias
                compute_se_kwargs["maximum_distance_factor"] = _chemenv_settings[
                    "distance_cutoff"
                ]

            if "minimum_angle_factor" in _chemenv_settings:
                compute_se_kwargs["minimum_angle_factor"] = _chemenv_settings[
                    "minimum_angle_factor"
                ]
            elif "angle_cutoff" in _chemenv_settings:  # alias
                compute_se_kwargs["minimum_angle_factor"] = _chemenv_settings[
                    "angle_cutoff"
                ]

            # Pass through other known LGF settings directly
            for key in (
                "voronoi_normalized_distance_tolerance",
                "voronoi_normalized_angle_tolerance",
                "valences",
            ):
                if key in _chemenv_settings:
                    compute_se_kwargs[key] = _chemenv_settings[key]

            se_detailed: StructureEnvironments = lgf.compute_structure_environments(
                **compute_se_kwargs
            )

            # Determine strategy parameters
            # Use Pymatgen strategy defaults unless specific strategy settings are provided
            strat_dist_cutoff = _chemenv_settings.get(
                "strategy_distance_cutoff",
                _chemenv_settings.get("distance_cutoff", 1.4),
            )
            strat_angle_cutoff = _chemenv_settings.get(
                "strategy_angle_cutoff", _chemenv_settings.get("angle_cutoff", 0.3)
            )

            strategy = SimplestChemenvStrategy(
                distance_cutoff=strat_dist_cutoff, angle_cutoff=strat_angle_cutoff
            )
            lse = LightStructureEnvironments.from_structure_environments(
                strategy=strategy, structure_environments=se_detailed
            )
        except Exception as exc:
            print(
                f"Skipping structure {struct_idx} ({struct.formula}) due to ChemEnv error: {exc}"
            )
            continue

        if not lse.coordination_environments:
            continue

        for site_idx in range(len(struct)):
            if not lse.coordination_environments[site_idx]:
                continue
            for ce_info in lse.coordination_environments[site_idx]:
                ce_symbol = ce_info["ce_symbol"]
                cn_val = symbol_cn_map.get(ce_symbol)
                if cn_val is None:
                    if ce_symbol == "S:1":
                        cn_val = 1
                    elif ce_symbol.startswith("M:"):
                        try:
                            cn_val = int(ce_symbol.split(":")[1])
                        except (ValueError, IndexError):
                            cn_val = 0
                    else:  # Includes "NULL" or other unmapped
                        cn_val = 0

                if cn_val > 0:
                    key = (cn_val, ce_symbol)
                    structure_ce_counts[key] = structure_ce_counts.get(key, 0) + 1.0

        if normalize and structure_ce_counts:
            total_ce_for_struct = sum(structure_ce_counts.values())
            if total_ce_for_struct > 0:
                for key_in_struct in structure_ce_counts:
                    structure_ce_counts[key_in_struct] /= total_ce_for_struct

        for (cn_val_agg, ce_sym_agg), count_val_agg in structure_ce_counts.items():
            all_cn_ce_counts.append(
                {
                    Key.coord_num: cn_val_agg,
                    Key.chem_env_symbol: ce_sym_agg,
                    Key.count: count_val_agg,
                }
            )

    if not all_cn_ce_counts:
        fig = px.sunburst(
            pd.DataFrame(columns=[Key.coord_num, Key.chem_env_symbol, Key.count])
        )
        fig.layout.title = {"text": "No CN/CE data to display.", "x": 0.5}
        return fig

    df_counts = pd.DataFrame(all_cn_ce_counts)
    df_grouped = df_counts.groupby(
        [Key.coord_num, Key.chem_env_symbol], as_index=False
    )[Key.count].sum()

    if max_slices_cn is not None and max_slices_cn > 0:
        cn_summary = (
            df_grouped.groupby(Key.coord_num)[Key.count]
            .sum()
            .sort_values(ascending=False)
        )
        if len(cn_summary) > max_slices_cn:
            top_cns = cn_summary.head(max_slices_cn).index
            df_top_cns_details = df_grouped[df_grouped[Key.coord_num].isin(top_cns)]

            if max_slices_mode == "other":
                other_cns_data = cn_summary.iloc[max_slices_cn:]
                other_cn_entry = pd.DataFrame(
                    [
                        {
                            Key.coord_num: f"Other CNs ({len(other_cns_data)} more)",
                            Key.chem_env_symbol: "",
                            Key.count: other_cns_data.sum(),
                        }
                    ]
                )
                df_grouped = pd.concat(
                    [df_top_cns_details, other_cn_entry], ignore_index=True
                )
            else:  # "drop" mode
                df_grouped = df_top_cns_details

    df_grouped["cn_sort_key"] = pd.to_numeric(
        df_grouped[Key.coord_num], errors="coerce"
    ).fillna(float("inf"))
    df_grouped = df_grouped.sort_values(
        by=["cn_sort_key", Key.count], ascending=[True, False]
    ).drop(columns=["cn_sort_key"])

    if max_slices_ce is not None and max_slices_ce > 0:
        df_grouped = _limit_slices(
            df_grouped,
            group_col=Key.coord_num,
            count_col=Key.count,
            max_slices=max_slices_ce,
            max_slices_mode=max_slices_mode,
            child_col_for_other_label=Key.chem_env_symbol,
            other_label=None,  # Use default "Other (N more not shown)"
        )

    sunburst_defaults = dict(color_discrete_sequence=px.colors.qualitative.Plotly)
    fig = px.sunburst(
        df_grouped,
        path=[Key.coord_num, Key.chem_env_symbol],
        values=Key.count,
        **sunburst_defaults | kwargs,
    )

    if fig.data:  # Check if data exists (i.e., not an empty plot)
        if show_counts == "value":
            fig.data[0].texttemplate = "%{label}: %{value}"
            fig.data[0].textinfo = "none"
        elif show_counts == "percent":
            fig.data[0].texttemplate = "%{label}: %{percentParent:.1%}"
            fig.data[0].textinfo = "none"
        elif show_counts == "value+percent":
            fig.data[0].texttemplate = "%{label}: %{value} (%{percentParent:.1%})"
            fig.data[0].textinfo = "none"
        elif not show_counts:  # show_counts is False
            fig.data[0].texttemplate = "%{label}"
            fig.data[0].textinfo = "none"
        elif show_counts not in ("value", "percent", "value+percent", False):
            raise ValueError(
                f"Invalid {show_counts=}, must be 'value', 'percent', "
                "'value+percent', or False"
            )

    fig.update_layout(
        margin=dict(l=10, r=10, b=10, pad=10),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        title_x=0.5,
    )

    return fig
