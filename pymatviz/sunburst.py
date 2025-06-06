"""Hierarchical multi-level pie charts (i.e. sunbursts).

E.g. for crystal symmetry distributions.
"""

from __future__ import annotations

import textwrap
import warnings
from typing import TYPE_CHECKING, Literal, get_args

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pymatviz as pmv
from pymatviz import chem_env
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
        fig.data[0].texttemplate = "%{label}<br>N=%{value:.2f}"
    elif show_counts == "value+percent":
        fig.data[0].textinfo = "label+value+percent entry"
        fig.data[0].texttemplate = "%{label}<br>N=%{value:.2f}<br>%{percentEntry}"
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


def chem_env_sunburst(
    structures: Structure | Sequence[Structure],
    *,
    chem_env_settings: dict[str, Any] | Literal["chemenv", "crystal_nn"] = "crystal_nn",
    max_slices_cn: int | None = None,
    max_slices_ce: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    show_counts: ShowCounts = "value",
    normalize: bool = False,
) -> go.Figure:
    """Create sunburst plot of coordination numbers and environments.

    Uses pymatgen's ChemEnv module by default for detailed geometric analysis with
    dozens of cutoffs to determine coordination numbers and chemical environment
    symbols (T:4, O:6, etc.), which is comprehensive but slow.     For faster analysis,
    use chem_env_settings="crystal_nn" to employ CrystalNN
    (Zimmerman et al. 2017, DOI: 10.3389/fmats.2017.00034),
    which provides coordination numbers and approximate environment classification
    using order parameters but with less detailed geometric information.

    Performance Note:
        Based on informal benchmarks, "crystal_nn" is ~90x faster than "chemenv"
        (e.g., 0.025s vs 2.25s for a small test set), which is why "crystal_nn"
        is the default. For large datasets or interactive applications, "crystal_nn"
        provides a good speed vs detail trade-off, while "chemenv" may give
        better geometric accuracy.

    Args:
        structures (Structure | Sequence[Structure]): Structures to analyze.
        chem_env_settings (dict[str, Any] | "chemenv" | "crystal_nn"): Analysis method.
            - "crystal_nn" (default): Use CrystalNN (faster)
            - "chemenv": Use ChemEnv module (slower)
            - dict: Custom ChemEnv settings (original behavior)
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
    if chem_env_settings == "crystal_nn":
        return _chem_env_sunburst_crystal_nn(
            structures=structures,
            max_slices_cn=max_slices_cn,
            max_slices_ce=max_slices_ce,
            max_slices_mode=max_slices_mode,
            show_counts=show_counts,
            normalize=normalize,
        )
    # Handle legacy case and explicit "chemenv"
    if chem_env_settings == "chemenv":
        chem_env_settings = {}

    # Original ChemEnv implementation
    return _chem_env_sunburst_chem_env(
        structures=structures,
        chem_env_settings=chem_env_settings,
        max_slices_cn=max_slices_cn,
        max_slices_ce=max_slices_ce,
        max_slices_mode=max_slices_mode,
        show_counts=show_counts,
        normalize=normalize,
    )


def _chem_env_sunburst_chem_env(
    structures: Structure | Sequence[Structure],
    *,
    chem_env_settings: dict[str, Any],
    max_slices_cn: int | None = None,
    max_slices_ce: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    show_counts: ShowCounts = "value",
    normalize: bool = False,
) -> go.Figure:
    """ChemEnv-based implementation of chem_env_sunburst."""
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometries as coord_geoms  # noqa: E501
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder as coord_finder  # noqa: E501
    import pymatgen.analysis.chemenv.coordination_environments.structure_environments as struct_envs  # noqa: E501
    from pymatgen.analysis.chemenv.coordination_environments import chemenv_strategies

    structs = normalize_structures(structures).values()

    if show_counts not in get_args(ShowCounts):
        raise ValueError(f"Invalid {show_counts=}")

    settings = chem_env_settings or {}
    chem_env_data: list[dict[str, Any]] = []

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

                for env_list in lse.coordination_environments or []:
                    for coord_env in env_list or []:
                        ce_symbol = coord_env["ce_symbol"]
                        cn_val = chem_env.get_cn_from_symbol(
                            ce_symbol, symbol_cn_mapping
                        )
                        key = (cn_val, ce_symbol)
                        coord_envs_dict[key] = coord_envs_dict.get(key, 0) + 1

                for (cn_val, ce_symbol), env_count in coord_envs_dict.items():
                    final_count = env_count
                    if normalize:
                        total = sum(coord_envs_dict.values())
                        if total > 0:
                            final_count = env_count / total

                    chem_env_dict = dict(
                        coord_num=cn_val, chem_env_symbol=ce_symbol, count=final_count
                    )
                    chem_env_data.append(chem_env_dict)

            except (ImportError, RuntimeError) as exc:
                warnings.warn(
                    f"ChemEnv analysis failed for structure: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

    except (ImportError, RuntimeError) as exc:
        warnings.warn(f"ChemEnv setup failed: {exc}", UserWarning, stacklevel=2)

    return _process_chem_env_data_sunburst(
        chem_env_data=chem_env_data,
        max_slices_cn=max_slices_cn,
        max_slices_ce=max_slices_ce,
        max_slices_mode=max_slices_mode,
        show_counts=show_counts,
    )


def _chem_env_sunburst_crystal_nn(
    structures: Structure | Sequence[Structure],
    *,
    max_slices_cn: int | None = None,
    max_slices_ce: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    show_counts: ShowCounts = "value",
    normalize: bool = False,
) -> go.Figure:
    """CrystalNN-based implementation of chem_env_sunburst (faster but may be less
    accurate, not benchmarked so unclear).
    """
    from pymatgen.analysis.local_env import CrystalNN

    structs = normalize_structures(structures).values()

    if show_counts not in get_args(ShowCounts):
        raise ValueError(f"Invalid {show_counts=}")

    chem_env_data: list[dict[str, Any]] = []
    crystal_nn = CrystalNN()

    try:
        for structure in structs:
            try:
                # Get coordination info for each site
                for site_idx in range(len(structure)):
                    # Get coordination number
                    nn_info = crystal_nn.get_nn_info(structure, site_idx)
                    cn_val = len(nn_info)

                    # Get best matching coordination environment using order parameters
                    ce_symbol = chem_env.classify_local_env_with_order_params(
                        structure, site_idx, cn_val
                    )

                    # Add to data
                    final_count = 1.0
                    if normalize:
                        final_count = 1.0 / len(structure)  # Normalize per structure

                    chem_env_dict = dict(
                        coord_num=cn_val, chem_env_symbol=ce_symbol, count=final_count
                    )
                    chem_env_data.append(chem_env_dict)

            except (ImportError, RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"CrystalNN analysis failed for structure: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

    except (ImportError, RuntimeError) as exc:
        warnings.warn(f"CrystalNN setup failed: {exc}", UserWarning, stacklevel=2)

    return _process_chem_env_data_sunburst(
        chem_env_data=chem_env_data,
        max_slices_cn=max_slices_cn,
        max_slices_ce=max_slices_ce,
        max_slices_mode=max_slices_mode,
        show_counts=show_counts,
    )


def _process_chem_env_data_sunburst(
    chem_env_data: list[dict[str, Any]],
    max_slices_cn: int | None,
    max_slices_ce: int | None,
    max_slices_mode: Literal["other", "drop"],
    show_counts: ShowCounts,
) -> go.Figure:
    """Process chem env data and create sunburst plot."""
    if not chem_env_data:
        fig = go.Figure()
        fig.layout.title = dict(
            text="No CN/CE data to display",
            x=0.5,
            font_size=16,
        )
        return fig

    df_chem_env = pd.DataFrame(chem_env_data)
    df_chem_env = df_chem_env.groupby(["coord_num", "chem_env_symbol"], as_index=False)[
        "count"
    ].sum()

    # Limit CN slices
    if max_slices_cn:
        df_cn_grouped = df_chem_env.groupby("coord_num", as_index=False)["count"].sum()
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
            df_chem_env = df_chem_env[df_chem_env["coord_num"].isin(selected_cns)]
        else:
            # For "other" mode, combine excluded CNs
            selected_cns = {
                cn
                for cn in df_cn_grouped["coord_num"]
                if not str(cn).startswith("Other")
            }

            excluded_data = df_chem_env[~df_chem_env["coord_num"].isin(selected_cns)]
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
                df_chem_env = pd.concat(
                    [
                        df_chem_env[df_chem_env["coord_num"].isin(selected_cns)],
                        other_entry,
                    ],
                    ignore_index=True,
                )

    # Limit CE slices within each CN
    if max_slices_ce:
        df_ce_limited = []
        for cn_val in df_chem_env["coord_num"].unique():
            df_cn_subset = df_chem_env[df_chem_env["coord_num"] == cn_val]
            df_cn_subset = _limit_slices(
                df_cn_subset,
                group_col="chem_env_symbol",
                count_col="count",
                max_slices=max_slices_ce,
                max_slices_mode=max_slices_mode,
                other_label="Other CEs",
                child_col_for_other_label="chem_env_symbol",
            )
            df_ce_limited.append(df_cn_subset)
        df_chem_env = pd.concat(df_ce_limited, ignore_index=True)

    # Apply text wrapping to chem env symbols to allow for larger font in small cells
    df_chem_env["chem_env_symbol"] = df_chem_env["chem_env_symbol"].map(
        lambda text: "<br>".join(
            textwrap.wrap(text, width=15, break_long_words=True, break_on_hyphens=True)
        )
    )

    fig = px.sunburst(
        df_chem_env, path=["coord_num", "chem_env_symbol"], values="count"
    )

    # Apply text formatting
    text_templates = {
        "value": "%{label}: %{value:.2f}",
        "percent": "%{label}: %{percentParent:.1%}",
        "value+percent": "%{label}: %{value:.2f} (%{percentParent:.1%})",
        False: "%{label}",
    }

    fig.data[0].update(
        texttemplate=text_templates[show_counts],
        textinfo="none",
        marker=dict(line=dict(color="white")),
    )

    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
