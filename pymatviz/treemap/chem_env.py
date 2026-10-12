"""Coordination number and chemical environment treemap visualizations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import pandas as pd
import plotly.express as px

from pymatviz import chem_env


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import plotly.graph_objects as go

    from pymatviz.chem_env import ChemEnvSettings
    from pymatviz.typing import AnyStructure, ShowCounts


CnFormatter = Callable[[int | str, int, int], str] | Literal[False] | None


def default_cn_formatter(coord_num: int | str, count: float, total: float) -> str:
    """Default formatter for coordination number with counts and percentages.

    Args:
        coord_num (int | str): Coordination number
        count (float): Count for this coordination number
        total (float): Total count across all coordination numbers

    Returns:
        str: Formatted string for display
    """
    # Format count to avoid long decimal trails
    if isinstance(count, float):
        if count >= 1:
            count_str = f"{count:.2f}".rstrip("0").rstrip(".")
        else:
            count_str = f"{count:.3f}".rstrip("0").rstrip(".")
    else:
        count_str = f"{count:,}"

    return f"CN {coord_num} (N={count_str}, {count / total:.1%})"


def chem_env_treemap(
    structures: AnyStructure | Sequence[AnyStructure],
    *,
    chem_env_settings: ChemEnvSettings = "crystal_nn",
    max_cells_cn: int | None = None,
    max_cells_ce: int | None = None,
    normalize: bool = False,
    show_counts: ShowCounts = "value",
    cn_formatter: CnFormatter = None,
    **kwargs: Any,
) -> go.Figure:
    """Create treemap plot of coordination numbers and chemical environments.

    Uses CrystalNN by default; select "chemenv" for detailed geometric analysis.

    Args:
        structures (AnyStructure | Sequence[AnyStructure]): Structures to analyze.
        chem_env_settings (dict[str, Any] | "chemenv" | "crystal_nn"):
            Analysis method.
            - "crystal_nn" (default): Use CrystalNN (faster)
            - "chemenv": Use ChemEnv module (slower)
            - dict: Custom ChemEnv settings
        max_cells_cn (int | None): Maximum CN cells to show. Defaults to None.
        max_cells_ce (int | None): Maximum CE cells per CN to show. Defaults to None.
        normalize (bool): Whether to normalize counts per structure. Defaults to False.
        show_counts ("value" | "percent" | "value+percent" | False): How to display
            counts. Defaults to "value".
        cn_formatter (CnFormatter): Custom formatter for CN labels. Defaults to None.
            Can be False to disable formatting.
        **kwargs: Additional keyword arguments passed to plotly.express.treemap.

    Returns:
        Plotly Figure with treemap plot

    Raises:
        ValueError: For invalid show_counts or chem_env_settings.

    Tips and Customization:
    - rounded corners: fig.update_traces(marker=dict(cornerradius=5))
    - colors:
        - discrete: color_discrete_sequence=px.colors.qualitative.Set2
        - custom: color_discrete_map={'CN 4': 'red', 'CN 6': 'blue'}
    - max depth: fig.update_traces(maxdepth=2)
    - patterns/textures: fig.update_traces(marker=dict(pattern=dict(shape=["|"])))
    - hover info: fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}')
    - root color: fig.update_traces(root_color="lightgrey")
    - custom text display: fig.update_traces(textinfo="label+value+percent")

    Example:
        >>> import pymatviz as pmv
        >>> from pymatgen.core import Structure
        >>> # Assuming you have structures
        >>> structures = [...]  # List of Structure objects
        >>> # Basic treemap
        >>> fig1 = pmv.chem_env_treemap(structures)
        >>> # Limit to top 5 CNs and top 3 CEs per CN
        >>> fig2 = pmv.chem_env_treemap(structures, max_cells_cn=5, max_cells_ce=3)
        >>> # Normalize counts per structure
        >>> fig3 = pmv.chem_env_treemap(structures, normalize=True)
    """
    chem_env_data = chem_env.collect_chem_env_data(
        structures,
        chem_env_settings=chem_env_settings,
        normalize=normalize,
        show_counts=show_counts,
    )

    return _process_chem_env_data_treemap(
        chem_env_data=chem_env_data,
        max_cells_cn=max_cells_cn,
        max_cells_ce=max_cells_ce,
        show_counts=show_counts,
        cn_formatter=cn_formatter,
        **kwargs,
    )


def _process_chem_env_data_treemap(
    chem_env_data: list[dict[str, Any]],
    max_cells_cn: int | None,
    max_cells_ce: int | None,
    show_counts: ShowCounts,
    cn_formatter: CnFormatter,
    **kwargs: Any,
) -> go.Figure:
    """Process chem env data and create treemap plot."""
    if not chem_env_data:
        return chem_env.empty_chem_env_figure()

    df_chem_env = chem_env.aggregate_chem_env_data(chem_env_data)

    # Add counts and percentages to CN labels if requested
    if cn_formatter is None:
        cn_formatter = default_cn_formatter
    if cn_formatter is not False and show_counts is not False:
        cn_totals = df_chem_env.groupby("coord_num")["count"].sum()
        total_count = cn_totals.sum()
        df_chem_env["coord_num_display"] = df_chem_env["coord_num"].map(
            lambda x: cn_formatter(x, cn_totals[x], total_count)
        )
    else:
        df_chem_env["coord_num_display"] = df_chem_env["coord_num"].map(
            lambda x: f"CN {x}"
        )

    # Limit the number of CNs if requested
    if max_cells_cn is not None and max_cells_cn > 0:
        # Group by coordination number and sort by count
        cn_totals = (
            df_chem_env.groupby("coord_num")["count"].sum().sort_values(ascending=False)
        )

        if len(cn_totals) > max_cells_cn:
            # Keep the top N CNs and lump the rest into an "Other CNs" entry
            is_top_cn = df_chem_env["coord_num"].isin(cn_totals.index[:max_cells_cn])
            other_label = f"Other CNs ({len(cn_totals) - max_cells_cn} more not shown)"
            other_entry = pd.DataFrame(
                {
                    "coord_num": ["Other CNs"],
                    "coord_num_display": [other_label],
                    "chem_env_symbol": [other_label],
                    "count": [df_chem_env.loc[~is_top_cn, "count"].sum()],
                }
            )
            df_chem_env = pd.concat(
                [df_chem_env[is_top_cn], other_entry], ignore_index=True
            )

    # Limit the number of CEs within each CN if requested
    if max_cells_ce is not None and max_cells_ce > 0:
        df_ce_limited = []

        for cn_val in df_chem_env["coord_num"].unique():
            cn_subset = df_chem_env[df_chem_env["coord_num"] == cn_val]

            if len(cn_subset) > max_cells_ce:
                # Keep the top CEs and lump the rest into an "Other CEs" entry
                cn_subset_sorted = cn_subset.sort_values("count", ascending=False)
                top_ces = cn_subset_sorted.iloc[:max_cells_ce]
                num_omitted = len(cn_subset) - max_cells_ce
                other_entry = pd.DataFrame(
                    {
                        "coord_num": [cn_val],
                        "coord_num_display": [top_ces["coord_num_display"].iloc[0]],
                        "chem_env_symbol": [
                            f"Other CEs ({num_omitted} more not shown)"
                        ],
                        "count": [cn_subset_sorted["count"].iloc[max_cells_ce:].sum()],
                    }
                )
                cn_subset = pd.concat([top_ces, other_entry], ignore_index=True)
            df_ce_limited.append(cn_subset)

        df_chem_env = pd.concat(df_ce_limited, ignore_index=True)

    df_chem_env["chem_env_symbol"] = df_chem_env["chem_env_symbol"].map(
        chem_env.wrap_chem_env_label
    )

    treemap_kwargs = {
        "path": ["coord_num_display", "chem_env_symbol"],
        "values": "count",
    }
    fig = px.treemap(df_chem_env, **treemap_kwargs | kwargs)
    fig.data[0].update(
        texttemplate=chem_env.CHEM_ENV_TEXT_TEMPLATES[show_counts], textinfo="none"
    )

    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
