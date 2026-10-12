"""Sunburst plot of coordination numbers and environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import plotly.express as px

from pymatviz import chem_env
from pymatviz.sunburst.helpers import _limit_slices, _limit_slices_per_group


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import plotly.graph_objects as go

    from pymatviz.chem_env import ChemEnvSettings
    from pymatviz.typing import AnyStructure, ShowCounts


def chem_env_sunburst(
    structures: AnyStructure | Sequence[AnyStructure],
    *,
    chem_env_settings: ChemEnvSettings = "crystal_nn",
    max_slices_cn: int | None = None,
    max_slices_ce: int | None = None,
    max_slices_mode: Literal["other", "drop"] = "other",
    show_counts: ShowCounts = "value",
    normalize: bool = False,
) -> go.Figure:
    """Create sunburst plot of coordination numbers and environments.

    Uses CrystalNN by default; select "chemenv" for detailed geometric analysis.

    Args:
        structures (AnyStructure | Sequence[AnyStructure]): Structures to analyze.
        chem_env_settings (dict[str, Any] | "chemenv" | "crystal_nn"): Analysis method.
            - "crystal_nn" (default): Use CrystalNN (faster)
            - "chemenv": Use ChemEnv module (slower)
            - dict: Custom ChemEnv settings
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
        ValueError: For invalid show_counts, chem_env_settings or max_slices_mode.
    """
    if max_slices_mode not in ("other", "drop"):
        raise ValueError(f"Invalid {max_slices_mode=}, must be 'other' or 'drop'")
    chem_env_data = chem_env.collect_chem_env_data(
        structures,
        chem_env_settings=chem_env_settings,
        normalize=normalize,
        show_counts=show_counts,
    )

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
        return chem_env.empty_chem_env_figure()

    df_chem_env = chem_env.aggregate_chem_env_data(chem_env_data)

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

    if max_slices_ce:  # Limit CE slices within each CN
        df_chem_env = _limit_slices_per_group(
            df_chem_env,
            group_col="coord_num",
            count_col="count",
            max_slices=max_slices_ce,
            max_slices_mode=max_slices_mode,
            other_label="Other CEs",
            child_col_for_other_label="chem_env_symbol",
        )

    df_chem_env["chem_env_symbol"] = df_chem_env["chem_env_symbol"].map(
        chem_env.wrap_chem_env_label
    )

    fig = px.sunburst(
        df_chem_env, path=["coord_num", "chem_env_symbol"], values="count"
    )
    fig.data[0].update(
        texttemplate=chem_env.CHEM_ENV_TEXT_TEMPLATES[show_counts],
        textinfo="none",
        marker=dict(line=dict(color="white")),
    )

    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
