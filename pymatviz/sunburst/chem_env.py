"""Sunburst plot of coordination numbers and environments."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pymatviz import chem_env
from pymatviz.process_data import normalize_structures
from pymatviz.sunburst.helpers import _limit_slices
from pymatviz.typing import AnyStructure, ShowCounts


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def chem_env_sunburst(
    structures: AnyStructure | Sequence[AnyStructure],
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
        structures (AnyStructure | Sequence[AnyStructure]): Structures to analyze.
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
    structs = normalize_structures(structures).values()

    if show_counts not in get_args(ShowCounts):
        raise ValueError(f"Invalid {show_counts=}")

    if chem_env_settings == "crystal_nn":
        chem_env_data = chem_env.collect_coord_envs_crystal_nn(
            structs, normalize=normalize
        )
    else:  # "chemenv" (legacy default) or custom ChemEnv settings dict
        settings = {} if chem_env_settings == "chemenv" else chem_env_settings
        chem_env_data = chem_env.collect_coord_envs_chemenv(
            structs, chem_env_settings=settings, normalize=normalize
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
