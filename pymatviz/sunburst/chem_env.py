"""Sunburst plot of coordination numbers and environments."""

from __future__ import annotations

import textwrap
import warnings
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
    structures: AnyStructure | Sequence[AnyStructure],
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
    structures: AnyStructure | Sequence[AnyStructure],
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
