"""Coordination number and chemical environment treemap visualizations."""

from __future__ import annotations

import textwrap
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, get_args

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pymatviz import chem_env
from pymatviz.process_data import normalize_structures
from pymatviz.typing import AnyStructure, ShowCounts


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


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
    chem_env_settings: dict[str, Any] | Literal["chemenv", "crystal_nn"] = "crystal_nn",
    max_cells_cn: int | None = None,
    max_cells_ce: int | None = None,
    normalize: bool = False,
    show_counts: ShowCounts = "value",
    cn_formatter: CnFormatter = None,
    **kwargs: Any,
) -> go.Figure:
    """Create treemap plot of coordination numbers and chemical environments.

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
        chem_env_settings (dict[str, Any] | "chemenv" | "crystal_nn"):
            Analysis method.
            - "crystal_nn" (default): Use CrystalNN (faster)
            - "chemenv": Use ChemEnv module (slower)
            - dict: Custom ChemEnv settings (original behavior)
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
        ValueError: For invalid inputs

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
        >>> # Use faster CrystalNN analysis
        >>> fig4 = pmv.chem_env_treemap(structures, chem_env_settings="crystal_nn")
    """
    if chem_env_settings == "crystal_nn":
        return _chem_env_treemap_crystal_nn(
            structures=structures,
            max_cells_cn=max_cells_cn,
            max_cells_ce=max_cells_ce,
            normalize=normalize,
            show_counts=show_counts,
            cn_formatter=cn_formatter,
            **kwargs,
        )
    # Handle legacy case and explicit "chemenv"
    if chem_env_settings == "chemenv":
        chem_env_settings = {}

    # Original ChemEnv implementation
    return _chem_env_treemap_chem_env(
        structures=structures,
        chem_env_settings=chem_env_settings,
        max_cells_cn=max_cells_cn,
        max_cells_ce=max_cells_ce,
        normalize=normalize,
        show_counts=show_counts,
        cn_formatter=cn_formatter,
        **kwargs,
    )


def _chem_env_treemap_chem_env(
    structures: AnyStructure | Sequence[AnyStructure],
    *,
    chem_env_settings: dict[str, Any],
    max_cells_cn: int | None = None,
    max_cells_ce: int | None = None,
    normalize: bool = False,
    show_counts: ShowCounts = "value",
    cn_formatter: CnFormatter = None,
    **kwargs: Any,
) -> go.Figure:
    """ChemEnv-based implementation of chem_env_treemap."""
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

                for coord_envs in lse.coordination_environments or []:
                    for coord_env in coord_envs or []:
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

    return _process_chem_env_data_treemap(
        chem_env_data=chem_env_data,
        max_cells_cn=max_cells_cn,
        max_cells_ce=max_cells_ce,
        show_counts=show_counts,
        cn_formatter=cn_formatter,
        **kwargs,
    )


def _chem_env_treemap_crystal_nn(
    structures: AnyStructure | Sequence[AnyStructure],
    *,
    max_cells_cn: int | None = None,
    max_cells_ce: int | None = None,
    normalize: bool = False,
    show_counts: ShowCounts = "value",
    cn_formatter: CnFormatter = None,
    **kwargs: Any,
) -> go.Figure:
    """CrystalNN-based implementation of chem_env_treemap (faster but may be less
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

    # Add counts and percentages to CN labels if requested
    if cn_formatter is None and show_counts is not False:
        cn_formatter = default_cn_formatter
    if (
        cn_formatter is not False
        and cn_formatter is not None
        and show_counts is not False
    ):
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
            # Take the top N CNs
            top_cns = set(cn_totals.iloc[:max_cells_cn].index)

            # Filter data to keep only top CNs
            top_cn_data = df_chem_env[df_chem_env["coord_num"].isin(top_cns)]

            # Create "Other CNs" entry for the remaining CNs
            other_cn_data = df_chem_env[~df_chem_env["coord_num"].isin(top_cns)]
            if not other_cn_data.empty:
                other_count = other_cn_data["count"].sum()
                num_omitted = len(cn_totals) - max_cells_cn

                other_entry = pd.DataFrame(
                    {
                        "coord_num": ["Other CNs"],
                        "coord_num_display": [
                            f"Other CNs ({num_omitted} more not shown)"
                        ],
                        "chem_env_symbol": [
                            f"Other CNs ({num_omitted} more not shown)"
                        ],
                        "count": [other_count],
                    }
                )

                df_chem_env = pd.concat([top_cn_data, other_entry], ignore_index=True)
            else:
                df_chem_env = top_cn_data

    # Limit the number of CEs within each CN if requested
    if max_cells_ce is not None and max_cells_ce > 0:
        df_ce_limited = []

        for cn_val in df_chem_env["coord_num"].unique():
            cn_subset = df_chem_env[df_chem_env["coord_num"] == cn_val]

            if len(cn_subset) > max_cells_ce:
                # Sort by count and take top CEs
                cn_subset_sorted = cn_subset.sort_values("count", ascending=False)
                top_ces = cn_subset_sorted.iloc[:max_cells_ce]

                # Create "Other CEs" entry for remaining CEs
                other_ces = cn_subset_sorted.iloc[max_cells_ce:]
                if not other_ces.empty:
                    other_count = other_ces["count"].sum()
                    num_omitted = len(cn_subset) - max_cells_ce

                    # Get the coord_num_display for this CN
                    coord_num_display = top_ces["coord_num_display"].iloc[0]

                    other_entry = pd.DataFrame(
                        {
                            "coord_num": [cn_val],
                            "coord_num_display": [coord_num_display],
                            "chem_env_symbol": [
                                f"Other CEs ({num_omitted} more not shown)"
                            ],
                            "count": [other_count],
                        }
                    )

                    df_ce_limited.append(
                        pd.concat([top_ces, other_entry], ignore_index=True)
                    )
                else:
                    df_ce_limited.append(top_ces)
            else:
                df_ce_limited.append(cn_subset)

        df_chem_env = pd.concat(df_ce_limited, ignore_index=True)

    # Apply text wrapping to chem env symbols to allow for larger font in small cells
    df_chem_env["chem_env_symbol"] = df_chem_env["chem_env_symbol"].map(
        lambda text: "<br>".join(
            textwrap.wrap(text, width=15, break_long_words=True, break_on_hyphens=True)
        )
    )

    # Create the treemap
    treemap_kwargs = {
        "path": ["coord_num_display", "chem_env_symbol"],
        "values": "count",
        **kwargs,
    }

    # Handle show_counts parameter
    if show_counts == "value":
        treemap_kwargs["labels"] = {
            "coord_num_display": "Coordination Number",
            "chem_env_symbol": "Chemical Environment",
            "count": "Count",
        }
    elif show_counts == "percent":
        treemap_kwargs["labels"] = {
            "coord_num_display": "Coordination Number",
            "chem_env_symbol": "Chemical Environment",
        }
        # For percent mode, plotly will automatically calculate percentages
    elif show_counts == "value+percent":
        treemap_kwargs["labels"] = {
            "coord_num_display": "Coordination Number",
            "chem_env_symbol": "Chemical Environment",
            "count": "Count",
        }
    # If show_counts is False, we don't modify labels

    fig = px.treemap(df_chem_env, **treemap_kwargs)

    # Set background color to transparent
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
