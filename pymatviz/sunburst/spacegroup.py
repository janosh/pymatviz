"""Sunburst plot of crystal systems."""

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.process_data import get_spacegroup_number, is_structure_like
from pymatviz.typing import ShowCounts
from pymatviz.utils.plotting import apply_show_counts


def spacegroup_sunburst(
    data: Sequence[int | str | Structure] | pd.Series,
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

    Raises:
        ValueError: If data is empty.
    """
    if max_slices_mode not in ("other", "drop"):
        raise ValueError(f"Invalid {max_slices_mode=}, must be 'other' or 'drop'")
    if len(data) == 0:
        raise ValueError("spacegroup_sunburst requires non-empty data")

    # materialize as list[Any] for Series-safe positional access and untyped iteration
    values: list[Any] = data.tolist() if isinstance(data, pd.Series) else list(data)
    if is_structure_like(values[0]):  # if 1st item is structure-like, assume all are
        spg_nums: list[int] = []
        for idx, struct in enumerate(values):
            try:
                spg_nums.append(get_spacegroup_number(struct))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(
                    "Could not determine space group for structure at index "
                    f"{idx} ({type(struct).__name__}). Pass periodic pymatgen "
                    "Structure objects or ASE Atoms supported by moyopy."
                ) from exc
        series = pd.Series(spg_nums)
    else:
        series = pd.Series(values)
    if series.isna().any():
        raise ValueError("Space group data must not contain missing values")

    df_spg_counts = pd.DataFrame(series.value_counts().reset_index())
    df_spg_counts.columns = [Key.spg_num, "count"]

    df_spg_counts[Key.crystal_system] = df_spg_counts[Key.spg_num].map(
        pmv.utils.spg_to_crystal_sys
    )

    if max_slices and max_slices > 0:
        ranks = df_spg_counts.groupby(Key.crystal_system)["count"].rank(
            ascending=False, method="first"
        )
        hidden = df_spg_counts[ranks > max_slices]
        df_spg_counts = df_spg_counts[ranks <= max_slices]
        if max_slices_mode == "other" and not hidden.empty:
            groups = hidden.groupby(Key.crystal_system)["count"]
            other = groups.sum().to_frame()
            other[Key.spg_num] = groups.size().map(
                lambda count: f"Other ({count} more not shown)"
            )
            df_spg_counts = pd.concat([df_spg_counts, other.reset_index()])

    sunburst_defaults = dict(color_discrete_sequence=px.colors.qualitative.G10)

    fig = px.sunburst(
        df_spg_counts,
        path=[Key.crystal_system, Key.spg_num],
        values="count",
        **sunburst_defaults | kwargs,
    )

    apply_show_counts(fig, show_counts, value_fmt=".2f")

    fig.layout.margin = dict(l=10, r=10, b=10, pad=10)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
