"""Sunburst plot of crystal systems."""

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.process_data import is_structure_like
from pymatviz.sunburst.helpers import (
    _apply_sunburst_show_counts,
    _limit_slices_per_group,
)
from pymatviz.typing import ShowCounts


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
    if len(data) == 0:
        raise ValueError("spacegroup_sunburst requires non-empty data")

    # materialize as list[Any] for Series-safe positional access and untyped iteration
    values: list[Any] = data.tolist() if isinstance(data, pd.Series) else list(data)
    if is_structure_like(values[0]):  # if 1st item is structure-like, assume all are
        try:
            from moyopy import MoyoDataset
            from moyopy.interface import MoyoAdapter
        except ImportError as exc:
            raise RuntimeError(
                "moyopy is required to pass Structure objects to "
                "spacegroup_sunburst. Install it with `pip install moyopy`."
            ) from exc

        spg_nums: list[int] = []
        for idx, struct in enumerate(values):
            try:
                spg_nums.append(MoyoDataset(MoyoAdapter.from_py_obj(struct)).number)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(
                    "Could not determine space group for structure at index "
                    f"{idx} ({type(struct).__name__}). Pass periodic pymatgen "
                    "Structure objects or ASE Atoms supported by moyopy."
                ) from exc
        series = pd.Series(spg_nums)
    else:
        series = pd.Series(values)

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
    df_spg_counts = _limit_slices_per_group(
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

    _apply_sunburst_show_counts(fig, show_counts)

    fig.layout.margin = dict(l=10, r=10, b=10, pad=10)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
