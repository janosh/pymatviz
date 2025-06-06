"""Helper functions for sunburst plots."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from pymatviz.enums import Key


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
