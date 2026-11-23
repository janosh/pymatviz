"""Package paths, element lookups (from kemio's element table) and DataFrame column
extraction.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import kemio
import numpy as np
import pandas as pd


PKG_DIR: str = os.path.dirname(__file__)
ROOT: str = os.path.dirname(PKG_DIR)

# kemio's element properties (masses, radii, ...), one row per element in Z order
df_ptable = pd.DataFrame(kemio.element_table().as_dict()).set_index("symbol")


atomic_numbers: dict[str, int] = {
    symbol: atomic_num for atomic_num, symbol in enumerate(df_ptable.index, start=1)
}
element_symbols: dict[int, str] = {
    atomic_num: symbol for symbol, atomic_num in atomic_numbers.items()
}


def check_unique_labels(labels: Any, name: str) -> None:
    """Raise ValueError naming the repeated labels of a pd.Index or a Series' index,
    which dict conversion would silently drop. Other types pass unchecked.
    """
    index = labels.index if isinstance(labels, pd.Series) else labels
    if isinstance(index, pd.Index) and index.has_duplicates:
        dupes = index[index.duplicated()].unique().tolist()
        raise ValueError(f"{name} must be unique, got duplicates {dupes}")


def str_labels[T](labeled: Mapping[Any, T]) -> dict[str, T]:
    """Mapping with its keys as strings.

    Raises:
        ValueError: If distinct keys share a string form (e.g. 1 and "1").
    """
    as_str = {str(key): val for key, val in labeled.items()}
    if len(as_str) < len(labeled):
        raise ValueError(f"labels must be unique as strings, got {list(labeled)}")
    return as_str


def align_series(*values: Any) -> list[Any]:
    """Align the pandas Series among values (top-level or values of a Mapping, like
    a dict of named predictions) on their shared index labels (inner join, in the
    first Series' order), so paired data match by label rather than position. Other
    values pass through unchanged. A no-op if fewer than 2 Series or all their
    indexes are equal.

    Raises:
        ValueError: If a Series index has duplicates, the indexes share no label, or
            they differ while plain arrays (which can't be aligned) are present.
    """
    flat = [
        val
        for value in values
        for val in (value.values() if isinstance(value, Mapping) else [value])
    ]
    all_series = [val for val in flat if isinstance(val, pd.Series)]
    if len(all_series) < 2:
        return list(values)
    for srs in all_series:
        check_unique_labels(srs, f"index of Series {srs.name!r} (to align on)")
    first_index = all_series[0].index
    if all(srs.index.equals(first_index) for srs in all_series[1:]):
        return list(values)
    if plain := [
        type(val).__name__
        for val in flat
        # nested dicts (e.g. per-classifier kwargs) count as unalignable too
        if not isinstance(val, pd.Series | str | None)
        and (isinstance(val, Mapping) or np.ndim(val) > 0)
    ]:
        raise ValueError(
            f"can't align pandas Series with differing indexes against unlabeled "
            f"{plain}: pass all as Series or all as arrays"
        )
    common = first_index
    for srs in all_series[1:]:
        common = common.intersection(srs.index, sort=False)
    if common.empty:
        names = [srs.name for srs in all_series]
        raise ValueError(f"pandas Series {names} share no index labels to align on")

    def aligned(val: Any) -> Any:
        """Val reindexed to the common labels if a Series, else val."""
        return val.loc[common] if isinstance(val, pd.Series) else val

    return [
        {key: aligned(val) for key, val in value.items()}
        if isinstance(value, Mapping)
        else aligned(value)
        for value in values
    ]


def df_to_arrays(df: pd.DataFrame | None, *args: Any) -> list[Any]:
    """Columns of df (rows with NaN in any of them dropped) if df is given, else args
    as arrays (pandas Series aligned by index, see align_series). A list of column
    names becomes a ``{col: array}`` dict.

    Raises:
        TypeError: If df is neither a DataFrame nor None.
        ValueError: If df is None but args holds column names, or df is given and args
            holds arrays.
    """
    if df is None:
        if cols := [arg for arg in args if isinstance(arg, str)]:
            raise ValueError(f"got column names but no df to get data from: {cols}")
        return list(map(np.asarray, align_series(*args)))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df should be pandas DataFrame or None, got {type(df)}")
    is_col = [isinstance(arg, str | int) for arg in args]
    is_col_list = [
        isinstance(arg, list | tuple) and all(isinstance(col, str | int) for col in arg)
        for arg in args
    ]
    if bad := [
        arg
        for arg, single, multi in zip(args, is_col, is_col_list, strict=True)
        if not (single or multi)
    ]:
        raise ValueError(
            f"with a df, pass column names (or lists of them), not data, got {bad}"
        )
    flat_cols = [
        col
        for arg, single in zip(args, is_col, strict=True)
        for col in ([arg] if single else arg)
    ]
    df_no_nan = df.dropna(subset=flat_cols)
    return [
        df_no_nan[arg].to_numpy()
        if single
        else {col: df_no_nan[col].to_numpy() for col in arg}
        for arg, single in zip(args, is_col, strict=True)
    ]
