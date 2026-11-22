"""Package paths, element lookups (from kemio's element table) and DataFrame column
extraction.
"""

from __future__ import annotations

import os
from typing import Any

import kemio
import numpy as np
import pandas as pd


PKG_DIR: str = os.path.dirname(__file__)
ROOT: str = os.path.dirname(PKG_DIR)

# kemio's element properties (masses, radii, ...), one row per element in Z order
element_columns = kemio.element_table().as_dict()
df_ptable: pd.DataFrame = pd.DataFrame(element_columns).set_index("symbol")


atomic_numbers: dict[str, int] = {
    symbol: atomic_num for atomic_num, symbol in enumerate(df_ptable.index, start=1)
}
element_symbols: dict[int, str] = {
    atomic_num: symbol for symbol, atomic_num in atomic_numbers.items()
}


def df_to_arrays(df: pd.DataFrame | None, *args: Any) -> list[Any]:
    """Columns of df (rows with NaN in any of them dropped) if df is given, else args
    as arrays. A list of column names becomes a ``{col: array}`` dict.

    Raises:
        TypeError: If df is neither a DataFrame nor None.
        ValueError: If df is None but args holds column names, or df is given and args
            holds arrays.
    """
    if df is None:
        if cols := [arg for arg in args if isinstance(arg, str)]:
            raise ValueError(f"got column names but no df to get data from: {cols}")
        return list(map(np.asarray, args))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df should be pandas DataFrame or None, got {type(df)}")
    is_col = [isinstance(arg, str | int) for arg in args]
    if bad := [
        arg
        for arg, single in zip(args, is_col, strict=True)
        if not single
        and not (
            isinstance(arg, list | tuple) and all(isinstance(c, str | int) for c in arg)
        )
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
