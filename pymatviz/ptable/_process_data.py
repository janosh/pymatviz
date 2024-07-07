"""Utils for data preprocessing and related tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd


# Data types that can be passed to PTableProjector and normalized by data_preprocessor
# to SupportedValueType
SupportedDataType = Union[
    dict[str, Union[float, Sequence[float], np.ndarray]], pd.DataFrame, pd.Series
]

# Data types used internally by ptable plotters (returned by preprocess_ptable_data)
SupportedValueType = Union[Sequence[float], np.ndarray]


def get_df_nest_level(df_in: pd.DataFrame, *, col: str) -> int:
    """Check for maximum nest level in a DataFrame column.

    Definition of nest level:
        Level 0 (no list):       "Fe": 1
        Level 1 (flat list):     "Co": [1, 2]
        Level 2 (nested lists):  "Ni": [[1, 2], [3, 4], ]
        ...

    Args:
        df_in (pd.DataFrame): The DataFrame to check.
        col (str): Name of the column to check.

    Returns:
        int: The maximum nest level.
    """
    return df_in[col].map(lambda val: np.asarray(val).ndim).max()
