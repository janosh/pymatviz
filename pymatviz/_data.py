"""Utils for data preprocessing and related tasks."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Union, get_args

import numpy as np
import pandas as pd
from pymatgen.core import Element

from pymatviz.enums import Key


if TYPE_CHECKING:
    from typing import Literal


# Data types used internally by ptable plotters
SupportedValueType = Union[Sequence[float], np.ndarray]

# Data types that can be passed to PTableProjector and data_preprocessor
SupportedDataType = Union[
    dict[str, Union[float, Sequence[float], np.ndarray]], pd.DataFrame, pd.Series
]


def check_for_missing_inf(df: pd.DataFrame, col: str) -> tuple[bool, bool]:
    """Check if there is NaN or infinity in a DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame to check.
        col (str): Name of the column to check.

    Returns:
        tuple[bool, bool]: Has NaN, has infinity.
    """
    # Check if there is missing value or infinity
    all_values = df[col].explode().explode().explode()

    # Convert to numeric, forcing non-numeric types to NaN
    all_values = pd.to_numeric(all_values, errors="coerce")

    # Check for NaN
    has_nan = False
    if all_values.isna().to_numpy().any():
        warnings.warn("NaN found in data.", stacklevel=2)
        has_nan = True

    # Check for infinity
    has_inf = False
    if np.isinf(all_values).to_numpy().any():
        warnings.warn("Infinity found in data.", stacklevel=2)
        has_inf = True

    return has_nan, has_inf


def get_df_nest_level(df: pd.DataFrame, col: str, max_level: int = 5) -> int:
    """Check for maximum nest level in a DataFrame column.

    Definition of nest level:
        Level 0 (no list):       "Fe": 1
        Level 1 (flat list):     "Co": [1, 2]
        Level 2 (nested lists):  "Ni": [[1, 2], [3, 4], ]
        ...

    Args:
        df (pd.DataFrame): The DataFrame to check.
        col (str): Name of the column to check.
        max_level (int): Max nest level to try up to.

    Returns:
        int: The maximum nest level.
    """
    current_level = 0
    while current_level <= max_level:
        # Check if any element in the column is a list
        if df[col].apply(lambda x: isinstance(x, list)).any():
            current_level += 1
            # Explode the lists into separate rows
            df = df.explode(col)
        else:
            return current_level

    return current_level


def replace_missing_and_infinity(
    df: pd.DataFrame,
    col: str,
    missing_strategy: Literal["zero", "mean"] = "mean",
) -> pd.DataFrame:
    """Replace missing value (NaN) and infinity.

    Infinity would be replaced by vmax(∞) or vmin(-∞).
    Missing value would be replaced by selected strategy:
        - zero: replace with zero
        - mean: replace with mean value

    Args:
        df (DataFrame): DataFrame to process.
        col (str): Name of the column to process.
        missing_strategy: missing value replacement strategy.
    """
    # Check for NaN and infinity
    has_nan, has_inf = check_for_missing_inf(df, Key.heat_val)

    # Can only handle nest level 1 at this moment
    if (has_nan or has_inf) and get_df_nest_level(df, col) > 1:
        raise RuntimeError("Unable to process NaN and inf for nest level >1.")

    # Replace missing value with selected strategy
    if has_nan:
        pass

    # Replace infinity
    if has_inf:
        pass

    return df


def preprocess_ptable_data(data: SupportedDataType) -> pd.DataFrame:
    """Preprocess input data for ptable plotters, including:
        - Convert all data types to pd.DataFrame.
        - Impute missing values.
        - Handle anomalies such as NaN, infinity.
        - Write vmin/vmax as metadata into the DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with element names
            as index and values as columns.

    Example:
        >>> data_dict: dict = {
            "H": 1.0,
            "He": [2.0, 4.0],
            "Li": [[6.0, 8.0], [10.0, 12.0]],
        }

        OR
        >>> data_df: pd.DataFrame = pd.DataFrame(
            data_dict.items(),
            columns=["element", "heat_val"]
            ).set_index("element")

        OR
        >>> data_series: pd.Series = pd.Series(data_dict)

        >>> preprocess_data(data_dict / df / series)

        # TODO: the following example looks incorrect, element
        should be the index instead of a column.
        Maybe support both, as it appears pass element as a
        column (without setting it as index) is easier
        What is the default behavior when pd.from_csv()?

             Element   Value
        0    H         [1.0, ]
        1    He        [2.0, 4.0]
        2    Li        [[6.0, 8.0], [10.0, 12.0]]

        Metadata:
            vmin: 1.0
            vmax: 12.0
    """

    def format_pd_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Format pd.DataFrame that may not meet expected format."""

        def fix_df_elem_as_col(df: pd.DataFrame) -> pd.DataFrame:
            """Fix pd.DataFrame where elements are in a single column."""
            # Copy and reset index to move element names to a column
            new_df = df.copy()
            new_df = new_df.reset_index()

            # Find the column with element names
            elem_col_name = None
            for col in new_df.columns:
                if set(new_df[col]).issubset(set(map(str, Element))):
                    elem_col_name = col
                    break

            # Fix failed: cannot find the elements column
            if elem_col_name is None:
                return None

            # Rename the column with elements and set it as index
            new_df = new_df.rename(columns={elem_col_name: Key.element})
            new_df = new_df.set_index(Key.element)

            # Zip the remaining values into a single column
            value_cols = [
                col_name for col_name in new_df.columns if col_name != Key.element
            ]
            new_df[Key.heat_val] = new_df[value_cols].apply(list, axis=1)

            # Drop the old value columns
            return new_df[[Key.heat_val]]

        # Check if input DataFrame is in expected format
        if Key.element == df.index.name and Key.heat_val in df.columns:
            return df

        # Re-format it to expected
        warnings.warn("DataFrame has unexpected format, trying to fix.", stacklevel=2)

        # Try to search for elements as a column
        fixed_df = fix_df_elem_as_col(df)
        if fixed_df is not None:
            return fixed_df

        # Try to search for elements as a row
        fixed_df = fix_df_elem_as_col(df.transpose())
        if fixed_df is not None:
            return fixed_df

        raise RuntimeError("Cannot fix provided DataFrame.")

    # Convert supported data types to DataFrame
    if isinstance(data, pd.DataFrame):
        data_df = format_pd_dataframe(data)

    elif isinstance(data, pd.Series):
        data_df = data.to_frame(name=Key.heat_val)
        data_df.index.name = Key.element

    elif isinstance(data, dict):
        data_df = pd.DataFrame(
            data.items(), columns=[Key.element, Key.heat_val]
        ).set_index(Key.element)

    else:
        type_name = type(data).__name__  # line too long
        raise TypeError(
            f"{type_name} unsupported, choose from {get_args(SupportedDataType)}"
        )

    # Convert all values to 1D np.array
    data_df[Key.heat_val] = [
        # String is Iterable too so would be converted to list of chars
        # but users shouldn't pass strings anyway
        np.array(list(val) if isinstance(val, Iterable) else [val])
        for val in data_df[Key.heat_val]
    ]

    # Handle missing value and infinity
    data_df = replace_missing_and_infinity(data_df, col=Key.heat_val)

    # Flatten up to triple nested lists
    values = data_df[Key.heat_val].explode().explode().explode()
    numeric_values = pd.to_numeric(values, errors="coerce")

    # Write vmin/vmax/mean into df.attrs
    data_df.attrs["vmin"] = numeric_values.min()
    data_df.attrs["mean"] = numeric_values.mean()
    data_df.attrs["vmax"] = numeric_values.max()
    return data_df
