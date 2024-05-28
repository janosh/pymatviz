"""Utils for data preprocessing and related tasks."""

import warnings
from collections.abc import Iterable, Sequence
from typing import Union, get_args

import numpy as np
import pandas as pd
from pymatgen.core import Element

from pymatviz.enums import Key


# Data types used internally by ptable plotters
SupportedValueType = Union[Sequence[float], np.ndarray]

# Data types that can be passed to PTableProjector and data_preprocessor
SupportedDataType = Union[
    dict[str, Union[float, Sequence[float], np.ndarray]], pd.DataFrame, pd.Series
]


def check_for_missing_inf(df: pd.DataFrame) -> tuple[bool, bool]:
    """Check if there is NaN or infinity in pandas DataFrame.

    Returns:
        tuple[bool, bool]: Has NaN, has infinity.
    """
    # Check if there is missing value or infinity
    all_values = df[Key.heat_val].explode().explode().explode()

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
        """Fix pd.DataFrame that does not meet expected format."""

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
        warnings.warn(
            "pd.DataFrame has unexpected format, trying to fix.", stacklevel=2
        )

        # Try to search for elements as a column
        fixed_df = fix_df_elem_as_col(df)
        if fixed_df is not None:
            return fixed_df

        # Try to search for elements as a row
        fixed_df = fix_df_elem_as_col(df.transpose())
        if fixed_df is not None:
            return fixed_df

        raise RuntimeError("Cannot fix provided DataFrame.")

    def handle_missing_and_infinity(
        df: pd.DataFrame,
        # missing_strategy: Literal["zero", "mean"] = "mean",
    ) -> pd.DataFrame:
        """Handle missing value (NaN) and infinity.

        Infinity would be replaced by vmax(∞) or vmin(-∞).
        Missing value would be replaced by selected strategy:
            - zero: replace with zero
            - mean: replace with mean value
        """
        # Check for NaN and infinity
        has_nan, has_inf = check_for_missing_inf(df)

        return df

    # Check and handle different supported data types
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

    # Handle missing and anomalous values
    data_df = handle_missing_and_infinity(data_df)

    # Flatten up to triple nested lists
    values = data_df[Key.heat_val].explode().explode().explode()
    numeric_values = pd.to_numeric(values, errors="coerce")

    # Write vmin/vmax into df.attrs for colorbar
    data_df.attrs["vmin"] = numeric_values.min()  # ignores NaNs
    data_df.attrs["vmax"] = numeric_values.max()
    return data_df
