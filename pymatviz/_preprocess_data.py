"""Utils for data preprocessing and related tasks."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Union, get_args, overload

import numpy as np
import pandas as pd
from pymatgen.core import Element

from pymatviz.enums import Key


if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

# Data types that can be passed to PTableProjector and normalized by data_preprocessor
# to SupportedValueType
SupportedDataType = Union[
    dict[str, Union[float, Sequence[float], np.ndarray]], pd.DataFrame, pd.Series
]

# Data types used internally by ptable plotters (returned by preprocess_ptable_data)
SupportedValueType = Union[Sequence[float], np.ndarray]


def check_for_missing_inf(
    df_in: pd.DataFrame, *, col: str
) -> tuple[bool, bool, dict[str, set[Literal["nan", "inf"]]]]:
    """Check if there is NaN or infinity in a DataFrame column.

    Args:
        df_in (pd.DataFrame): DataFrame to check.
        col (str): Name of the column to check.

    Returns:
        tuple[
            bool: Has NaN.
            bool: Has infinity.
            dict: Element name to set["nan", "inf"] mapping.
        ]
    """
    # Convert to numeric, forcing non-numeric types to NaN
    all_values = df_in[col].explode().explode().explode()
    all_values = pd.to_numeric(all_values, errors="coerce")

    # Check for NaN and infinity by row
    has_nan = False
    has_inf = False
    anomalies: dict[str, set[Literal["nan", "inf"]]] = {}

    for idx, value in all_values.items():
        if pd.isna(value):
            if idx not in anomalies:
                anomalies[idx] = set()
            anomalies[idx].add("nan")
            has_nan = True

        elif np.isinf(value):
            if idx not in anomalies:
                anomalies[idx] = set()
            anomalies[idx].add("inf")
            has_inf = True

    if has_nan:
        warnings.warn("NaN found in data", stacklevel=2)
    if has_inf:
        warnings.warn("Infinity found in data", stacklevel=2)

    return has_nan, has_inf, anomalies


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


def replace_missing_and_infinity(
    df_in: pd.DataFrame,
    *,
    col: str,
    missing_strategy: Literal["zero", "mean"] = "mean",
) -> tuple[pd.DataFrame, dict[str, set[Literal["nan", "inf"]]]]:
    """Replace missing value (NaN) and infinity.

    Infinity would be replaced by vmax(∞) or vmin(-∞).
    Missing value would be replaced by selected strategy:
        - zero: replace with zero
        - mean: replace with mean value

    Args:
        df_in (DataFrame): DataFrame to process.
        col (str): Name of the column to process.
        missing_strategy ("zero" | "mean"): Missing value replacement strategy.

    Returns:
        tuple[
            pd.DataFrame: Preprocessed DataFrame.
            dict: Element name to set["nan", "inf"] mapping.
        ]
    """
    # Check for NaN and infinity
    has_nan, has_inf, anomalies = check_for_missing_inf(df_in, col=Key.heat_val)

    if not has_nan and not has_inf:
        return df_in, {}

    nest_level = get_df_nest_level(df_in, col=col)
    if (has_nan or has_inf) and nest_level > 1:
        # Can only handle nest level 1 at this moment
        raise NotImplementedError(
            f"Unable to replace NaN and inf for nest_level>1, got {nest_level}"
        )

    # Get replacement value for missing and infinity
    values = df_in[col].explode()
    numeric_values = pd.to_numeric(values, errors="coerce")

    replacement_nan = 0 if missing_strategy == "zero" else numeric_values.mean()
    replacement_inf_pos = numeric_values[numeric_values != np.inf].max()
    replacement_inf_neg = numeric_values[numeric_values != -np.inf].min()

    replacements = {
        np.nan: replacement_nan,
        np.inf: replacement_inf_pos,
        -np.inf: replacement_inf_neg,
    }

    df_in[col] = df_in[col].apply(
        lambda val: replacements.get(val, val)
        if isinstance(val, float)
        else np.array([replacements.get(v, v) for v in val])
    )

    return df_in, anomalies


def log_scale(
    data: pd.DataFrame,
    *,
    col: str,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """Log scale a pandas DataFrame, which might contain floats or
    sequences of floats.

    Args:
        data (pd.DataFrame): DataFrame to scale.
        col (str): Name of the column to scale.
        eps (float): A small epsilon to avoid log(0).

    Returns:
        pd.DataFrame: Log scaled DataFrame.
    """

    @overload
    def log_transform(val: float, eps: float) -> float:
        pass

    @overload
    def log_transform(val: NDArray, eps: float) -> NDArray:
        pass

    def log_transform(val: float | NDArray, eps: float) -> float | NDArray:
        """Apply logarithm on sequences of floats.

        Args:
            val (float | NDArray): Value(s) to apply log.
            eps (float): A small epsilon to avoid log(0).
        """
        # Sequences of floats (should be NDArray only after preprocessing)
        try:
            return np.log(val)

        # Catch illegal values for log
        except FloatingPointError:
            warnings.warn(f"Illegal log for {val}", stacklevel=2)
            return np.log(np.maximum(val, eps))

    # Apply logarithm to each element in the column
    np.seterr(all="raise")  # raise FloatingPointError instead of warn
    data[col] = data[col].apply(lambda x: log_transform(x, eps))

    return data


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize data by the total sum.

    TODO: might need a more descriptive name.

    Args:
        data (pd.DataFrame): DataFrame to scale.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    # Calculate the total sum
    total_sum = data.map(np.sum).sum().sum()

    # Normalize each value
    return data.map(lambda x: x / total_sum)


# TODO: overload type
def preprocess_ptable_data(
    data: SupportedDataType,
    *,
    missing_strategy: Literal["zero", "mean"] = "mean",
    return_anomalies: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, set[Literal["nan", "inf"]]]]:
    """Preprocess input data for ptable plotters, including:
        - Convert all data types to pd.DataFrame.
        - Replace missing values (NaN) by selected strategy.
        - Replace infinities with vmax(∞) or vmin(-∞).
        - Write vmin/mean/vmax as metadata into the DataFrame.

    Args:
        data (dict[str, float | Sequence[float]] | pd.DataFrame | pd.Series):
            Input data to preprocess.
        missing_strategy: missing value replacement strategy.
        return_anomalies (bool): Whether to return elements with NaN or infinity.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with element names
            as index and values as columns.
        (Optional) dict[str, set["nan", "inf"]]: An element to anomalies mapping.

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

        >>> data_df
            Element   Value
            H         [1.0, ]
            He        [2.0, 4.0]
            Li        [[6.0, 8.0], [10.0, 11.0]]

        Metadata (data_df.attrs):
            vmin: 1.0
            mean: 6.0
            vmax: 11.0
    """

    def format_pd_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Format pd.DataFrame that may not meet expected format."""

        def fix_df_elem_as_col(df: pd.DataFrame) -> pd.DataFrame:
            """Fix pd.DataFrame where elements are in a single column."""
            # Copy and reset index to move element names to a column
            new_df = df.copy().reset_index()

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
            new_df[Key.heat_val] = new_df[value_cols].apply(list, axis="columns")

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

        raise KeyError(
            f"Cannot handle dataframe={df}, needs row/column named "
            f"{Key.element} and {Key.heat_val}"
        )

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
        type_name = type(data).__name__
        raise TypeError(
            f"{type_name} unsupported, choose from {get_args(SupportedDataType)}"
        )

    # Convert all data to NumPy array
    data_df[Key.heat_val] = data_df[Key.heat_val].apply(
        lambda val: np.array(list(val))
        if isinstance(val, Iterable) and not isinstance(val, str)
        else np.array([val])
    )

    # Replace missing value (NaN) and infinity
    data_df, anomalies = replace_missing_and_infinity(
        data_df, col=Key.heat_val, missing_strategy=missing_strategy
    )

    # Parse meta data
    numeric_values = pd.to_numeric(
        data_df[Key.heat_val].explode().explode().explode(), errors="coerce"
    )
    data_df.attrs["vmin"] = numeric_values.min()
    data_df.attrs["mean"] = numeric_values.mean()
    data_df.attrs["vmax"] = numeric_values.max()

    return (data_df, anomalies) if return_anomalies else data_df
