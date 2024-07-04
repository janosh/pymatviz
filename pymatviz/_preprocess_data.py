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
    from collections.abc import Sequence
    from typing import Callable, Literal, Self

    from numpy.typing import NDArray

# Data types that can be passed to PTableProjector and normalized by data_preprocessor
# to SupportedValueType
SupportedDataType = Union[
    dict[str, Union[float, Sequence[float], np.ndarray]], pd.DataFrame, pd.Series
]

# Data types used internally by ptable plotters (returned by preprocess_ptable_data)
SupportedValueType = Union[Sequence[float], np.ndarray]


class PTableData:
    """Store input data for ptable plotters.

    Attributes:
        data (pd.DataFrame): The preprocessed DataFrame with element names
            as index and values as columns.
        index_col (str): The index column header.
        val_col (str): The value column header.
        anomalies (dict[str, set["nan", "inf"]]): An element to anomalies mapping.
        nest_level (int): The nest level of DataFrame.

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

    def __init__(
        self,
        data: SupportedDataType,
        *,
        index_col: str = Key.element,
        val_col: str = Key.heat_val,
        check_missing: bool = True,
        missing_strategy: Literal["zero", "mean"] = "mean",
        check_infinity: bool = True,
        normalize: bool = False,
    ) -> None:
        """Preprocess data, including:
            - Convert all data types to pd.DataFrame.
            - Replace missing values (NaN) by selected strategy.
            - Replace infinities with vmax(∞) or vmin(-∞).
            - Write vmin/mean/vmax as metadata into the DataFrame.

        Args:
            data (dict[str, float | Sequence[float]] | pd.DataFrame | pd.Series):
                Input data to preprocess.
            index_col (str, optional): The index column header. Defaults to Key.index.
            val_col (str, optional): The value column header. Defaults to Key.heat_val.
            check_missing (bool): Whether to check and replace missing values.
            missing_strategy ("zero", "mean"): Missing value replacement strategy.
                - zero: Replace with zero.
                - mean: Replace with mean value.
            check_infinity (bool): Whether to check and replace infinities.
            normalize (bool): Whether to normalize data.
        """
        # Convert and set data, and write metadata
        self.index_col = index_col
        self.val_col = val_col
        self.data = data

        # Get nest level
        self.nest_level = get_df_nest_level(self.data, col=self.val_col)

        # Replace missing values and infinities, and update anomalies
        self.anomalies = {} if (check_missing or check_infinity) else "NA"
        if check_missing:
            self.check_and_replace_missing(missing_strategy=missing_strategy)
        if check_infinity:
            self.check_and_replace_infinity()

        # Normalize data by the total sum
        if normalize:
            self.normalize()

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data: SupportedDataType) -> None:
        """Preprocess and set data, including:
        - Convert supported data types to DataFrame.
        - Convert all values to NumPy array.
        - Write vmin/mean/vmax as metadata into the DataFrame.
        """
        # Convert supported data types to DataFrame
        if isinstance(data, pd.DataFrame):
            data = self._format_pd_dataframe(data)

        elif isinstance(data, pd.Series):
            data = data.to_frame(name=self.val_col)
            data.index.name = self.index_col

        elif isinstance(data, dict):
            data = pd.DataFrame(
                data.items(), columns=[self.index_col, self.val_col]
            ).set_index(self.index_col)

        else:
            type_name = type(data).__name__
            raise TypeError(
                f"{type_name} unsupported, choose from {get_args(SupportedDataType)}"
            )

        # Convert all values to NumPy array  # TODO: reuse apply
        data[self.val_col] = data[self.val_col].apply(
            lambda val: np.array(list(val))
            if isinstance(val, Iterable) and not isinstance(val, str)
            else np.array([val])
        )

        self._data = data

        # Write vmin/mean/vmax as metadata into the DataFrame
        # Note: this is nested inside the setter so that metadata
        # would be updated whenever data gets updated
        self._write_meta_data()

    def _write_meta_data(self) -> None:
        """Parse meta data and write into data.attrs.

        Currently would handle the following:
            vmin: The min value.
            mean: The mean value.
            vmax: The max value.
        """
        numeric_values = pd.to_numeric(
            self._data[self.val_col].explode().explode().explode(), errors="coerce"
        )
        self._data.attrs["vmin"] = numeric_values.min()
        self._data.attrs["mean"] = numeric_values.mean()
        self._data.attrs["vmax"] = numeric_values.max()

    @staticmethod
    def _format_pd_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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

    @classmethod
    def from_formulas(cls, formulas: Sequence[str] | dict[str, float]) -> Self:
        """Initialize PTableData from sequences of chemical formulas
        or dict.

        # TODO: finish docstring and method.

        # TODO: merge count_elements function

        """
        # Convert sequences of chemical formulas to XXX TODO:
        if isinstance(formulas, dict):
            pass
        else:
            pass

    def normalize(self) -> None:
        """Normalize data by the total sum."""
        total_sum = self._data.map(np.sum).sum().sum()
        self.data = self._data.map(lambda x: x / total_sum)

    def apply(self, func: Callable) -> None:
        """Apply a function to all values in value column.

        Args:
            func (Callable): The function to be applied.
        """
        original_data = self.data
        original_data[self.val_col] = original_data[self.val_col].apply(func)

        # Ensure metadata is updated by using the setter method
        # TODO: double check if this is necessary (update unit test as well)
        self.data = original_data

    def log_scale(
        self,
        *,
        eps: float = 1e-10,
    ) -> None:
        """Log scale the data.

        Args:
            eps (float): A small epsilon to avoid log(0).
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
                eps (float): A small epsilon to avoid zero or negative values.
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
        self.apply(lambda x: log_transform(x, eps))

    def drop_elements(self, elements: Sequence[str]) -> None:
        """Drop selected elements from data.

        Args:
            elements (Sequence[str]): Elements to drop.
        """
        original_data = self.data
        df_dropped = original_data.drop(elements, axis=0)

        self.data = df_dropped

    def check_and_replace_missing(
        self,
        missing_strategy: Literal["zero", "mean"],
    ) -> bool:
        """Check if there is missing value (NaN) in the values column,
        and replace them according to selected strategy.

        Would update the anomalies attribute.

        Args:
            missing_strategy ("zero", "mean"): Missing value replacement strategy.
                - zero: Replace with zero.
                - mean: Replace with mean value.

        Returns:
            bool: Whether the column contains NaN.
        """
        # Convert to numeric, forcing non-numeric types to NaN
        all_values = self.data[self.val_col].explode().explode().explode()
        all_values = pd.to_numeric(all_values, errors="coerce")

        has_nan = False
        for elem, value in all_values.items():
            if pd.isna(value):
                if elem not in self.anomalies:
                    self.anomalies[elem] = set()
                self.anomalies[elem].add("nan")
                has_nan = True

        if has_nan:
            if self.nest_level > 1:
                raise NotImplementedError(
                    "Unable to replace NaN and inf for nest_level>1"
                )

            warnings.warn("NaN found in data", stacklevel=2)

            # Generate and apply replacement
            @overload
            def replace_nan(val: NDArray) -> NDArray:
                pass

            @overload
            def replace_nan(val: float) -> float:
                pass

            def replace_nan(val: NDArray | float) -> NDArray | float:
                """Replace missing value based on selected strategy.

                Args:
                    val (NDarray | float): Value to be processed.
                """
                replace_val = 0 if missing_strategy == "zero" else all_values.mean()

                if isinstance(val, np.ndarray):
                    return np.array([replace_val if np.isnan(v) else v for v in val])

                return replace_val if np.isnan(val) else val

            self.apply(replace_nan)

        return has_nan

    def check_and_replace_infinity(self) -> bool:
        """Check if there is infinity in the data column, and
        replace them with vmax(∞) or vmin(-∞) if any.

        Would update the anomalies attribute.

        Returns:
            bool: Whether the column contains NaN.
        """
        # Convert to numeric, forcing non-numeric types to NaN
        all_values = self.data[self.val_col].explode().explode().explode()
        all_values = pd.to_numeric(all_values, errors="coerce")

        has_inf = False

        for elem, value in all_values.items():
            if np.isinf(value):
                if elem not in self.anomalies:
                    self.anomalies[elem] = set()
                self.anomalies[elem].add("inf")
                has_inf = True

        if has_inf:
            if self.nest_level > 1:
                raise NotImplementedError(
                    "Unable to replace NaN and inf for nest_level>1."
                )

            warnings.warn("Infinity found in data", stacklevel=2)

            # Generate and apply replacement
            @overload
            def replace_inf(val: NDArray) -> NDArray:
                pass

            @overload
            def replace_inf(val: float) -> float:
                pass

            def replace_inf(val: NDArray | float) -> NDArray | float:
                """Replace infinities."""
                replacements = {
                    np.inf: all_values[all_values != np.inf].max(),
                    -np.inf: all_values[all_values != -np.inf].min(),
                }

                if isinstance(val, np.ndarray):
                    return np.array(
                        [replacements.get(v, v) if np.isinf(v) else v for v in val]
                    )

                return replacements[val] if np.isinf(val) else val

            self.apply(replace_inf)

        return has_inf


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
