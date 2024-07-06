"""pymatviz utility functions."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition

from pymatviz.enums import ElemCountMode, Key
from pymatviz.utils import ElemValues, df_ptable


if TYPE_CHECKING:
    from collections.abc import Sequence


def count_elements(
    values: ElemValues,
    count_mode: ElemCountMode = ElemCountMode.composition,
    exclude_elements: Sequence[str] = (),
    fill_value: float | None = 0,
) -> pd.Series:
    """Count element occurrence in list of formula strings or dict-like compositions.
    If passed values are already a map from element symbol to counts, ensure the
    data is a pd.Series filled with zero values for missing element symbols.

    Provided as standalone function for external use or to cache long computations.
    Caching long element counts is done by refactoring
        ptable_heatmap(long_list_of_formulas) # slow
    to
        elem_counts = count_elements(long_list_of_formulas) # slow
        ptable_heatmap(elem_counts) # fast, only rerun this line to update the plot

    Args:
        values (dict[str, int | float] | pd.Series | list[str]): Iterable of
            composition strings/objects or map from element symbols to heatmap values.
        count_mode ('(element|fractional|reduced)_composition'):
            Only used when values is a list of composition strings/objects.
            - composition (default): Count elements in each composition as is,
                i.e. without reduction or normalization.
            - fractional_composition: Convert to normalized compositions in which the
                amounts of each species sum to before counting.
                Example: Fe2 O3 -> Fe0.4 O0.6
            - reduced_composition: Convert to reduced compositions (i.e. amounts
                normalized by greatest common denominator) before counting.
                Example: Fe4 P4 O16 -> Fe P O4.
            - occurrence: Count the number of times each element occurs in a list of
                formulas irrespective of compositions. E.g. [Fe2 O3, Fe O, Fe4 P4 O16]
                counts to {Fe: 3, O: 3, P: 1}.
        exclude_elements (Sequence[str]): Elements to exclude from the count. Defaults
            to ().
        fill_value (float | None): Value to fill in for missing elements. Defaults to 0.

    Returns:
        pd.Series: Map element symbols to heatmap values.
    """
    valid_count_modes = list(ElemCountMode.key_val_dict())
    if count_mode not in valid_count_modes:
        raise ValueError(f"Invalid {count_mode=} must be one of {valid_count_modes}")
    # Ensure values is Series if we got dict/list/tuple
    srs = pd.Series(values)

    if is_numeric_dtype(srs):
        pass
    elif is_string_dtype(srs) or {*map(type, srs)} <= {str, Composition}:
        # all items are formula strings or Composition objects
        if count_mode == "occurrence":
            srs = pd.Series(
                itertools.chain.from_iterable(
                    map(str, Composition(comp, allow_negative=True)) for comp in srs
                )
            ).value_counts()
        else:
            attr = (
                "element_composition" if count_mode == Key.composition else count_mode
            )
            srs = pd.DataFrame(
                getattr(Composition(formula, allow_negative=True), attr).as_dict()
                for formula in srs
            ).sum()  # sum up element occurrences
    else:
        raise ValueError(
            "Expected values to be map from element symbols to heatmap values or "
            f"list of compositions (strings or Pymatgen objects), got {values}"
        )

    try:
        # If index consists entirely of strings representing integers, convert to ints
        srs.index = srs.index.astype(int)
    except (ValueError, TypeError):
        pass

    if pd.api.types.is_integer_dtype(srs.index):
        # If index is all integers, assume they represent atomic
        # numbers and map them to element symbols (H: 1, He: 2, ...)
        idx_min, idx_max = srs.index.min(), srs.index.max()
        if idx_max > 118 or idx_min < 1:
            raise ValueError(
                "element value keys were found to be integers and assumed to represent "
                f"atomic numbers, but values range from {idx_min} to {idx_max}, "
                "expected range [1, 118]."
            )
        map_atomic_num_to_elem_symbol = (
            df_ptable.reset_index().set_index("atomic_number").symbol
        )
        srs.index = srs.index.map(map_atomic_num_to_elem_symbol)

    # Ensure all elements are present in returned Series (with value zero if they
    # weren't in values before)
    srs = srs.reindex(df_ptable.index, fill_value=fill_value).rename("count")

    if len(exclude_elements) > 0:
        if isinstance(exclude_elements, str):
            exclude_elements = [exclude_elements]
        if isinstance(exclude_elements, tuple):
            exclude_elements = list(exclude_elements)
        try:
            srs = srs.drop(exclude_elements)
        except KeyError as exc:
            bad_symbols = ", ".join(x for x in exclude_elements if x not in srs)
            raise ValueError(
                f"Unexpected symbol(s) {bad_symbols} in {exclude_elements=}"
            ) from exc

    return srs
