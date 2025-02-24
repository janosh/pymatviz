"""pymatviz utility functions."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition, Structure

from pymatviz.enums import ElemCountMode, Key
from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatviz.typing import ElemValues, FormulaGroupBy


def count_elements(
    values: ElemValues,
    count_mode: ElemCountMode = ElemCountMode.composition,
    exclude_elements: Sequence[str] = (),
    fill_value: float | None = None,
) -> pd.Series:
    """Count element occurrence in list of formula strings or dict-like compositions.

    If passed values are already a map from element symbol to counts, ensure the
    data is a pd.Series filled with "fill_value" for missing element.

    Provided as standalone function for external use or to cache long computations.
    Caching long element counts is done by refactoring:
        ptable_heatmap_plotly(long_list_of_formulas) # slow
    to:
        elem_counts = count_elements(long_list_of_formulas) # slow
        ptable_heatmap_plotly(elem_counts) # fast, only rerun this line to update plot

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
        exclude_elements (Sequence[str]): Elements to exclude from the count.
            Defaults to ().
        fill_value (float | None): Value to fill in for missing elements.
            Defaults to None for NaN.

    Returns:
        pd.Series: Map element symbols to heatmap values.
    """
    valid_count_modes = set(ElemCountMode)
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
            df_ptable.reset_index().set_index(Key.atomic_number).symbol
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


def count_formulas(
    data: Sequence[str | Composition | Structure],
    *,
    group_by: FormulaGroupBy = "chem_sys",
) -> pd.DataFrame:
    """Process chemical system data into a standardized DataFrame format.

    Used e.g. by chem_sys_sunburst and chem_sys_treemap to preprocess input data.

    Args:
        data (Sequence[str | Composition | Structure]): Chemical systems. Can be:
            - Chemical system strings like ["Fe-O", "Li-P-O"]
            - Formula strings like ["Fe2O3", "LiPO4"]
            - Pymatgen Compositions
            - Pymatgen Structures
        group_by ("formula" | "reduced_formula" | "chem_sys"): How to group formulas:
            - "formula": Each unique formula is counted separately.
            - "reduced_formula": Formulas are reduced to simplest ratios.
            - "chem_sys": All formulas with same elements are grouped.

    Returns:
        pd.DataFrame: DataFrame with columns for arity, chemical system, and optionally
        formula, sorted by arity and chemical system for consistent ordering.

    Raises:
        ValueError: If data is empty or contains invalid formulas/elements.
        TypeError: If data contains unsupported types.
    """
    if len(data) == 0:
        raise ValueError("Empty input: data sequence is empty")

    # Map from number of elements to arity name
    arity_names: Final[dict[int, str]] = {
        1: "unary",
        2: "binary",
        3: "ternary",
        4: "quaternary",
        5: "quinary",
        6: "senary",
        7: "septenary",
        8: "octonary",
        9: "nonary",
        10: "denary",
    }

    # Convert all inputs to chemical systems (sorted tuples of element strings)
    systems: list[tuple[str, ...]] = []
    formulas: list[str | None] = []  # store formulas if not grouping by chem_sys
    for item in data:
        if isinstance(item, Structure):
            elems = sorted(item.composition.chemical_system.split("-"))
            if group_by == "formula":
                formula = str(item.composition)
            elif group_by == "reduced_formula":
                formula = item.composition.reduced_formula
            else:  # chem_sys
                formula = None
        elif isinstance(item, Composition):
            elems = sorted(item.chemical_system.split("-"))
            if group_by == "formula":
                formula = str(item)
            elif group_by == "reduced_formula":
                formula = item.reduced_formula
            else:  # chem_sys
                formula = None
        elif isinstance(item, str):
            if "-" in item:  # already a chemical system string
                elems = sorted(item.split("-"))
                formula = item if group_by != Key.chem_sys else None
            else:  # assume it's a formula string
                try:
                    comp = Composition(item)
                    elems = sorted(comp.chemical_system.split("-"))
                    if group_by == "formula":
                        formula = item  # preserve original formula string
                    elif group_by == "reduced_formula":
                        formula = comp.reduced_formula
                    else:  # chem_sys
                        formula = None
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid formula: {item}") from e
        else:
            raise TypeError(
                f"Expected str, Composition or Structure, got {type(item)} instead"
            )

        # Remove duplicates and sort elements
        elems = sorted(set(elems))
        if not all(elem.isalpha() for elem in elems):
            raise ValueError(f"Invalid elements in system: {item}")
        systems += [tuple(elems)]
        if group_by != Key.chem_sys:
            formulas += [formula]

    # Create a DataFrame with arity and chemical system columns
    df_systems = pd.DataFrame({"system": systems})
    if group_by != Key.chem_sys:
        df_systems[Key.formula] = formulas

    df_systems[Key.arity] = df_systems["system"].map(len)
    df_systems["arity_name"] = df_systems[Key.arity].map(
        lambda n_elems: arity_names.get(n_elems, f"{n_elems}-component")
    )
    df_systems[Key.chem_sys] = df_systems["system"].str.join("-")

    # Count occurrences of each system
    group_cols = ["arity_name", Key.chem_sys]
    if group_by != Key.chem_sys:
        group_cols += [Key.formula]

    df_counts = df_systems.value_counts(group_cols).reset_index()
    df_counts.columns = [*group_cols, Key.count]  # preserve original column names

    # Sort by arity and chemical system for consistent ordering
    return df_counts.sort_values(["arity_name", Key.chem_sys])
