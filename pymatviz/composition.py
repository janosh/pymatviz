"""Count elements and formulas in compositions, formulas and structures."""

from __future__ import annotations

import functools
import itertools
import math
from collections import defaultdict
from collections.abc import Mapping
from numbers import Number
from typing import TYPE_CHECKING, Any

import matscikit as mk
import numpy as np
import pandas as pd
from matscikit.interop import has_class
from pandas.api.types import is_numeric_dtype, is_string_dtype

from pymatviz.enums import ElemCountMode
from pymatviz.structures import is_structure_like, to_structure
from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from collections.abc import Sequence

    from matscikit.interop import CompositionLike
    from typing_extensions import TypeIs

    from pymatviz.typing import (
        ElemCountModeLike,
        ElemValues,
        FormulaGroupBy,
        FormulaLike,
    )


def is_composition_like(obj: Any) -> TypeIs[CompositionLike]:
    """Whether obj is a formula string, a matscikit or pymatgen Composition."""
    return isinstance(obj, str | mk.Composition) or has_class(
        obj, "pymatgen.core.composition.Composition"
    )


def to_composition(obj: FormulaLike, *, allow_negative: bool = False) -> mk.Composition:
    """Composition (matscikit) of a formula string, {element: amount} mapping,
    matscikit or pymatgen Composition (oxidation states summed per element) or
    structure-like object.

    Raises:
        TypeError: For any other type.
    """
    if isinstance(obj, mk.Composition):
        return obj
    if is_structure_like(obj):
        return to_structure(obj).composition
    if not isinstance(obj, str | Mapping):
        raise TypeError(
            "Expected a formula string, mapping, Composition or structure, got "
            f"{type(obj).__name__}"
        )
    return mk.Composition(obj, allow_negative=allow_negative)


def count_elements(
    values: ElemValues,
    count_mode: ElemCountModeLike = ElemCountMode.composition,
    exclude_elements: str | Sequence[str] = (),
    fill_value: float | None = None,
) -> pd.Series:
    """Count element occurrence in list of formula strings or dict-like compositions.

    If passed values are already a map from element symbol to counts, ensure the
    data is a pd.Series filled with "fill_value" for missing element.

    Precompute counts once to replot long formula lists quickly.

    Args:
        values (dict[str, int | float] | pd.Series | list[str]): Iterable of
            composition strings/objects or map from element symbols to heatmap values.
        count_mode ('(element|fractional|reduced)_composition'):
            Only used when values is a list of composition strings/objects.
            - composition (default): Count elements in each composition as is,
                i.e. without reduction or normalization.
            - fractional_composition: Convert to normalized compositions in which the
                amounts of each species sum to one before counting.
                Example: Fe2 O3 -> Fe0.4 O0.6
            - reduced_composition: Convert to reduced compositions (i.e. amounts
                normalized by greatest common denominator) before counting.
                Example: Fe4 P4 O16 -> Fe P O4.
            - occurrence: Count the number of times each element occurs in a list of
                formulas irrespective of compositions. E.g. [Fe2 O3, Fe O, Fe4 P4 O16]
                counts to {Fe: 3, O: 3, P: 1}.
        exclude_elements (str | Sequence[str]): Element(s) to exclude from the count.
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
        pass  # already a map from element to value
    elif is_string_dtype(srs) or all(map(is_composition_like, srs)):
        comps = [to_composition(comp, allow_negative=True) for comp in srs]
        if count_mode == "occurrence":
            srs = pd.Series(
                itertools.chain.from_iterable(comp.elements for comp in comps)
            ).value_counts()
        else:
            # charged species (Fe3+) count as their elements in every mode
            attr = None if count_mode == "composition" else count_mode
            # collect each element's amounts and sum them with exactly rounded fsum
            # (faster than summing a wide DataFrame of mostly-NaN dict rows)
            elem_amounts: defaultdict[str, list[float]] = defaultdict(list)
            for comp in comps:
                comp = getattr(comp, attr) if attr else comp  # noqa: PLW2901
                for elem, amount in comp.as_dict().items():
                    elem_amounts[elem].append(amount)
            srs = pd.Series(
                {elem: math.fsum(amounts) for elem, amounts in elem_amounts.items()},
                dtype=float,
            )
    else:
        raise ValueError(
            "Expected values to be map from element symbols to heatmap values or "
            f"list of compositions (strings or Composition objects), got {values}"
        )

    try:  # integer keys (or their strings) are atomic numbers
        atomic_numbers = srs.index.astype(int)
    except (ValueError, TypeError):
        pass
    else:
        if any(
            isinstance(key, Number) and key != number
            for key, number in zip(srs.index, atomic_numbers, strict=True)
        ):
            raise ValueError(
                f"Atomic numbers must be integers, got: {srs.index.tolist()}"
            )
        numbers = np.asarray(atomic_numbers, dtype=int)
        if len(numbers) and (numbers.min() < 1 or numbers.max() > 118):
            raise ValueError(
                "element value keys were found to be integers and assumed to represent "
                f"atomic numbers, but values range from {numbers.min()} to "
                f"{numbers.max()}, expected range [1, 118]."
            )
        srs.index = atomic_numbers.map(
            df_ptable.reset_index().set_index("atomic_number").symbol
        )

    bad_symbols = srs.index.difference(df_ptable.index)
    if len(bad_symbols) > 0:
        raise ValueError(
            f"Unexpected element symbol(s): {', '.join(map(str, bad_symbols))}"
        )

    srs = srs.reindex(df_ptable.index, fill_value=fill_value).rename("count")
    excluded_elements = (
        [exclude_elements] if isinstance(exclude_elements, str) else [*exclude_elements]
    )
    if bad := [elem for elem in excluded_elements if elem not in srs]:
        raise ValueError(
            f"Unexpected symbol(s) {', '.join(bad)} in {excluded_elements=}"
        )
    return srs.drop(excluded_elements)


def _system_and_formula(
    item: FormulaLike, group_by: FormulaGroupBy
) -> tuple[tuple[str, ...], str]:
    """Sorted element tuple and formula label of one count_formulas input item."""
    if isinstance(item, str) and "-" in item:  # already a chemical system string
        elems, formula = item.split("-"), item
    else:
        try:
            comp = to_composition(item)
        except ValueError as exc:
            raise ValueError(f"Invalid formula: {item}") from exc
        elems = comp.chemical_system.split("-")
        if group_by == "reduced_formula":
            formula = comp.reduced_formula
        else:  # preserve original formula strings
            formula = item if isinstance(item, str) else str(comp)

    elems = sorted(set(elems))
    if not all(elem in df_ptable.index for elem in elems):
        raise ValueError(f"Invalid elements in system: {item}")
    return tuple(elems), formula


def count_formulas(
    data: Sequence[FormulaLike],
    *,
    group_by: FormulaGroupBy = "chem_sys",
) -> pd.DataFrame:
    """Process chemical system data into a standardized DataFrame format.

    Used e.g. by chem_sys_hierarchy to preprocess input data.

    Args:
        data (Sequence[FormulaLike]): Chemical systems. Can be:
            - Chemical system strings like ["Fe-O", "Li-P-O"]
            - Formula strings like ["Fe2O3", "LiPO4"]
            - matscikit or pymatgen Compositions
            - structure-like objects (matscikit, pymatgen, ASE, phonopy)
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
    valid_group_by = ("formula", "reduced_formula", "chem_sys")
    if group_by not in valid_group_by:
        raise ValueError(f"Invalid {group_by=}, must be one of {valid_group_by}")

    if len(data) == 0:
        raise ValueError("Empty input: data sequence is empty")

    arity_names = ("unary", "binary", "ternary", "quaternary", "quinary", "senary")
    arity_names += ("septenary", "octonary", "nonary", "denary")

    # repeated strings, the common case for large datasets, are parsed once
    parse_str = functools.cache(lambda item: _system_and_formula(item, group_by))
    df_systems = pd.DataFrame(
        [
            parse_str(item)
            if isinstance(item, str)
            else _system_and_formula(item, group_by)
            for item in data
        ],
        columns=pd.Index(["system", "formula"]),
    )
    df_systems["arity"] = df_systems["system"].map(len)
    df_systems["arity_name"] = df_systems["arity"].map(
        lambda n_elems: (
            arity_names[n_elems - 1]
            if n_elems <= len(arity_names)
            else f"{n_elems}-component"
        )
    )
    df_systems["chem_sys"] = df_systems["system"].str.join("-")

    formula_cols = ["formula"] if group_by != "chem_sys" else []
    group_cols = ["arity", "arity_name", "chem_sys", *formula_cols]
    df_counts = df_systems.value_counts(group_cols).reset_index()
    df_counts.columns = [*group_cols, "count"]
    return (
        df_counts.sort_values(["arity", "chem_sys", *formula_cols])
        .drop(columns="arity")
        .reset_index(drop=True)
    )
