"""pymatviz utility functions."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
import scipy.stats
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition, SiteCollection, Structure
from pymatgen.io.phonopy import get_pmg_structure

from pymatviz.enums import ElemCountMode, Key
from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from pymatviz.typing import AnyStructure, ElemValues, FormulaGroupBy, T


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
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"Invalid formula: {item}") from exc
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


STRUCTURE_CLASSES = [
    ("ase", ["Atoms"]),
    ("pymatgen.core", ["Structure", "IStructure", "Molecule", "IMolecule"]),
    ("phonopy.structure.atoms", ["PhonopyAtoms"]),
]


def is_structure_like(obj: Any) -> bool:
    """Check if object is structure-like."""
    for module_path, class_names in STRUCTURE_CLASSES:
        try:
            module = __import__(module_path, fromlist=class_names)
            if any(
                isinstance(obj, getattr(module, cls))
                for cls in class_names
                if hasattr(module, cls)
            ):
                return True
        except ImportError:
            pass
    return False


def is_ase_atoms(struct: Any) -> bool:
    """Check if the input is an ASE Atoms object without importing ase."""
    cls_name = f"{type(struct).__module__}.{type(struct).__qualname__}"
    return cls_name in ("ase.atoms.Atoms", "pymatgen.io.ase.MSONAtoms")


def is_phonopy_atoms(obj: Any) -> bool:
    """Check if object is PhonopyAtoms."""
    cls_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
    return cls_name == "phonopy.structure.atoms.PhonopyAtoms"


def is_trajectory_like(obj: Any) -> bool:
    """Check if object is trajectory-like."""
    return (
        isinstance(obj, (list, tuple))
        and len(obj) > 0
        and all(is_structure_like(item) or isinstance(item, dict) for item in obj)
    )


def is_composition_like(obj: Any) -> bool:
    """Check if object is composition-like."""
    try:
        from pymatgen.core import Composition

        return isinstance(obj, Composition)
    except ImportError:
        return False


def normalize_structures(
    systems: AnyStructure
    | Sequence[AnyStructure]
    | pd.Series
    | dict[str, AnyStructure],
) -> dict[str, Structure]:
    """Convert pymatgen Structures, ASE Atoms, or PhonopyAtoms or sequences/dicts of
    them to a dictionary of pymatgen Structures.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    def to_pmg_struct(item: Any) -> SiteCollection:
        if is_ase_atoms(item):
            return AseAtomsAdaptor().get_structure(item)
        if isinstance(item, SiteCollection):
            return item
        if is_phonopy_atoms(item):  # convert PhonopyAtoms to pymatgen Structure
            return get_pmg_structure(item)
        raise TypeError(
            f"Item must be a Pymatgen Structure, ASE Atoms, or PhonopyAtoms object, "
            f"got {type(item)}"
        )

    if is_ase_atoms(systems) or is_phonopy_atoms(systems):
        # Handles single ASE Atoms or PhonopyAtoms object
        systems = to_pmg_struct(systems)

    # Check for single Structure/IStructure first, before checking for Sequence
    # since they are Sequences but we don't want to iterate over sites
    if isinstance(systems, SiteCollection):
        # Use formula as key for single structure input
        return {systems.formula: systems}

    if hasattr(systems, "__len__") and len(systems) == 0:
        raise ValueError("Cannot plot empty set of structures")

    if isinstance(systems, dict):  # Process dict values, keep original keys
        return {key: to_pmg_struct(val) for key, val in systems.items()}

    if isinstance(systems, pd.Series):  # Keep original Series index as keys
        return {key: to_pmg_struct(val) for key, val in systems.items()}

    if isinstance(systems, (Sequence, pd.Series)) and not isinstance(systems, str):
        iterable_struct = list(systems) if isinstance(systems, pd.Series) else systems
        return {
            f"{idx} {(systems := to_pmg_struct(item)).formula}": systems
            for idx, item in enumerate(iterable_struct, start=1)
        }

    raise TypeError(
        f"Input must be a Pymatgen Structure, ASE Atoms, or PhonopyAtoms object, a "
        f"sequence (list, tuple, pd.Series), or a dict. Got {type(systems)=}"
    )


def normalize_to_dict(
    inputs: T | Sequence[T] | dict[str, T],
    cls: type[T] = Structure,
    key_gen: Callable[[T], str] = lambda obj: getattr(
        obj, "formula", type(obj).__name__
    ),
) -> dict[str, T]:
    """Normalize any kind of object or dict/list/tuple of them into to a dictionary.

    Args:
        inputs (T | Sequence[T] | dict[str, T]): A single object, a sequence of objects,
            or a dictionary of objects.
        cls (type[T], optional): The class of the objects to normalize. Defaults to
            pymatgen.core.Structure.
        key_gen (Callable[[T], str], optional): A function that generates a key for
            each object. Defaults to using the object's formula, assuming the objects
            are pymatgen.core.(Structure|Molecule).

    Returns:
        dict[str, T]: Map of objects with keys as object formulas or given keys.

    Raises:
        TypeError: If the input format is invalid.
    """
    if isinstance(inputs, cls):
        return {"": inputs}

    if (
        isinstance(inputs, list | tuple)
        and all(isinstance(obj, cls) for obj in inputs)
        and len(inputs) > 0
    ):
        out_dict: dict[str, T] = {}
        for obj in inputs:
            key = key_gen(obj)
            idx = 1
            while key in out_dict:
                key += f" {idx}"
                idx += 1
            out_dict[key] = obj
        return out_dict
    if isinstance(inputs, dict):
        return inputs
    if isinstance(inputs, pd.Series):
        return inputs.to_dict()

    cls_name = cls.__name__
    raise TypeError(
        f"Invalid {inputs=}, expected {cls_name} or dict/list/tuple of {cls_name}"
    )


def df_to_arrays(
    df: pd.DataFrame | None,
    *args: str | ArrayLike,
    strict: bool = True,
) -> list[ArrayLike | dict[str, ArrayLike]]:
    """If df is None, this is a no-op: args are returned as-is. If df is a
    dataframe, all following args are used as column names and the column data
    returned as arrays (after dropping rows with NaNs in any column).

    Args:
        df (pd.DataFrame | None): Optional pandas DataFrame.
        *args (list[ArrayLike | str]): Arbitrary number of arrays or column names in df.
        strict (bool, optional): If True, raise TypeError if df is not pd.DataFrame
            or None. If False, return args as-is. Defaults to True.

    Raises:
        ValueError: If df is not None and any of the args is not a df column name.
        TypeError: If df is not pd.DataFrame and not None.

    Returns:
        list[ArrayLike | dict[str, ArrayLike]]: Array data for each column name or
            dictionary of column names and array data.
    """
    if df is None:
        if cols := [arg for arg in args if isinstance(arg, str)]:
            raise ValueError(f"got column names but no df to get data from: {cols}")
        # Convert args to numpy arrays if they are not already
        return list(map(np.asarray, args))

    if not isinstance(df, pd.DataFrame):
        if not strict:
            return args  # type: ignore[return-value]
        raise TypeError(f"df should be pandas DataFrame or None, got {type(df)}")

    if arrays := [arg for arg in args if isinstance(arg, np.ndarray)]:
        raise ValueError(
            "don't pass dataframe and arrays to df_to_arrays(), should be either or, "
            f"got {arrays}"
        )

    flat_args = []
    # tuple doesn't support item assignment
    args = list(args)  # type: ignore[assignment]

    for col_name in args:
        if isinstance(col_name, str | int):
            flat_args.append(col_name)
        else:
            flat_args.extend(col_name)

    df_no_nan = df.dropna(subset=flat_args)
    for idx, col_name in enumerate(args):
        if isinstance(col_name, str | int):
            args[idx] = df_no_nan[col_name].to_numpy()  # type: ignore[index]
        else:
            col_data = df_no_nan[[*col_name]].to_numpy().T
            args[idx] = dict(zip(col_name, col_data, strict=True))  # type: ignore[index]

    return args  # type: ignore[return-value]


def bin_df_cols(
    df_in: pd.DataFrame,
    bin_by_cols: Sequence[str],
    *,
    group_by_cols: Sequence[str] = (),
    n_bins: int | Sequence[int] = 100,
    bin_counts_col: str = "bin_counts",
    density_col: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Bin columns of a DataFrame.

    Args:
        df_in (pd.DataFrame): Input dataframe to bin.
        bin_by_cols (Sequence[str]): Columns to bin.
        group_by_cols (Sequence[str]): Additional columns to group by. Defaults to ().
        n_bins (int): Number of bins to use. Defaults to 100.
        bin_counts_col (str): Column name for bin counts. Defaults to "bin_counts".
        density_col (str): Column name for density values. Defaults to "".
        verbose (bool): If True, report df length reduction. Defaults to True.

    Returns:
        pd.DataFrame: Binned DataFrame with original index name and values.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df_in = df_in.copy()

    if isinstance(n_bins, int):
        # broadcast integer n_bins to all bin_by_cols
        n_bins = [n_bins] * len(bin_by_cols)

    if len(bin_by_cols) != len(n_bins):
        raise ValueError(f"{len(bin_by_cols)=} != {len(n_bins)=}")

    cut_cols = [f"{col}_bins" for col in bin_by_cols]
    for col, bins, cut_col in zip(bin_by_cols, n_bins, cut_cols, strict=True):
        df_in[cut_col] = pd.cut(df_in[col].values, bins=bins)

    # Preserve the original index
    orig_index_name = df_in.index.name or "index"
    # Reset index so it participates in groupby. If the index name is already in the
    # columns, we it'll participate already and be set back to the index at the end.
    if orig_index_name not in df_in:
        df_in = df_in.reset_index()

    group = df_in.groupby(by=[*cut_cols, *group_by_cols], observed=True)

    df_bin = group.first().dropna()
    df_bin[bin_counts_col] = group.size()
    df_bin = df_bin.reset_index()

    if verbose:
        print(  # noqa: T201
            f"{1 - len(df_bin) / len(df_in):.1%} sample reduction from binning: from "
            f"{len(df_in):,} to {len(df_bin):,}"
        )

    if density_col:
        # compute kernel density estimate for each bin
        values = df_in[bin_by_cols].dropna().T
        gaussian_kde = scipy.stats.gaussian_kde(values.astype(float))

        xy_binned = df_bin[bin_by_cols].T
        density = gaussian_kde(xy_binned.astype(float))
        df_bin[density_col] = density / density.sum() * len(values)

    # Set the index back to the original index name
    return df_bin.set_index(orig_index_name)
