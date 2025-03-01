"""Data processing utils:
* df_ptable (DataFrame): Periodic table.
* atomic_numbers (dict[str, int]): Map elements to atomic numbers.
* element_symbols (dict[int, str]): Map atomic numbers to elements.

- bin_df_cols: Bin columns of a DataFrame.
- spg_to_crystal_sys: Get the crystal system for an international
    space group number.
- df_to_arrays: Convert DataFrame to arrays.
- html_tag: Wrap text in a span with custom style.
- normalize_to_dict: Normalize object or dict/list/tuple of them into to a dict.
- patch_dict: Context manager to temporarily patch the specified keys in a
    dictionary and restore it to its original state on context exit.
- si_fmt: Convert large numbers into human readable format using SI suffixes.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats
from moyopy import SpaceGroupType
from pymatgen.core import Structure

from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from pymatviz.typing import CrystalSystem, T

_elements_csv = f"{ROOT}/pymatviz/elements.csv"
df_ptable: pd.DataFrame = pd.read_csv(_elements_csv, comment="#").set_index("symbol")


atomic_numbers: dict[str, int] = {}
element_symbols: dict[int, str] = {}

for Z, symbol in enumerate(df_ptable.index, start=1):
    atomic_numbers[symbol] = Z
    element_symbols[Z] = symbol


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


# SpaceGroupType(number).hm_short is separated by a space like "F m -3 m"
hm_symbol_to_spg_num_map = {
    SpaceGroupType(number).hm_short: number for number in range(1, 230 + 1)
} | {SpaceGroupType(number).hm_full: number for number in range(1, 230 + 1)}
for spg, val in [*hm_symbol_to_spg_num_map.items()]:
    hm_symbol_to_spg_num_map[spg.replace(" ", "")] = val


def spg_to_crystal_sys(spg: float | str) -> CrystalSystem:
    """Get the crystal system for an international space group number."""
    # Ensure integer or float with no decimal part
    if isinstance(spg, str):
        spg = hm_symbol_to_spg_num_map.get(spg, spg)

    if not isinstance(spg, int | float) or spg != int(spg):
        raise ValueError(f"Invalid space group {spg}")

    if not (1 <= spg <= 230):
        raise ValueError(f"Invalid space group {spg}, must be 1 <= num <= 230")

    if 1 <= spg <= 2:
        return "triclinic"
    if spg <= 15:
        return "monoclinic"
    if spg <= 74:
        return "orthorhombic"
    if spg <= 142:
        return "tetragonal"
    if spg <= 167:
        return "trigonal"
    if spg <= 194:
        return "hexagonal"
    return "cubic"


def spg_num_to_from_symbol(spg: int | str) -> str | int:
    """Get the Hermann-Mauguin short or full symbol for an international space group
    number or vice versa.

    Args:
        spg: Either a space group number (int) or a Hermann-Mauguin symbol (str).

    Returns:
        int | str: str if input was a number, int if input was a str.
    """
    if isinstance(spg, str):  # Convert symbol to number
        if spg_num := hm_symbol_to_spg_num_map.get(spg):
            return spg_num

        raise ValueError(f"Invalid space group symbol {spg}")

    return SpaceGroupType(spg).hm_short.replace(" ", "")  # Convert number to symbol


def df_to_arrays(
    df: pd.DataFrame | None,
    *args: str | Sequence[str] | Sequence[ArrayLike],
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
        # pass through args as-is
        return args  # type: ignore[return-value]

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


def html_tag(text: str, tag: str = "span", style: str = "", title: str = "") -> str:
    """Wrap text in a span with custom style.

    Style defaults to decreased font size and weight e.g. to display units
    in plotly labels and annotations.

    Args:
        text (str): Text to wrap in span.
        tag (str, optional): HTML tag name. Defaults to "span".
        style (str, optional): CSS style string. Defaults to "". Special keys:
            "small": font-size: 0.8em; font-weight: lighter;
            "bold": font-weight: bold;
            "italic": font-style: italic;
            "underline": text-decoration: underline;
        title (str | None, optional): Title attribute which displays additional
            information in a tooltip. Defaults to "".

    Returns:
        str: HTML string with tag-wrapped text.
    """
    style = {
        "small": "font-size: 0.8em; font-weight: lighter;",
        "bold": "font-weight: bold;",
        "italic": "font-style: italic;",
        "underline": "text-decoration: underline;",
    }.get(style, style)
    attr_str = f" {title=}" if title else ""
    if style:
        attr_str += f" {style=}"
    return f"<{tag}{attr_str}>{text}</{tag}>"


def normalize_to_dict(
    inputs: T | Sequence[T] | dict[str, T],
    cls: type[T] = Structure,
    key_gen: Callable[[T], str] = lambda obj: getattr(
        obj, "formula", type(obj).__name__
    ),
) -> dict[str, T]:
    """Normalize any kind of object or dict/list/tuple of them into to a dictionary.

    Args:
        inputs: A single object, a sequence of objects, or a dictionary of objects.
        cls (type[T], optional): The class of the objects to normalize. Defaults to
            pymatgen.core.Structure.
        key_gen (Callable[[T], str], optional): A function that generates a key for
            each object. Defaults to using the object's formula, assuming the objects
            are pymatgen.core.(Structure|Molecule).

    Returns:
        A dictionary of objects with keys as object formulas or given keys.

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


@contextmanager
def patch_dict(
    dct: dict[Any, Any], *args: Any, **kwargs: Any
) -> Generator[dict[Any, Any], None, None]:
    """Context manager to temporarily patch the specified keys in a dictionary and
    restore it to its original state on context exit.

    Useful e.g. for temporary plotly fig.layout mutations:

        with patch_dict(fig.layout, showlegend=False):
            fig.write_image("plot.pdf")

    Args:
        dct (dict): The dictionary to be patched.
        *args: Only first element is read if present. A single dictionary containing the
            key-value pairs to patch.
        **kwargs: The key-value pairs to patch, provided as keyword arguments.

    Yields:
        dict: The patched dictionary incl. temporary updates.
    """
    # if both args and kwargs are passed, kwargs will overwrite args
    updates = {**args[0], **kwargs} if args and isinstance(args[0], dict) else kwargs

    # save original values as shallow copy for speed
    # warning: in-place changes to nested dicts and objects will persist beyond context!
    patched = dct.copy()

    # apply updates
    patched.update(updates)

    yield patched


def si_fmt(
    val: float,
    *,
    fmt: str = ".1f",
    sep: str = "",
    binary: bool = False,
    decimal_threshold: float = 0.01,
) -> str:
    """Convert large numbers into human readable format using SI suffixes.

    Supports binary (1024) and metric (1000) mode.

    https://nist.gov/pml/weights-and-measures/metric-si-prefixes

    Args:
        val (int | float): Some numerical value to format.
        binary (bool, optional): If True, scaling factor is 2^10 = 1024 else 1000.
            Defaults to False.
        fmt (str): f-string format specifier. Configure precision and left/right
            padding in returned string. Defaults to ".1f". Can be used to ensure leading
            or trailing whitespace for shorter numbers. See
            https://docs.python.org/3/library/string.html#format-specification-mini-language.
        sep (str): Separator between number and postfix. Defaults to "".
        decimal_threshold (float): abs(value) below 1 but above this threshold will be
            left as decimals. Only below this threshold is a greek suffix added (milli,
            micro, etc.). Defaults to 0.01. i.e. 0.01 -> "0.01" while
            0.0099 -> "9.9m". Setting decimal_threshold=0.1 would format 0.01 as "10m"
            and leave 0.1 as is.

    Returns:
        str: Formatted number.
    """
    factor = 1024 if binary else 1000
    _scale = ""

    if abs(val) >= 1:
        # 1, Kilo, Mega, Giga, Tera, Peta, Exa, Zetta, Yotta
        for _scale in ("", "k", "M", "G", "T", "P", "E", "Z", "Y"):
            if abs(val) < factor:
                break
            val /= factor
    elif val != 0 and abs(val) < decimal_threshold:
        # milli, micro, nano, pico, femto, atto, zepto, yocto
        for _scale in ("", "m", "μ", "n", "p", "f", "a", "z", "y"):
            if abs(val) >= 1:
                break
            val *= factor

    return f"{val:{fmt}}{sep}{_scale}"


si_fmt_int = partial(si_fmt, fmt=".0f")
