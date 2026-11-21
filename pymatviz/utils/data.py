"""Element and space-group lookups and display formatting."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from moyopy import SpaceGroupType

from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from pymatviz.typing import CrystalSystem

_elements_csv = f"{ROOT}/pymatviz/elements.csv"
df_ptable: pd.DataFrame = pd.read_csv(_elements_csv, comment="#").set_index("symbol")


atomic_numbers: dict[str, int] = {
    symbol: atomic_num for atomic_num, symbol in enumerate(df_ptable.index, start=1)
}
element_symbols: dict[int, str] = {
    atomic_num: symbol for symbol, atomic_num in atomic_numbers.items()
}


# SpaceGroupType(number).hm_short is separated by a space like "F m -3 m"
_spaced_hm_symbols = {
    SpaceGroupType(number).hm_short: number for number in range(1, 230 + 1)
} | {SpaceGroupType(number).hm_full: number for number in range(1, 230 + 1)}
hm_symbol_to_spg_num_map = _spaced_hm_symbols | {
    spg.replace(" ", ""): number for spg, number in _spaced_hm_symbols.items()
}


def spg_to_crystal_sys(spg: float | str) -> CrystalSystem:
    """Get the crystal system for an international space group number."""
    if isinstance(spg, str):
        spg = hm_symbol_to_spg_num_map.get(spg, spg)

    if not isinstance(spg, int | float) or spg != int(spg):
        raise ValueError(f"Invalid space group {spg}")

    if not (1 <= spg <= 230):
        raise ValueError(f"Invalid space group {spg}, must be 1 <= num <= 230")

    crys_sys_max_spg: tuple[tuple[CrystalSystem, int], ...] = (
        ("triclinic", 2),
        ("monoclinic", 15),
        ("orthorhombic", 74),
        ("tetragonal", 142),
        ("trigonal", 167),
        ("hexagonal", 194),
    )
    return next((crys for crys, max_spg in crys_sys_max_spg if spg <= max_spg), "cubic")


def spg_num_to_from_symbol(spg: int | str) -> str | int:
    """Get the Hermann-Mauguin short or full symbol for an international space group
    number or vice versa.

    Args:
        spg (int | str): A space group number (int) or a Hermann-Mauguin symbol (str).

    Returns:
        int | str: str if input was a number, int if input was a str.
    """
    if isinstance(spg, str):  # Convert symbol to number
        if spg_num := hm_symbol_to_spg_num_map.get(spg):
            return spg_num

        raise ValueError(f"Invalid space group symbol {spg}")

    return SpaceGroupType(spg).hm_short.replace(" ", "")  # Convert number to symbol


def html_tag(text: str, tag: str = "span", style: str = "", title: str = "") -> str:
    """Wrap text in a span with custom style.

    Style defaults to decreased font size and weight e.g. to display units in
    labels. MatterViz axis titles keep the text (and <sub>/<sup>) but drop other
    tags' styling.

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
        for scale in ("k", "M", "G", "T", "P", "E", "Z", "Y"):
            if abs(val) < factor:
                break
            val /= factor
            _scale = scale
    elif val != 0 and abs(val) < decimal_threshold:
        # milli, micro, nano, pico, femto, atto, zepto, yocto
        for scale in ("m", "μ", "n", "p", "f", "a", "z", "y"):
            if abs(val) >= 1:
                break
            val *= factor
            _scale = scale

    return f"{val:{fmt}}{sep}{_scale}"


si_fmt_int = partial(si_fmt, fmt=".0f")


def df_to_arrays(
    df: pd.DataFrame | None,
    *args: str | ArrayLike,
    strict: bool = True,
) -> list[Any]:
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
        list[Any]: Array data for each column name, dictionary of column names and
            array data, or original string args when strict=False and df is not a
            DataFrame.
    """
    if df is None:
        if cols := [arg for arg in args if isinstance(arg, str)]:
            raise ValueError(f"got column names but no df to get data from: {cols}")
        # Convert args to numpy arrays if they are not already
        return list(map(np.asarray, args))

    if not isinstance(df, pd.DataFrame):
        if not strict:
            return list(args)
        raise TypeError(f"df should be pandas DataFrame or None, got {type(df)}")

    if arrays := [arg for arg in args if isinstance(arg, np.ndarray)]:
        raise ValueError(
            "don't pass dataframe and arrays to df_to_arrays(), should be either or, "
            f"got {arrays}"
        )

    flat_args = []
    for col_name in args:
        if isinstance(col_name, str | int):
            flat_args.append(col_name)
        else:
            flat_args.extend(col_name)  # ty: ignore[invalid-argument-type]

    df_no_nan = df.dropna(subset=flat_args)
    out_args: list[Any] = []
    for col_name in args:
        if isinstance(col_name, str | int):
            out_args.append(df_no_nan[col_name].to_numpy())
        else:
            col_data = df_no_nan[[*col_name]].to_numpy().T  # ty: ignore[not-iterable]
            out_args.append(dict(zip(col_name, col_data, strict=True)))  # ty: ignore[invalid-argument-type,no-matching-overload]

    return out_args
