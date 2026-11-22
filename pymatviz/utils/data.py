"""Element and space-group lookups and display formatting."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import matscikit as mk
import numpy as np
import pandas as pd

from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from matscikit.options import CrystalSystem

_elements_csv = f"{ROOT}/pymatviz/elements.csv"
df_ptable: pd.DataFrame = pd.read_csv(_elements_csv, comment="#").set_index("symbol")


atomic_numbers: dict[str, int] = {
    symbol: atomic_num for atomic_num, symbol in enumerate(df_ptable.index, start=1)
}
element_symbols: dict[int, str] = {
    atomic_num: symbol for symbol, atomic_num in atomic_numbers.items()
}


def spg_to_crystal_sys(spg: float | str) -> CrystalSystem:
    """Crystal system of a space group number or Hermann-Mauguin/Hall symbol.

    Raises:
        ValueError: For unknown symbols and non-integral or out-of-range numbers.
    """
    try:
        number = mk.space_group_number(spg) if isinstance(spg, str) else spg
        if number != int(number):
            raise ValueError(f"{number} is not integral")  # noqa: TRY301
        return mk.crystal_system(int(number))
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid space group {spg}: {exc}") from exc


def spg_num_to_from_symbol(spg: float | str) -> str | int:
    """Short Hermann-Mauguin symbol (e.g. "Fm-3m") of a space group number, or the
    number of a Hermann-Mauguin/Hall symbol (loose spellings and settings allowed).

    Raises:
        ValueError: For unknown symbols or numbers outside 1-230.
    """
    if isinstance(spg, str):
        try:
            return mk.space_group_number(spg)
        except ValueError as exc:
            raise ValueError(f"Invalid space group symbol {spg}") from exc
    if spg % 1 != 0:  # also catches inf and nan
        raise ValueError(f"Invalid space group number {spg}")
    return mk.space_group_symbol(int(spg))


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


def df_to_arrays(df: pd.DataFrame | None, *args: Any) -> list[Any]:
    """Columns of df (rows with NaN in any of them dropped) if df is given, else args
    as arrays. A list of column names becomes a ``{col: array}`` dict.

    Raises:
        TypeError: If df is neither a DataFrame nor None.
        ValueError: If df is None but args holds column names, or df is given and args
            holds arrays.
    """
    if df is None:
        if cols := [arg for arg in args if isinstance(arg, str)]:
            raise ValueError(f"got column names but no df to get data from: {cols}")
        return list(map(np.asarray, args))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df should be pandas DataFrame or None, got {type(df)}")
    is_col = [isinstance(arg, str | int) for arg in args]
    if bad := [
        arg
        for arg, single in zip(args, is_col, strict=True)
        if not single
        and not (
            isinstance(arg, list | tuple) and all(isinstance(c, str | int) for c in arg)
        )
    ]:
        raise ValueError(
            f"with a df, pass column names (or lists of them), not data, got {bad}"
        )
    flat_cols = [
        col
        for arg, single in zip(args, is_col, strict=True)
        for col in ([arg] if single else arg)
    ]
    df_no_nan = df.dropna(subset=flat_cols)
    return [
        df_no_nan[arg].to_numpy()
        if single
        else {col: df_no_nan[col].to_numpy() for col in arg}
        for arg, single in zip(args, is_col, strict=True)
    ]
