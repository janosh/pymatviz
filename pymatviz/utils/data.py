"""Data processing utils:
* df_ptable (DataFrame): Periodic table.
* atomic_numbers (dict[str, int]): Map elements to atomic numbers.
* element_symbols (dict[int, str]): Map atomic numbers to elements.

- spg_to_crystal_sys: Get the crystal system for an international
    space group number.
- html_tag: Wrap text in a span with custom style.
- patch_dict: Context manager to temporarily patch the specified keys in a
    dictionary and restore it to its original state on context exit.
- si_fmt/si_fmt_int: Convert large numbers into human readable format using SI suffixes.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd
from moyopy import SpaceGroupType

from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from pymatviz.typing import CrystalSystem

_elements_csv = f"{ROOT}/pymatviz/elements.csv"
df_ptable: pd.DataFrame = pd.read_csv(_elements_csv, comment="#").set_index("symbol")


atomic_numbers: dict[str, int] = {}
element_symbols: dict[int, str] = {}

for Z, symbol in enumerate(df_ptable.index, start=1):
    atomic_numbers[symbol] = Z
    element_symbols[Z] = symbol


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
