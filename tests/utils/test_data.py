from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz.utils.data import df_to_arrays
from tests.conftest import y_pred, y_true


if TYPE_CHECKING:
    from matscikit.options import CrystalSystem


SPG_BY_CRYSTAL_SYS = {
    "triclinic": (1, "P1", "P-1"),
    "monoclinic": (15, "P2/m", "C2/c", "P2_1/c", "P12_1/c1", "C12/c1"),
    "orthorhombic": (16, "Pnma", "Cmcm"),
    "tetragonal": (75, "P4/mmm", "I4/mcm"),
    "trigonal": (143, "R-3m", "R3c"),
    "hexagonal": (168, "P6/mmm", "P6_3/mmc"),
    "cubic": (230, "Fm-3m", "Ia-3d"),
}


@pytest.mark.parametrize(
    ("spg", "crystal_sys"),
    [
        (spg, crystal_sys)
        for crystal_sys, spgs in SPG_BY_CRYSTAL_SYS.items()
        for spg in spgs
    ],
)
def test_spg_to_crystal_sys(spg: int | str, crystal_sys: CrystalSystem) -> None:
    """Space group numbers and symbols map to their crystal system."""
    assert crystal_sys == pmv.utils.spg_to_crystal_sys(spg)


@pytest.mark.parametrize(
    "spg", [-1, 0, 231, 1.2, float("inf"), "invalid", "P999", "X2/m"]
)
def test_spg_to_crystal_sys_invalid(spg: float | str) -> None:
    """Out-of-range numbers and unknown symbols raise."""
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        pmv.utils.spg_to_crystal_sys(spg)


def test_si_fmt() -> None:
    """Format floats with SI prefixes, custom formats, separators and thresholds."""
    assert pmv.utils.si_fmt(0) == "0.0"
    assert pmv.utils.si_fmt(123) == "123.0"
    assert pmv.utils.si_fmt(1234) == "1.2k"
    assert pmv.utils.si_fmt(123456) == "123.5k"
    assert pmv.utils.si_fmt(12345678, fmt=">6.2f", sep=" ") == " 12.35 M"
    assert pmv.utils.si_fmt(-0.00123, fmt=".3g", binary=False) == "-1.23m"
    assert pmv.utils.si_fmt(0.00000123, fmt="5.1f", sep="\t", binary=True) == "  1.3\tμ"
    assert pmv.utils.si_fmt(1.23e-6, fmt="5.1f", sep="\t", binary=False) == "  1.2\tμ"
    assert pmv.utils.si_fmt(0.321, fmt=".2f") == "0.32"
    assert pmv.utils.si_fmt(-0.93) == "-0.9"
    assert pmv.utils.si_fmt(-0.93, fmt=".2f") == "-0.93"
    assert pmv.utils.si_fmt(-0.1) == "-0.1"
    assert pmv.utils.si_fmt(-0.001) == "-1.0m"
    assert pmv.utils.si_fmt(-0.001, decimal_threshold=0.001, fmt=".3f") == "-0.001"
    assert pmv.utils.si_fmt(-1) == "-1.0"
    assert pmv.utils.si_fmt(1.23456789e-10, fmt="5.1f", sep="\t") == "123.5\tp"


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_si_fmt_extreme_scales(binary: bool, sign: int) -> None:
    """Values beyond the last prefix retain their magnitude."""
    factor = 1024 if binary else 1000
    assert pmv.utils.si_fmt(sign * factor**9, binary=binary, fmt=".3g") == (
        f"{sign * factor:.3g}Y"
    )
    assert pmv.utils.si_fmt(sign * factor**-9, binary=binary, fmt=".3g") == (
        f"{sign / factor:.3g}y"
    )


def test_si_fmt_int() -> None:
    """Format integers with SI prefixes."""
    assert pmv.utils.si_fmt_int(0) == "0"
    assert pmv.utils.si_fmt_int(123) == "123"
    assert pmv.utils.si_fmt_int(1234) == "1k"
    assert pmv.utils.si_fmt_int(123456) == "123k"
    assert pmv.utils.si_fmt_int(12345678, fmt=">6.2f", sep=" ") == " 12.35 M"
    assert pmv.utils.si_fmt_int(-1) == "-1"
    assert pmv.utils.si_fmt_int(1.23456789e-10, sep="\t") == "123\tp"


@pytest.mark.parametrize(
    ("text", "tag", "title", "style"),
    [
        ("foo", "span", "", ""),
        ("bar", "small", "some title", "color: red;"),
        ("baz", "div", "long title " * 10, "font-size: 0.8em;"),
        ("", "strong", " ", "font-size: 0.8em; font-weight: lighter;"),
        ("", "strong", " ", "small"),
    ],
)
def test_html_tag(text: str, tag: str, title: str, style: str) -> None:
    """Wrap text in a tag with title and style (or the "small" style preset)."""
    orig_style = style
    style = {"small": "font-size: 0.8em; font-weight: lighter;"}.get(style, style)
    attrs = f" {title=} " if title else ""
    attrs += f"{style=}" if style else ""
    assert (
        pmv.utils.html_tag(text, tag=tag, title=title, style=orig_style)
        == f"<{tag}{attrs}>{text}</{tag}>"
    )


def test_spg_num_to_from_symbol_roundtrip() -> None:
    """Number -> symbol -> number roundtrips for all 230 space groups."""
    for num in range(1, 231):
        symbol = pmv.utils.spg_num_to_from_symbol(num)
        assert pmv.utils.spg_num_to_from_symbol(symbol) == num, f"{num=}, {symbol=}"
    assert pmv.utils.spg_num_to_from_symbol(225.0) == "Fm-3m"
    for bad in (2.5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="Invalid space group number"):
            pmv.utils.spg_num_to_from_symbol(bad)


def test_df_to_arrays() -> None:
    """Columns or arrays give the same data, and bad df or column names raise."""
    df_regr = pd.DataFrame({"x": y_true, "y": y_pred})
    for arrays in (df_to_arrays(None, y_true, y_pred), df_to_arrays(df_regr, "x", "y")):
        np.testing.assert_allclose(arrays, [y_true, y_pred], rtol=0, atol=0)
    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        df_to_arrays("foo", y_true, y_pred)  # ty: ignore[invalid-argument-type]
    with pytest.raises(KeyError, match="not-a-col"):
        df_to_arrays(df_regr, "not-a-col", "x")
    with pytest.raises(ValueError, match="got column names but no df"):
        df_to_arrays(None, "x", y_pred)
    with pytest.raises(ValueError, match="with a df, pass column names"):
        df_to_arrays(df_regr, "x", y_pred)
    with pytest.raises(ValueError, match="with a df, pass column names"):
        df_to_arrays(df_regr, "x", [y_pred])
    _, y_cols = df_to_arrays(df_regr, "x", ["y", "x"])
    assert list(y_cols) == ["y", "x"]
    np.testing.assert_array_equal(y_cols["y"], y_pred)
