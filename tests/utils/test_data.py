from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz.utils.data import df_to_arrays
from tests.conftest import y_pred, y_true


if TYPE_CHECKING:
    from pymatviz.typing import CrystalSystem


@pytest.mark.parametrize(
    ("spg", "crystal_sys"),
    [
        (1, "triclinic"),
        (15, "monoclinic"),
        (16, "orthorhombic"),
        (75, "tetragonal"),
        (143, "trigonal"),
        (168, "hexagonal"),
        (230, "cubic"),
        ("P1", "triclinic"),
        ("P-1", "triclinic"),
        ("P2/m", "monoclinic"),
        ("C2/c", "monoclinic"),
        ("Pnma", "orthorhombic"),
        ("Cmcm", "orthorhombic"),
        ("P4/mmm", "tetragonal"),
        ("I4/mcm", "tetragonal"),
        ("R-3m", "trigonal"),
        ("R3c", "trigonal"),
        ("P6/mmm", "hexagonal"),
        ("P6_3/mmc", "hexagonal"),
        ("Fm-3m", "cubic"),
        ("Ia-3d", "cubic"),
        ("P2_1/c", "monoclinic"),
        ("P12_1/c1", "monoclinic"),
        ("C12/c1", "monoclinic"),
    ],
)
def test_spg_to_crystal_sys(spg: int, crystal_sys: CrystalSystem) -> None:
    assert crystal_sys == pmv.utils.spg_to_crystal_sys(spg)


@pytest.mark.parametrize("spg", [-1, 0, 231, 1.2, "invalid", "P999", "X2/m"])
def test_spg_to_crystal_sys_invalid(spg: int) -> None:
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        pmv.utils.spg_to_crystal_sys(spg)


def test_si_fmt() -> None:
    assert pmv.utils.si_fmt(0) == "0.0"
    assert pmv.utils.si_fmt(123) == "123.0"
    assert pmv.utils.si_fmt(1234) == "1.2k"
    assert pmv.utils.si_fmt(123456) == "123.5k"
    assert pmv.utils.si_fmt(12345678, fmt=">6.2f", sep=" ") == " 12.35 M"
    assert pmv.utils.si_fmt(-0.00123, fmt=".3g", binary=False) == "-1.23m"
    assert pmv.utils.si_fmt(0.00000123, fmt="5.1f", sep="\t", binary=True) == "  1.3\tμ"
    assert (
        pmv.utils.si_fmt(0.00000123, fmt="5.1f", sep="\t", binary=False) == "  1.2\tμ"
    )
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
    orig_style = style
    style = {"small": "font-size: 0.8em; font-weight: lighter;"}.get(style, style)
    attrs = f" {title=} " if title else ""
    attrs += f"{style=}" if style else ""
    assert (
        pmv.utils.html_tag(text, tag=tag, title=title, style=orig_style)
        == f"<{tag}{attrs}>{text}</{tag}>"
    )


def test_spg_num_to_from_symbol_roundtrip() -> None:
    """Test that converting from number to symbol and back gives the original number."""
    for num in range(1, 230 + 1):
        symbol = pmv.utils.spg_num_to_from_symbol(num)
        num_back = pmv.utils.spg_num_to_from_symbol(symbol)
        assert num == num_back, f"Roundtrip failed for {num} -> {symbol} -> {num_back}"


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
    with pytest.raises(ValueError, match="pass column names or arrays, not both"):
        df_to_arrays(df_regr, "x", y_pred)
    _, y_cols = df_to_arrays(df_regr, "x", ["y", "x"])
    assert list(y_cols) == ["y", "x"]
    np.testing.assert_array_equal(y_cols["y"], y_pred)
