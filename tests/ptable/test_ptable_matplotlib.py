from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pymatgen.core.periodic_table import Element

import pymatviz as pmv
from pymatviz.enums import Key


if TYPE_CHECKING:
    from typing import Any, Literal

    from pymatgen.core import Composition

df_ptable = pmv.df_ptable.copy()  # avoid changing the df in place


@pytest.fixture
def glass_elem_counts(glass_formulas: pd.Series[Composition]) -> pd.Series[int]:
    return pmv.count_elements(glass_formulas)


@pytest.fixture
def steel_formulas() -> list[str]:
    """Unusually fractional compositions, good for testing edge cases.

    Output of:
    from matminer.datasets import load_dataset

    load_dataset("matbench_steels").composition.head(2)
    """
    return [
        "Fe0.620C0.000953Mn0.000521Si0.00102Cr0.000110Ni0.192Mo0.0176V0.000112"
        "Nb0.0000616Co0.146Al0.00318Ti0.0185",
        "Fe0.623C0.00854Mn0.000104Si0.000203Cr0.147Ni0.0000971Mo0.0179V0.00515"
        "N0.00163Nb0.0000614Co0.188W0.00729Al0.000845",
    ]


@pytest.fixture
def steel_elem_counts(steel_formulas: pd.Series[Composition]) -> pd.Series[int]:
    return pmv.count_elements(steel_formulas)


class TestPtableHeatmap:
    @pytest.mark.parametrize("hide_f_block", ["auto", False, True])
    def test_basic_heatmap_plotter(self, hide_f_block: bool | Literal["auto"]) -> None:
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass],
            hide_f_block=hide_f_block,
            cbar_title="Element Count",
            return_type="figure",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 127 if hide_f_block is True else 181, len(fig.axes)

    @pytest.mark.parametrize("log", [False, True])
    def test_log_scale(self, log: bool) -> None:
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass], log=log, return_type="figure"
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 181

    @pytest.mark.parametrize(
        "values_show_mode", ["percent", "fraction", "value", "off"]
    )
    def test_values_show_mode(
        self, values_show_mode: Literal["percent", "fraction", "value", "off"]
    ) -> None:
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass],
            value_show_mode=values_show_mode,
            return_type="figure",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 181

    @pytest.mark.parametrize("values_show_mode", ["percent", "fraction"])
    def test_log_in_percent_mode(
        self, values_show_mode: Literal["percent", "fraction", "value", "off"]
    ) -> None:
        with pytest.raises(ValueError, match="Combining log scale and"):
            pmv.ptable_heatmap(
                df_ptable[Key.atomic_mass],
                log=True,
                value_show_mode=values_show_mode,
            )

    @pytest.mark.parametrize(
        "cbar_range", [(0, 300), (None, 300), (0, None), (None, None)]
    )
    def test_cbar_range(self, cbar_range: tuple[float | None, float | None]) -> None:
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass], cbar_range=cbar_range, return_type="figure"
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 181

    @pytest.mark.parametrize("values_fmt", ["auto", ".3g", ".2g"])
    def test_values_fmt(self, values_fmt: str) -> None:
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass], value_fmt=values_fmt, return_type="figure"
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 181

    def test_cbar_kwargs(self) -> None:
        cbar_kwargs = dict(orientation="horizontal")
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass],
            cbar_kwargs=cbar_kwargs,
            return_type="figure",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 181

    def test_tile_size(self) -> None:
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass], tile_size=(1, 1), return_type="figure"
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 181

    def test_text_style(self) -> None:
        symbol_kwargs = dict(fontsize=12)
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass],
            text_colors="red",
            symbol_kwargs=symbol_kwargs,
            return_type="figure",
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 181

    def test_return_type(self) -> None:
        fig = pmv.ptable_heatmap(
            df_ptable[Key.atomic_mass],
            return_type="figure",
        )
        assert isinstance(fig, plt.Figure)

        with pytest.warns(match="We encourage you to return plt.figure"):
            ax = pmv.ptable_heatmap(df_ptable[Key.atomic_mass])
        assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize("hide_f_block", ["auto", False, True])
def test_ptable_heatmap_splits(hide_f_block: bool) -> None:
    """Test pmv.ptable_heatmap_splits with arbitrary data length."""
    rng = np.random.default_rng()
    data_dict: dict[str, Any] = {
        elem.symbol: [
            rng.integers(0, 10, size=rng.integers(1, 4))  # random value for each split
        ]
        for elem in Element
    }

    # Also test different data types
    data_dict["H"] = {"a": 1, "b": 2}.values()
    data_dict["He"] = [1, 2]
    data_dict["Li"] = np.array([1, 2])
    data_dict["Be"] = 1
    data_dict["B"] = 2.0

    cbar_title = "Periodic Table Evenly-Split Tiles Plots"
    fig = pmv.ptable_heatmap_splits(
        data_dict,
        colormap="coolwarm",
        start_angle=135,
        cbar_title=cbar_title,
        hide_f_block=hide_f_block,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 127 if hide_f_block is True else 181
    cbar_ax = fig.axes[-1]
    assert cbar_ax.get_title() == cbar_title


def test_ptable_heatmap_ratio(
    steel_formulas: list[str],
    glass_formulas: list[str],
    steel_elem_counts: pd.Series[int],
    glass_elem_counts: pd.Series[int],
) -> None:
    # composition strings
    not_in_numerator = ("lightgray", "gray: not in 1st list")
    not_in_denominator = ("lightskyblue", "blue: not in 2nd list")
    not_in_either = ("white", "white: not in either")

    # Call the function and get the Figure
    fig = pmv.ptable_heatmap_ratio(
        glass_formulas,
        steel_formulas,
        not_in_numerator=not_in_numerator,
        not_in_denominator=not_in_denominator,
        not_in_either=not_in_either,
    )

    # Ensure the returned object is a Figure
    assert isinstance(fig, plt.Figure)

    # Extract the Axes from the Figure
    ax = fig.gca()

    # Check presence of legend handles "not in numerator" and "not in denominator"
    legend = ax.get_legend()
    assert legend is None

    # Get text annotations
    texts = fig.texts
    assert len(texts) == 3
    all_texts = [txt.get_text() for txt in texts]
    for not_in in (not_in_numerator, not_in_denominator, not_in_either):
        assert not_in[1] in all_texts

    # Element counts
    fig = pmv.ptable_heatmap_ratio(glass_elem_counts, steel_elem_counts, normalize=True)

    # Mixed element counts and composition
    fig = pmv.ptable_heatmap_ratio(
        glass_formulas, steel_elem_counts, exclude_elements=("O", "P")
    )
    fig = pmv.ptable_heatmap_ratio(
        glass_elem_counts, steel_formulas, not_in_numerator=None
    )


@pytest.mark.parametrize(
    ("data", "symbol_pos", "hist_kwargs"),
    [
        ({"H": [1, 2, 3], "He": [4, 5, 6]}, (0, 0), None),
        (pd.DataFrame({"Fe": [1, 2, 3], "O": [4, 5, 6]}), (0, 0), None),
        (
            dict(H=[1, 2, 3], He=[4, 5, 6]),
            (1, 1),
            {},
        ),
        (
            dict(H=np.array([1, 2, 3]), He=np.array([4, 5, 6])),
            (1, 1),
            {},
        ),
        (
            pd.Series([[1, 2, 3], [4, 5, 6]], index=["H", "He"]),
            (1, 1),
            dict(xy=(0, 0)),
        ),
    ],
)
def test_ptable_hists(
    data: pd.DataFrame | pd.Series | dict[str, list[int]],
    symbol_pos: tuple[int, int],
    hist_kwargs: dict[str, Any],
) -> None:
    fig_0 = pmv.ptable_hists(
        data,
        symbol_pos=symbol_pos,
        child_kwargs=hist_kwargs,
    )
    assert isinstance(fig_0, plt.Figure)

    # Test partial x_range
    fig_1 = pmv.ptable_hists(
        data,
        x_range=(2, None),
        symbol_pos=symbol_pos,
        child_kwargs=hist_kwargs,
    )
    assert isinstance(fig_1, plt.Figure)


@pytest.mark.parametrize("hide_f_block", [False, True])
def test_ptable_lines(hide_f_block: bool) -> None:
    """Test pmv.ptable_lines."""
    fig = pmv.ptable_lines(
        data={
            "Fe": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "O": [[10, 11], [12, 13], [14, 15]],
        },
        hide_f_block=hide_f_block,
    )
    assert isinstance(fig, plt.Figure)
    expected_n_axes = 126 if hide_f_block else 180
    assert len(fig.axes) == expected_n_axes


@pytest.mark.parametrize("hide_f_block", [False, True])
def test_ptable_scatters(hide_f_block: bool) -> None:
    """Test pmv.ptable_scatters."""
    fig = pmv.ptable_scatters(
        data={
            "Fe": [[1, 2, 3], [4, 5, 6]],
            "O": [[10, 11], [12, 13]],
        },
        hide_f_block=hide_f_block,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 126 if hide_f_block else 181


def test_ptable_scatters_colored() -> None:
    """Test pmv.ptable_scatters with 3rd color dimension."""
    fig = pmv.ptable_scatters(
        data={
            "Fe": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "O": [[10, 11], [12, 13], [14, 15]],
        },
        colormap="coolwarm",
        cbar_title="Test pmv.ptable_scatters",
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 127
