from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pymatgen.core.periodic_table import Element

from pymatviz import (
    count_elements,
    ptable_heatmap,
    ptable_heatmap_ratio,
    ptable_heatmap_splits,
    ptable_hists,
    ptable_lines,
    ptable_scatters,
)
from pymatviz.enums import ElemColorScheme
from pymatviz.ptable import PTableProjector
from pymatviz.utils import df_ptable, si_fmt, si_fmt_int


if TYPE_CHECKING:
    from typing import Any, ClassVar

    from pymatgen.core import Composition


class TestPTableProjector:
    test_dict: ClassVar = {
        "H": 1,  # int
        "He": [2.0, 4.0],  # float list
        "Li": np.array([6.0, 8.0]),  # float array
        "Na": 11.0,  # float
        "Mg": {"a": -1, "b": 14.0}.values(),  # dict_values
        "Al": {-1, 2.3},  # mixed int/float set
    }

    def test_elem_types(self) -> None:
        projector = PTableProjector(data=self.test_dict)
        assert projector.elem_types == {
            "Noble Gas",
            "Metal",
            "Alkaline Earth Metal",
            "Nonmetal",
            "Alkali Metal",
        }

    def test_elem_colors(self) -> None:
        data = self.test_dict
        projector = PTableProjector(data=data)
        color_subset = {
            "Ac": (0.4392156862745098, 0.6705882352941176, 0.9803921568627451),
            "Zr": (0, 1, 0),
        }
        assert projector.elem_colors.items() > color_subset.items()

        vesta_colors = PTableProjector(
            data=data, elem_colors=ElemColorScheme.vesta
        ).elem_colors
        assert vesta_colors == projector.elem_colors
        jmol_colors = PTableProjector(
            data=data, elem_colors=ElemColorScheme.jmol
        ).elem_colors
        assert jmol_colors != projector.elem_colors

        with pytest.raises(
            ValueError,
            match="elem_colors must be 'vesta', 'jmol', or a custom dict, "
            "got elem_colors='foobar'",
        ):
            PTableProjector(data=data, elem_colors="foobar")  # type: ignore[arg-type]

    def test_hide_f_block(self) -> None:
        # check default is True if no f-block elements in data
        assert PTableProjector(data=self.test_dict).hide_f_block is True
        assert PTableProjector(data={"H": 1}).hide_f_block is True
        # check default is False if f-block elements in data
        assert PTableProjector(data=self.test_dict | {"La": 1}).hide_f_block is False
        assert PTableProjector(data={"La": 1}).hide_f_block is False
        # check override
        assert PTableProjector(data={"La": 1}, hide_f_block=True).hide_f_block is True

    def test_get_elem_type_color(self) -> None:
        projector = PTableProjector(data=self.test_dict)

        assert projector.get_elem_type_color("H") == "green"
        assert projector.get_elem_type_color("Fe") == "blue"

    @pytest.mark.parametrize(
        "data, elem_type_colors",
        [
            # data=dict, elem colors=empty dict
            ({"Li": [1, 2, 3], "Na": [4, 5, 6], "K": [7, 8, 9]}, {}),
            # data=series, elem colors=dict
            (
                pd.Series([1, 2, 3], index=["Fe", "Fe", "Fe"]),
                {"Transition Metal": "red", "Nonmetal": "blue"},
            ),
            # data=dataframe, elem colors=None
            (pd.DataFrame({"Fe": [1, 2, 3], "O": [4, 5, 6], "P": [7, 8, 9]}), None),
        ],
    )
    def test_add_element_type_legend_data_types(
        self,
        data: pd.DataFrame | pd.Series | dict[str, list[float]],
        elem_type_colors: dict[str, str] | None,
    ) -> None:
        projector = PTableProjector(data=data, elem_type_colors=elem_type_colors)

        legend_title = "Element Types"
        legend_kwargs = dict(loc="upper right", ncol=5, fontsize=12, title=legend_title)
        projector.add_elem_type_legend(kwargs=legend_kwargs)

        legend = plt.gca().get_legend()
        assert isinstance(legend, mpl.legend.Legend)
        assert len(legend.get_texts()) in {1, 2}
        legend_labels = {text.get_text() for text in legend.get_texts()}
        assert legend_labels <= {"Transition Metal", "Alkali Metal", "Nonmetal"}
        assert legend._ncols == 5  # noqa: SLF001

        assert legend.get_title().get_text() == legend_title
        assert legend.get_texts()[0].get_fontsize() == 12


@pytest.fixture()
def glass_elem_counts(glass_formulas: pd.Series[Composition]) -> pd.Series[int]:
    return count_elements(glass_formulas)


@pytest.fixture()
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


@pytest.fixture()
def steel_elem_counts(steel_formulas: pd.Series[Composition]) -> pd.Series[int]:
    return count_elements(steel_formulas)


def test_ptable_heatmap(
    glass_formulas: list[str], glass_elem_counts: pd.Series[int]
) -> None:
    ax = ptable_heatmap(glass_formulas)
    assert isinstance(ax, plt.Axes)
    assert len(ax.texts) == 236
    # ensure only 118 elements are labeled
    labels = {txt.get_text() for txt in ax.texts}
    allowed_labels = {*df_ptable.index} | {*map(str, range(182))} | {"-"}
    assert labels <= allowed_labels, f"{labels - allowed_labels=}"

    ptable_heatmap(glass_formulas, log=True)

    # custom colormap
    ptable_heatmap(glass_formulas, log=True, colorscale="summer")

    # heat_mode normalized to total count
    ptable_heatmap(glass_formulas, heat_mode="fraction")
    ptable_heatmap(glass_formulas, heat_mode="percent")

    # without heatmap values
    ptable_heatmap(glass_formulas, heat_mode=None)
    ptable_heatmap(glass_formulas, log=True, heat_mode=None)

    # element properties as heatmap values
    ptable_heatmap(df_ptable.atomic_mass)

    # element properties as heatmap values
    ptable_heatmap(df_ptable.atomic_mass, text_color=("red", "blue"))

    # custom cbar_range
    ptable_heatmap(glass_formulas, cbar_range=(0, 100))
    ptable_heatmap(glass_formulas, log=True, cbar_range=(None, 100))
    ptable_heatmap(glass_formulas, log=True, cbar_range=(1, None))

    with pytest.raises(ValueError, match="Invalid vmin or vmax"):
        # can't use cbar_min=0 with log=True
        ptable_heatmap(glass_formulas, log=True, cbar_range=(0, None))

    # cbar_kwargs
    ax = ptable_heatmap(glass_formulas, cbar_kwargs=dict(orientation="horizontal"))
    cax = ax.inset_axes([0.1, 0.9, 0.8, 0.05])
    ptable_heatmap(glass_formulas, cbar_kwargs={"cax": cax, "format": "%.3f"})

    # element counts
    ptable_heatmap(glass_elem_counts)

    with pytest.raises(ValueError, match="Combining log color scale and"):
        ptable_heatmap(glass_formulas, log=True, heat_mode="percent")

    ptable_heatmap(glass_elem_counts, exclude_elements=["O", "P"])

    with pytest.raises(ValueError, match=r"Unexpected symbol\(s\) foobar"):
        ptable_heatmap(glass_elem_counts, exclude_elements=["foobar"])

    # cbar_fmt as string
    ax = ptable_heatmap(glass_elem_counts, cbar_fmt=".3f")
    cbar_labels = [label.get_text() for label in ax.child_axes[0].get_xticklabels()]
    assert cbar_labels[:2] == ["0.000", "50.000"]

    # cbar_fmt as function
    ax = ptable_heatmap(glass_elem_counts, fmt=si_fmt)
    ax = ptable_heatmap(
        glass_elem_counts, fmt=lambda x: f"{x:.0f}", cbar_fmt=si_fmt_int
    )
    ax = ptable_heatmap(glass_elem_counts, cbar_fmt=lambda x, _: f"{x:.3f} kg")

    ax = ptable_heatmap(glass_elem_counts, heat_mode="percent", cbar_fmt=".3%")
    cbar_ax = ax.child_axes[0]
    cbar_1st_label = cbar_ax.get_xticklabels()[0].get_text()
    assert cbar_1st_label == "0.000%"
    cbar_title = cbar_ax.title
    assert str(cbar_title) == "Text(0.5, 1.0, 'Element Count')"
    # check colorbar title font color is black and hence visible on white background
    assert cbar_title.get_color() == "black"

    # tile_size
    ptable_heatmap(df_ptable.atomic_mass, tile_size=1)
    ptable_heatmap(df_ptable.atomic_mass, tile_size=(0.9, 1))

    # bad colorscale should raise ValueError
    bad_name = "bad color scale"
    with pytest.raises(
        ValueError,
        match=f"{bad_name!r} is not a valid value for name; supported values are "
        "'Accent', 'Accent_r'",
    ):
        ptable_heatmap(glass_formulas, colorscale=bad_name)

    # test text_style
    ptable_heatmap(glass_formulas, text_style=dict(color="red", fontsize=12))

    # test show_scale (with heat_mode)
    ptable_heatmap(glass_formulas, heat_mode="percent", show_scale=False)


def test_ptable_heatmap_text_color_consistency(glass_formulas: list[str]) -> None:
    ax = ptable_heatmap(glass_formulas)
    check_text_color_consistency(ax)

    ax = ptable_heatmap(glass_formulas, log=True)
    check_text_color_consistency(ax)

    ax = ptable_heatmap(glass_formulas, text_color=("red", "blue"))
    check_text_color_consistency(ax)

    ax = ptable_heatmap(glass_formulas, heat_mode="percent")
    check_text_color_consistency(ax)

    ax = ptable_heatmap(glass_formulas, show_values=False)
    check_text_color_consistency(ax)


def check_text_color_consistency(ax: plt.Axes) -> None:
    """Check that in every element tile of a matplotlib ptable_heatmap, the element
    symbol text color matches the heatmap value's text color."""
    rectangles = [patch for patch in ax.patches if isinstance(patch, plt.Rectangle)]

    assert len(ax.texts) > 0, "No text elements found in the plot"
    assert len(rectangles) > 0, "No rectangle elements found in the plot"

    for rect in rectangles:
        x, y = rect.get_xy()
        width, height = rect.get_width(), rect.get_height()

        # Find texts within this rectangle
        rect_texts = [
            text
            for text in ax.texts
            if x <= text.get_position()[0] <= x + width
            and y <= text.get_position()[1] <= y + height
        ]
        symbol = rect_texts[0].get_text()
        n_rects = len(rect_texts)
        assert n_rects in {1, 2}, (
            f"Unexpected number of text elements={n_rects} in element tile at "
            f"{(symbol, x, y)}, should be 1 or 2 (depending on show_values)"
        )

        if len(rect_texts) == 2:  # Element symbol and value
            symbol_text, value_text = rect_texts
            assert (
                symbol_text.get_color() == value_text.get_color()
            ), f"Text color mismatch for element at position {(x, y)}"
        elif len(rect_texts) == 1:  # Only element symbol (when show_values=False)
            pass


@pytest.mark.parametrize("hide_f_block", [None, False, True])
def test_ptable_heatmap_splits(hide_f_block: bool) -> None:
    """Test ptable_heatmap_splits with arbitrary data length."""
    data_dict: dict[str, Any] = {
        elem.symbol: [
            random.randint(0, 10)  # random value for each split
            # random number of 1-4 splits per element
            for _ in range(random.randint(1, 4))
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
    fig = ptable_heatmap_splits(
        data_dict,
        colormap="coolwarm",
        start_angle=135,
        cbar_title=cbar_title,
        hide_f_block=hide_f_block,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 127 if hide_f_block else 181
    cbar_ax = fig.axes[-1]
    assert cbar_ax.get_title() == cbar_title


def test_ptable_heatmap_ratio(
    steel_formulas: list[str],
    glass_formulas: list[str],
    steel_elem_counts: pd.Series[int],
    glass_elem_counts: pd.Series[int],
) -> None:
    # composition strings
    not_in_numerator = ("#eff", "gray: not in 1st list")
    not_in_denominator = ("lightskyblue", "blue: not in 2nd list")
    not_in_either = ("white", "white: not in either")
    ax = ptable_heatmap_ratio(
        glass_formulas,
        steel_formulas,
        not_in_numerator=not_in_numerator,
        not_in_denominator=not_in_denominator,
        not_in_either=not_in_either,
    )
    assert isinstance(ax, plt.Axes)

    # check presence of legend handles "not in numerator" and "not in denominator"
    legend = ax.get_legend()
    assert legend is None
    # get text annotations
    texts = ax.texts
    assert len(texts) == 239
    all_texts = [txt.get_text() for txt in texts]
    for not_in in (not_in_numerator, not_in_denominator, not_in_either):
        assert not_in[1] in all_texts

    # element counts
    ptable_heatmap_ratio(glass_elem_counts, steel_elem_counts, normalize=True)

    # mixed element counts and composition
    ptable_heatmap_ratio(glass_formulas, steel_elem_counts, exclude_elements=("O", "P"))
    ptable_heatmap_ratio(glass_elem_counts, steel_formulas, not_in_numerator=None)


@pytest.mark.parametrize(
    "data, symbol_pos, hist_kwargs",
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
    fig_0 = ptable_hists(
        data,
        symbol_pos=symbol_pos,
        child_kwargs=hist_kwargs,
    )
    assert isinstance(fig_0, plt.Figure)

    # Test partial x_range
    fig_1 = ptable_hists(
        data,
        x_range=(2, None),
        symbol_pos=symbol_pos,
        child_kwargs=hist_kwargs,
    )
    assert isinstance(fig_1, plt.Figure)


@pytest.mark.parametrize("hide_f_block", [False, True])
def test_ptable_lines(hide_f_block: bool) -> None:
    """Test ptable_lines."""
    fig = ptable_lines(
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
    """Test ptable_scatters."""
    fig = ptable_scatters(
        data={
            "Fe": [[1, 2, 3], [4, 5, 6]],
            "O": [[10, 11], [12, 13]],
        },
        hide_f_block=hide_f_block,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 126 if hide_f_block else 181


def test_ptable_scatters_colored() -> None:
    """Test ptable_scatters with 3rd color dimension."""
    fig = ptable_scatters(
        data={
            "Fe": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "O": [[10, 11], [12, 13], [14, 15]],
        },
        colormap="coolwarm",
        cbar_title="Test ptable_scatters",
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 127
