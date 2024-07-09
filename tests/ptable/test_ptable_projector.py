from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pymatviz.enums import ElemColorScheme
from pymatviz.ptable._projector import (
    HMapPTableProjector,
    OverwriteTileValueColor,
    PTableProjector,
    TileValueColor,
)


if TYPE_CHECKING:
    from typing import ClassVar

    from matplotlib.typing import ColorType

    from pymatviz.ptable.ptable_matplotlib import ElemStr


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


class TestPtableHeatmapGenTileValueColors:
    test_dict: ClassVar = {
        "H": 1,  # int
        "He": [2.0],  # float list
    }

    def test_invalid_data_length(self) -> None:
        test_dict = {"H": [1, 2]}
        projector = HMapPTableProjector(data=test_dict, colormap="viridis")

        with pytest.raises(ValueError, match="Data for H should be length 1"):
            projector.generate_tile_value_colors()

    @pytest.mark.parametrize(
        "text_color, expected",
        [
            ("red", "red"),
            ("AUTO", "white"),
            ({"H": "green"}, "green"),
        ],
    )
    def test_text_colors_single_text_color(
        self, text_color: ColorType, expected: ColorType
    ) -> None:
        projector = HMapPTableProjector(data=self.test_dict, colormap="viridis")
        tile_entries = projector.generate_tile_value_colors(text_colors=text_color)

        tile_entry = tile_entries["H"]

        assert_allclose(tile_entry.value, 1.0)
        assert_allclose(
            mpl.colors.to_rgb(tile_entry.text_color), mpl.colors.to_rgb(expected)
        )
        assert_allclose(
            mpl.colors.to_rgb(tile_entry.tile_color), (0.267004, 0.004874, 0.329415)
        )

    @pytest.mark.parametrize(
        "overwrite_tile, expected_tile",
        [
            (  # overwrite value alone
                {"He": OverwriteTileValueColor("hi", None, None)},
                TileValueColor("hi", "black", ([0.993248, 0.906157, 0.143936])),
            ),
            (  # overwrite text_color alone
                {"He": OverwriteTileValueColor(None, "yellow", None)},
                TileValueColor(
                    2.0, mpl.colors.to_rgb("yellow"), ([0.993248, 0.906157, 0.143936])
                ),
            ),
            (  # overwrite tile_color alone
                {"He": OverwriteTileValueColor(None, None, "grey")},
                TileValueColor(2.0, "black", mpl.colors.to_rgb("grey")),
            ),
            (  # overwrite all three
                {"He": OverwriteTileValueColor("hi", "yellow", "grey")},
                TileValueColor(
                    "hi", mpl.colors.to_rgb("yellow"), mpl.colors.to_rgb("grey")
                ),
            ),
        ],
    )
    def test_apply_overwrite_tiles(
        self,
        overwrite_tile: dict[ElemStr, OverwriteTileValueColor],
        expected_tile: TileValueColor,
    ) -> None:
        projector = HMapPTableProjector(data=self.test_dict, colormap="viridis")
        tile_entries = projector.generate_tile_value_colors(
            text_colors="AUTO", overwrite_tiles=overwrite_tile
        )

        tile_entry = tile_entries["He"]

        try:
            assert_allclose(tile_entry.value, expected_tile.value)
        except np.exceptions.DTypePromotionError:
            assert tile_entry.value == expected_tile.value

        assert_allclose(
            mpl.colors.to_rgb(tile_entry.text_color),
            mpl.colors.to_rgb(expected_tile.text_color),
        )
        assert_allclose(
            mpl.colors.to_rgb(tile_entry.tile_color), expected_tile.tile_color
        )

    def test_inf_nan_excluded_color(self) -> None:
        inf_color = "yellow"
        nan_color = "red"
        excluded_color = "lightgrey"

        test_dict = {
            "Li": np.inf,
            "Be": np.nan,
            "B": 1.0,
            "C": 2.0,
        }
        projector = HMapPTableProjector(data=test_dict, exclude_elements=["B"])

        assert projector.anomalies["Li"] == {"inf"}  # type: ignore[index]
        assert projector.anomalies["Be"] == {"nan"}  # type: ignore[index]

        tile_entries = projector.generate_tile_value_colors(
            inf_color=inf_color, nan_color=nan_color, excluded_tile_color=excluded_color
        )

        assert_allclose(
            mpl.colors.to_rgb(tile_entries["Li"].tile_color),
            mpl.colors.to_rgb(inf_color),
        )
        assert_allclose(
            mpl.colors.to_rgb(tile_entries["Be"].tile_color),
            mpl.colors.to_rgb(nan_color),
        )
        assert_allclose(
            mpl.colors.to_rgb(tile_entries["B"].tile_color),
            mpl.colors.to_rgb(excluded_color),
        )
