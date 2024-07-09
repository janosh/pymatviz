from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pymatviz.enums import ElemColorScheme
from pymatviz.ptable.matplotlib import PTableProjector


if TYPE_CHECKING:
    from typing import ClassVar


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
            PTableProjector(data=data, elem_colors="foobar")

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
