import pytest
from matplotlib.offsetbox import AnchoredText

from pymatviz.utils import add_mae_r2_box, get_crystal_sys

from .conftest import y_pred, y_true


def test_add_mae_r2_box():

    text_box = add_mae_r2_box(y_pred, y_true)

    assert isinstance(text_box, AnchoredText)

    txt = "$\\mathrm{MAE} = 0.116$\n$R^2 = 0.740$"
    assert text_box.txt.get_text() == txt

    prefix, suffix = "Metrics:\n", "\nthe end"
    text_box = add_mae_r2_box(y_pred, y_true, prefix=prefix, suffix=suffix)
    assert text_box.txt.get_text() == prefix + txt + suffix


@pytest.mark.parametrize(
    "input, expected",
    [
        (1, "triclinic"),
        (15, "monoclinic"),
        (16, "orthorhombic"),
        (75, "tetragonal"),
        (143, "trigonal"),
        (168, "hexagonal"),
        (230, "cubic"),
    ],
)
def test_get_crystal_sys(input, expected):
    assert expected == get_crystal_sys(input)


@pytest.mark.parametrize("spg", [-1, 0, 231])
def test_get_crystal_sys_invalid(spg):
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        get_crystal_sys(spg)
