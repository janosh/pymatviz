from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pymatgen.analysis.local_env import NearNeighbors, VoronoiNN
from pymatgen.core import Lattice, Structure

from pymatviz.structure_viz import plot_structure_2d


lattice = Lattice.cubic(5)
disordered_struct = Structure(
    lattice, [{"Fe": 0.75, "C": 0.25}, "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)


@pytest.mark.parametrize("radii", [None, 0.5])
@pytest.mark.parametrize("rotation", ["0x,0y,0z", "10x,-10y,0z"])
@pytest.mark.parametrize("labels", [True, False, {"P": "Phosphor"}])
# show_bonds=True|VoronoiNN used to raise AttributeError on
# disordered structures https://github.com/materialsproject/pymatgen/issues/2070
# which we work around by only considering majority species on each site
@pytest.mark.parametrize("show_bonds", [False, True, VoronoiNN])
@pytest.mark.parametrize("standardize_struct", [None, False, True])
def test_plot_structure_2d(
    radii: float | None,
    rotation: str,
    labels: bool | dict[str, str | float],
    show_bonds: bool | NearNeighbors,
    standardize_struct: bool | None,
) -> None:
    ax = plot_structure_2d(
        disordered_struct,
        atomic_radii=radii,
        rotation=rotation,
        site_labels=labels,
        show_bonds=show_bonds,
        standardize_struct=standardize_struct,
    )
    assert isinstance(ax, plt.Axes)

    assert ax.get_aspect() == 1, "aspect ratio should be set to 'equal', i.e. 1:1"
    x_min, x_max, y_min, y_max = ax.axis()
    assert x_min == y_min == 0, "x/y_min should be 0"
    assert x_max > 5
    assert y_max > 5

    patch_counts = pd.Series(
        [type(patch).__name__ for patch in ax.patches]
    ).value_counts()
    assert patch_counts["Wedge"] == len(disordered_struct.composition)

    assert patch_counts["PathPatch"] > 182


@pytest.mark.parametrize("axis", [True, False, "on", "off", "square", "equal"])
def test_plot_structure_2d_axis(axis: str | bool) -> None:
    ax = plot_structure_2d(disordered_struct, axis=axis)
    assert ax.axes.axison is False if axis in (False, "off") else True
