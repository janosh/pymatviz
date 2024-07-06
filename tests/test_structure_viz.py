from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pymatgen.analysis.local_env import NearNeighbors, VoronoiNN
from pymatgen.core import Structure

from pymatviz.enums import ElemColorScheme, Key
from pymatviz.structure_viz import plot_structure_2d


if TYPE_CHECKING:
    from collections.abc import Sequence

coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
disordered_struct = Structure(
    lattice := np.eye(3) * 5, species=[{"Fe": 0.75, "C": 0.25}, "O"], coords=coords
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

    min_expected_n_patches = 182
    assert patch_counts["PathPatch"] > min_expected_n_patches


@pytest.mark.parametrize("axis", [True, False, "on", "off", "square", "equal"])
def test_plot_structure_2d_axis(axis: str | bool) -> None:
    ax = plot_structure_2d(disordered_struct, axis=axis)
    assert isinstance(ax, plt.Axes)
    assert ax.axes.axison == (axis not in (False, "off"))


@pytest.mark.parametrize(
    "site_labels",
    [True, False, "symbol", "species", {"Fe": "Iron"}, {"Fe": 1.0}, ["Fe", "O"]],
)
def test_plot_structure_2d_site_labels(
    site_labels: bool | str | dict[str, str | float] | Sequence[str],
) -> None:
    ax = plot_structure_2d(disordered_struct, site_labels=site_labels)
    assert isinstance(ax, plt.Axes)
    if site_labels is False:
        assert not ax.axes.texts
    else:
        label = ax.axes.texts[0].get_text()
        assert label in ("Fe", "O", "1.0", "Iron")


def test_plot_structure_2d_warns() -> None:
    # for sites with negative fractional coordinates
    orig_coords = disordered_struct[0].frac_coords.copy()
    disordered_struct[0].frac_coords = [-0.1, 0.1, 0.1]
    standardize_struct = False
    try:
        with pytest.warns(
            UserWarning,
            match=(
                "your structure has negative fractional coordinates but you passed "
                f"{standardize_struct=}, you may want to set standardize_struct=True"
            ),
        ):
            plot_structure_2d(disordered_struct, standardize_struct=standardize_struct)
    finally:
        disordered_struct[0].frac_coords = orig_coords

    # warns when passing subplot_kwargs for a single structure
    with pytest.warns(
        UserWarning, match="subplot_kwargs are ignored when plotting a single structure"
    ):
        plot_structure_2d(disordered_struct, subplot_kwargs={"facecolor": "red"})


struct1 = Structure(lattice, ["Fe", "O"], coords=coords)
struct1.properties = {"id": "struct1"}
struct2 = Structure(lattice, ["Co", "O"], coords=coords)
struct2.properties = {Key.mat_id: "struct2"}
struct3 = Structure(lattice, ["Ni", "O"], coords=coords)
struct3.properties = {"ID": "struct3", "name": "nickel oxide"}  # extra properties
struct4 = Structure(lattice, ["Cu", "O"], coords=coords)


def test_plot_structure_2d_multiple() -> None:
    # Test dict[str, Structure]
    struct_dict = {
        "struct1": struct1,
        "struct2": struct2,
        "struct3": struct3,
        "struct4": struct4,
    }
    fig, axs = plot_structure_2d(struct_dict, n_cols=3)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (2, 3)
    sub_titles = [f"{idx + 1}. {key}" for idx, key in enumerate(struct_dict)]
    assert [ax.get_title() for ax in axs.flat] == sub_titles + [""] * 2

    # Test pandas.Series[Structure]
    struct_series = pd.Series(struct_dict)
    fig, axs = plot_structure_2d(struct_series)
    assert axs.shape == (4,)  # default n_cols=4
    assert [ax.get_title() for ax in axs.flat] == sub_titles
    # Test subplot_kwargs
    fig, axs = plot_structure_2d(struct_series, subplot_kwargs={"squeeze": False})
    assert axs.shape == (1, 4)  # default n_cols=4

    # Test list[Structure]
    fig, axs = plot_structure_2d(list(struct_dict.values()), n_cols=2)
    assert axs.shape == (2, 2)
    assert axs.flat[0].get_title() == f"1. {struct1.properties['id']}"
    assert axs.flat[1].get_title() == f"2. {struct2.properties[Key.mat_id]}"
    assert axs.flat[2].get_title() == f"3. {struct3.properties['ID']}"
    struct4_title = f"4. {struct4.formula} ({struct4.get_space_group_info()[1]})"
    assert axs.flat[3].get_title() == struct4_title

    # Test subplot_title
    def subplot_title(struct: Structure, key: str | int) -> str:
        return f"{key} - {struct.formula}"

    fig, axs = plot_structure_2d(struct_series, subplot_title=subplot_title)
    sub_titles = [
        f"{idx}. {subplot_title(struct, key)}"
        for idx, (key, struct) in enumerate(struct_dict.items(), start=1)
    ]
    assert [ax.get_title() for ax in axs.flat] == sub_titles


def test_plot_structure_2d_color_warning() -> None:
    # Copernicium is not in the default color scheme
    copernicium = "Cn"
    struct = Structure(np.eye(3) * 5, [copernicium] * 2, coords=coords)

    for colors in ElemColorScheme:
        with pytest.warns(
            UserWarning, match=f"{copernicium!r} not in colors, using gray"
        ):
            plot_structure_2d(struct, elem_colors=colors)

    # create custom color scheme missing an element
    custom_colors = {"Fe": "red"}
    struct = Structure(np.eye(3) * 5, ["Fe", "O"], coords=coords)

    with pytest.warns(UserWarning, match="'O' not in colors, using gray"):
        plot_structure_2d(struct, elem_colors=custom_colors)


def test_plot_structure_2d_color_schemes() -> None:
    lattice = np.eye(3) * 5
    coords = (np.ones((3, 4)) * (0, 0.25, 0.5, 0.75)).T
    struct = Structure(lattice, ["Al", "Ar", "As", "Cl"], coords)

    # Plot with Jmol colors
    ax_jmol: plt.Axes = plot_structure_2d(struct, elem_colors=ElemColorScheme.jmol)
    jmol_colors = {
        patch.elem_symbol: patch.get_facecolor()
        for patch in ax_jmol.patches
        if isinstance(patch, plt.matplotlib.patches.Wedge)
    }

    # Plot with VESTA colors
    ax_vesta: plt.Axes = plot_structure_2d(struct, elem_colors=ElemColorScheme.vesta)
    vesta_colors = {
        patch.elem_symbol: patch.get_facecolor()
        for patch in ax_vesta.patches
        if isinstance(patch, plt.matplotlib.patches.Wedge)
    }

    assert (
        jmol_colors != vesta_colors
    ), f"{jmol_colors=}\n\nshould not equal\n\n{vesta_colors=}"
