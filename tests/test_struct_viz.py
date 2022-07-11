from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.compare import compare_images
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import SubstitutionTransformation

from pymatviz.structure_viz import plot_structure_2d

from .conftest import save_reference_img


os.makedirs(fixture_dir := "tests/fixtures/structure_viz", exist_ok=True)

lattice = Lattice.cubic(5)
struct = Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

disordered_struct: Structure = SubstitutionTransformation(
    {"Fe": {"Fe": 0.75, "C": 0.25}}
).apply_transformation(struct)


@pytest.mark.parametrize("radii", [None, 0.5])
@pytest.mark.parametrize("rotation", ["0x,0y,0z", "10x,-10y,0z"])
@pytest.mark.parametrize("labels", [True, False, {"P": "Phosphor"}])
# add True to [False, VoronoiNN] later to test CrystalNN which currently errors on
# disordered structures https://github.com/materialsproject/pymatgen/issues/2070
@pytest.mark.parametrize("show_bonds", [False, VoronoiNN])
def test_plot_structure_2d(tmp_path, radii, rotation, labels, show_bonds):
    # set explicit size to avoid ImageComparisonFailure in CI: sizes do not match
    # expected (700, 1350, 3), actual (480, 640, 3)
    plt.figure(figsize=(5, 5))

    ax = plot_structure_2d(
        disordered_struct,
        atomic_radii=radii,
        rotation=rotation,
        site_labels=labels,
        show_bonds=show_bonds,
    )
    assert isinstance(ax, plt.Axes)

    if isinstance(labels, dict):  # warning: we overwrite labels here
        labels = ",".join(f"{k}={v}" for k, v in labels.items())
    file_path = f"{radii=}_{rotation=}_{labels=}.png"
    tmp_img = f"{tmp_path}/{file_path}"

    if not os.path.exists(ref_img := f"{fixture_dir}/{file_path}"):
        save_reference_img(ref_img)
        return

    plt.savefig(tmp_img)
    plt.close()
    tolerance = 100
    assert compare_images(tmp_img, ref_img, tolerance) is None
