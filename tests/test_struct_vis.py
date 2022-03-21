import os

import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.compare import compare_images
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import SubstitutionTransformation

from pymatviz.struct_vis import plot_structure_2d

from .conftest import save_reference_img


os.makedirs(fixt_dir := "tests/fixtures/struct_vis", exist_ok=True)

latt = Lattice.cubic(5)
struct = Structure(latt, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

disord_struct: Structure = SubstitutionTransformation(
    {"Fe": {"Fe": 0.75, "C": 0.25}}
).apply_transformation(struct)


@pytest.mark.parametrize("radii", [0.5, 1.2])
@pytest.mark.parametrize("rot", ["0x,0y,0z", "10x,-10y,0z"])
@pytest.mark.parametrize("labels", [True, False, {"P": "Phosphor"}])
def test_plot_structure_2d(radii, rot, labels, tmpdir):
    # set explicit size to avoid ImageComparisonFailure in CI: sizes do not match
    # expected (700, 1350, 3), actual (480, 640, 3)
    plt.figure(figsize=(5, 5))

    ax = plot_structure_2d(
        disord_struct, atomic_radii=radii, rotation=rot, site_labels=labels
    )
    assert isinstance(ax, plt.Axes)
    fname = f"{radii=}_{rot=}_{labels=}.png"

    tmp_img = tmpdir.join(fname).strpath

    if not os.path.exists(ref_img := f"{fixt_dir}/{fname}"):
        save_reference_img(ref_img)
        return

    plt.savefig(tmp_img)
    plt.close()
    tolerance = 100
    assert compare_images(tmp_img, ref_img, tolerance) is None
