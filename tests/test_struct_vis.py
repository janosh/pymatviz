import os
import subprocess
from shutil import which

import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.compare import compare_images
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import SubstitutionTransformation

from ml_matrics.struct_vis import plot_structure_2d



os.makedirs(fixt_dir := "tests/fixtures/struct_vis", exist_ok=True)

latt = Lattice.cubic(5)
struct = Structure(latt, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

disord_struct: Structure = SubstitutionTransformation(
    {"Fe": {"Fe": 0.75, "C": 0.25}}
).apply_transformation(struct)


pngquant, zopflipng = which("pngquant"), which("zopflipng")


def save_fixture(save_to: str) -> None:
    plt.savefig(save_to)
    plt.close()

    if not pngquant:
        return print("Warning: pngquant not installed. Cannot compress new fixture.")
    if not zopflipng:
        return print("Warning: zopflipng not installed. Cannot compress new fixture.")

    subprocess.run(
        f"{pngquant} 32 --skip-if-larger --ext .png --force".split() + [save_to],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        [zopflipng, "-y", save_to, save_to],
        check=True,
        capture_output=True,
    )


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

    if not os.path.exists(fxt_img := f"{fixt_dir}/{fname}"):
        save_fixture(fxt_img)
        return

    plt.savefig(tmp_img)
    plt.close()
    tolerance = 0.8
    assert compare_images(tmp_img, fxt_img, tolerance) is None
