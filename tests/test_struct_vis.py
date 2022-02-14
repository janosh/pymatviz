import matplotlib.pyplot as plt
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import SubstitutionTransformation

from ml_matrics.struct_vis import plot_structure_2d


latt = Lattice.cubic(5)
struct = Structure(latt, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

disord_struct: Structure = SubstitutionTransformation(
    {"Fe": {"Fe": 0.75, "C": 0.25}}
).apply_transformation(struct)


@pytest.mark.parametrize("structure", [struct, disord_struct])
@pytest.mark.parametrize("radii", [0.5, 1.2])
@pytest.mark.parametrize("rotation", ["0x,0y,0z", "10x,-10y,0z"])
def test_plot_structure_2d(structure, radii, rotation):
    ax = plot_structure_2d(structure, atomic_radii=radii, rotation=rotation)
    assert isinstance(ax, plt.Axes)
