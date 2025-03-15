import os
import warnings
from collections.abc import Sequence

import numpy as np
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import BadInputSetWarning, MPStaticSet

from pymatviz.utils import ROOT


# silence verbose pymatgen warnings
warnings.filterwarnings("ignore", category=BadInputSetWarning)
warnings.filterwarnings("ignore", message="No Pauling electronegativity for")


def create_diatomic_inputs(
    distances: Sequence[float] = (1, 10, 40),
    box_size: tuple[float, float, float] = (10, 10, 20),
    elements: Sequence[str] | set[str] = (),
    base_dir: str = "diatomic-calcs",
) -> None:
    """Create VASP input files for all pairs of elements at different separations.
    The calculations can be run using run_vasp_diatomics.py, which will automatically
    handle running calculations in sequence and copying WAVECAR files between distances.

    Args:
        distances (tuple[float, ...]): If tuple and length is 3 and last item is int,
            values will be passed to np.logspace as (min_dist, max_dist, n_points).
            Else will be used as a list of distances to sample. Defaults to (1, 10, 40).
        box_size (tuple[float, float, float]): Size of the cubic box in Å.
            Defaults to (10, 10, 20).
        elements (set[str]): Elements to include. Defaults to all elements.
        base_dir (str): Base directory to store the input files. Defaults to
            "diatomic-calcs".
    """
    if (
        isinstance(distances, tuple)
        and len(distances) == 3
        and isinstance(distances[-1], int)
    ):
        min_dist, max_dist, n_points = distances
        distances = np.logspace(np.log10(min_dist), np.log10(max_dist), n_points)
    box = Lattice.orthorhombic(*box_size)

    if elements == ():
        # skip superheavy elements (most have no POTCARs and are radioactive)
        skip_elements = set(
            "Am At Bk Cf Cm Es Fr Fm Md No Lr Rf Po Db Sg Bh Hs Mt Ds Cn Nh Fl Mc Lv "  # noqa: SIM905
            "Ra Rg Ts Og".split()
        )
        elements = sorted({*map(str, Element)} - set(skip_elements))

    os.makedirs(base_dir, exist_ok=True)
    print(f"Created {base_dir=}")
    # Loop over all pairs of elements
    for elem1 in elements:
        elem1_dir = f"{base_dir}/{elem1}"
        os.makedirs(elem1_dir, exist_ok=True)

        for elem2 in elements:
            elem2_dir = f"{elem1_dir}/{elem1}-{elem2}"
            os.makedirs(elem2_dir, exist_ok=True)

            for dist in distances:
                # Center the atoms in the box
                center = np.array(box_size) / 2
                coords_1 = center - np.array([0, 0, dist / 2])
                coords_2 = center + np.array([0, 0, dist / 2])

                # Create the structure and input set
                dimer = Structure(
                    box, [elem1, elem2], (coords_1, coords_2), coords_are_cartesian=True
                )

                # Create directory for this distance
                dist_dir = f"{elem2_dir}/{dist=:.3f}"
                os.makedirs(dist_dir, exist_ok=True)

                # Generate VASP input files
                vasp_input_set = MPStaticSet(
                    dimer,
                    user_kpoints_settings=Kpoints(),  # sample a single k-point at Gamma
                    # disable symmetry since spglib in VASP sometimes detects false
                    # symmetries in dimers and fails
                    user_incar_settings={"ISYM": 0, "LH5": True},
                )
                vasp_input_set.write_input(dist_dir)

            print(f"Created inputs for {elem1}-{elem2} pair")


if __name__ == "__main__":
    create_diatomic_inputs(base_dir=f"{ROOT}/tmp/diatomic-calcs")
