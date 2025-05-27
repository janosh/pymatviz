"""Minimal example of automatic pymatgen object rendering in notebooks with pymatviz.

All pymatgen objects automatically render as interactive plots when returned from
notebook cells. Works in Jupyter, JupyterLab, Marimo, VSCode interactive windows,
and other notebook environments. Simply import pymatviz and use pymatgen objects!
"""

# %%
# ruff: noqa: B018
import json
from glob import glob

from ase.build import bulk
from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.phonons import PhononDBDoc
from pymatviz.utils.testing import TEST_FILES


# %% Crystal structures auto-render as 3D plots
lattice = Lattice.cubic(4.0)
Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


# %% Perovskite structure
Structure(
    lattice=Lattice.cubic(4.0338),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)


# %% Layered structure
Structure(
    lattice=Lattice.hexagonal(3.16, 12.30),
    species=["Mo", "S", "S"],
    coords=[(0, 0, 0.5), (0.333, 0.667, 0.375), (0.333, 0.667, 0.625)],
)


# %% Phonon objects also auto-render
mp_id, formula = "mp-2758", "Sr4Se4"
docs: dict[str, PhononDBDoc] = {}

for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.xz"):
    key = path.split("-")[-1].split(".")[0]
    with zopen(path, mode="rt") as file:
        docs[key] = json.loads(file.read(), cls=MontyDecoder)

# Individual phonon objects auto-render
# Use the first available document (they all have the same structure)
doc = next(iter(docs.values()))
bands = doc.phonon_bandstructure
dos = doc.phonon_dos

# Phonon band structure auto-renders
bands


# %% Phonon DOS auto-renders
dos


# %% Multiple structures
structures = {
    "Cubic": Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "Hexagonal": Structure(
        Lattice.hexagonal(3.16, 12.30),
        ["Mo", "S", "S"],
        [(0, 0, 0.5), (0.333, 0.667, 0.375), (0.333, 0.667, 0.625)],
    ),
    "Tetragonal": Structure(
        Lattice.tetragonal(4.59, 2.96),
        ["Ti", "O", "O"],
        [(0, 0, 0), (0.3, 0.3, 0), (0.7, 0.7, 0)],
    ),
}

structures["Cubic"]


# %%
structures["Hexagonal"]


# %%
structures["Tetragonal"]


# %% Manual control (optional) - New unified function
nacl = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
pmv.notebook_mode(on=False)  # Disable auto-rendering
nacl  # Now displays as text


# %%
pmv.notebook_mode(on=True)  # Re-enable auto-rendering
nacl  # Auto-renders again


# %% ASE Atoms objects also auto-render
bulk("Si", "diamond", a=5.43)


# %% XRD patterns auto-render
xrd_calc = XRDCalculator()
nacl_xrd = xrd_calc.get_pattern(nacl)
nacl_xrd  # Auto-renders as XRD pattern plot


# %% Additional examples with the unified API
pmv.notebook_mode(on=False)  # Disable auto-rendering
nacl  # Now displays as text


# %%
pmv.notebook_mode(on=True)  # Re-enable auto-rendering
nacl  # Auto-renders again
