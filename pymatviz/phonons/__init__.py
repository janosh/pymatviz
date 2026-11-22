"""Phonon density of states plots (with element/site projections) and phonon
database helpers. Plot band structures with BandStructureWidget or
BandsAndDosWidget, which take kemio, pymatgen and phonopy objects and vasprun.xml
paths directly.
"""

from pymatviz.phonons.figures import phonon_dos
from pymatviz.phonons.helpers import PhononDBDoc
