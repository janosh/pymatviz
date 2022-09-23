import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio


"""
Importing this module has side-effects that apply sensible (often, not always) global
defaults settings for plotly and matplotlib like increasing font size, prettier
axis labels (plotly only) and higher figure resolution (matplotlib only).

To use it, simply import this module before generating any plots:

import pymatviz.plot_defaults
# or
from pymatviz.plot_defaults import plt, px, pio
"""

# prettier
px.defaults.labels = {
    "n_atoms": "Atom Count",
    "n_elems": "Element Count",
    "gap expt": "Experimental band gap (eV)",
    "crystal_sys": "Crystal system",
    "n": "Refractive index n",
    "spg_num": "Space group",
    "n_wyckoff": "Number of Wyckoff positions",
    "n_sites": "Number of unit cell sites",
    "energy_per_atom": "Energy (eV/atom)",
}

pio.templates.default = "plotly_white"

# https://github.com/plotly/Kaleido/issues/122#issuecomment-994906924
# pio.kaleido.scope.mathjax = None

crystal_sys_order = (
    "cubic hexagonal trigonal tetragonal orthorhombic monoclinic triclinic".split()
)


plt.rc("font", size=16)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("figure", dpi=200, titlesize=18)
plt.rcParams["figure.constrained_layout.use"] = True
