# %%
from plotly.subplots import make_subplots
from pymatgen.core import Lattice, Structure

import pymatviz as pmv


# Create subplot figure
fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    subplot_titles=["BCC Iron", "FCC Copper", "HCP Magnesium"],
    vertical_spacing=0.05,
    horizontal_spacing=0.05,
)

# Body-centered cubic (BCC) - e.g., Iron
bcc_lattice = Lattice.cubic(2.87)
bcc_structure = Structure(
    bcc_lattice,
    ["Fe", "Fe"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
    coords_are_cartesian=False,
)
fig = pmv.plot_brillouin_zone_3d(
    bcc_structure,
    fig,
    subplot_idx=(1, 1),
    surface_kwargs={"color": "lightgreen", "opacity": 0.3},
    point_kwargs=False,
    path_kwargs=False,
)

# Face-centered cubic (FCC) - e.g., Copper
fcc_lattice = Lattice.cubic(3.61)
fcc_structure = Structure(
    fcc_lattice,
    ["Cu"] * 4,
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    coords_are_cartesian=False,
)
fig = pmv.plot_brillouin_zone_3d(
    fcc_structure,
    fig,
    subplot_idx=(1, 2),
    surface_kwargs={"color": "salmon", "opacity": 0.3},
    point_kwargs={"color": "darkred"},
    path_kwargs={"color": "darkred"},
)

# Hexagonal close-packed (HCP) - e.g., Magnesium
hcp_lattice = Lattice.hexagonal(3.21, 5.21)
hcp_structure = Structure(
    hcp_lattice,
    ["Mg"] * 2,
    [[0, 0, 0], [1 / 3, 2 / 3, 0.5]],
    coords_are_cartesian=False,
)
fig = pmv.plot_brillouin_zone_3d(
    hcp_structure,
    fig,
    subplot_idx=(1, 3),
    surface_kwargs={"color": "lightyellow", "opacity": 0.3},
    point_kwargs={"color": "goldenrod"},
    path_kwargs={"color": "goldenrod"},
)

# Update layout for all subplots
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.0, y=-1.2, z=0.0),
)

scene_kwargs = dict(
    camera=camera,
    xaxis_title="",
    yaxis_title="",
    zaxis_title="",
    xaxis_showticklabels=False,
    yaxis_showticklabels=False,
    zaxis_showticklabels=False,
    xaxis_showbackground=False,
    yaxis_showbackground=False,
    zaxis_showbackground=False,
    aspectmode="cube",
)
fig.update_layout(
    scene=scene_kwargs,
    scene2=scene_kwargs,
    scene3=scene_kwargs,
    showlegend=False,
    width=1000,
    height=400,
    margin=dict(l=0, r=0, t=0, b=0),
)

fig.show()

pmv.io.save_and_compress_svg(fig, "brillouin-zones")
