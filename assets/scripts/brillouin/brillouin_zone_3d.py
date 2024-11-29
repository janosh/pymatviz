# %%
import numpy as np
from pymatgen.core import Lattice, Structure

import pymatviz as pmv


# %% 1. Cubic (BCC Iron)
bcc_lattice = Lattice.cubic(5.0)
bcc_structure = Structure(bcc_lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
fig_bcc = pmv.brillouin_zone_3d(
    bcc_structure,
    surface_kwargs={"color": "lightblue", "opacity": 0.3},
    point_kwargs={"color": "darkblue"},
    path_kwargs={"color": "darkblue"},
)
fig_bcc.layout.title = dict(text="Cubic (BCC Fe)", x=0.5, y=0.97, font_size=20)
fig_bcc.show()


# %% 2. Tetragonal (Rutile TiO₂)
tetra_lattice = Lattice.tetragonal(4.0, 6.0)
tetra_structure = Structure(
    tetra_lattice,
    ["Ti", "O", "O"],
    [[0, 0, 0], [0.3, 0.3, 0], [0.7, 0.7, 0]],
)
fig_tetra = pmv.brillouin_zone_3d(
    tetra_structure,
    surface_kwargs={"color": "lightgreen", "opacity": 0.3},
    point_kwargs={"color": "darkgreen"},
    path_kwargs={"color": "darkgreen"},
)
fig_tetra.layout.title = dict(
    text="Tetragonal (Rutile TiO₂)", x=0.5, y=0.97, font_size=20
)
fig_tetra.show()


# %% 3. Orthorhombic (S₈)
ortho_lattice = Lattice.orthorhombic(4.0, 5.0, 6.0)
ortho_structure = Structure(
    ortho_lattice,
    ["S"] * 4,
    [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
)
fig_ortho = pmv.brillouin_zone_3d(
    ortho_structure,
    surface_kwargs={"color": "salmon", "opacity": 0.3},
    point_kwargs={"color": "darkred"},
    path_kwargs={"color": "darkred"},
)
fig_ortho.layout.title = dict(text="Orthorhombic (S₈)", x=0.5, y=0.97, font_size=20)
fig_ortho.show()


# %% 4. Hexagonal (Mg)
hex_lattice = Lattice.hexagonal(5.0, 8.0)
hex_structure = Structure(
    hex_lattice,
    ["Mg"] * 2,
    [[0, 0, 0], [1 / 3, 2 / 3, 0.5]],
)
fig_hex = pmv.brillouin_zone_3d(
    hex_structure,
    surface_kwargs={"color": "lightyellow", "opacity": 0.3},
    point_kwargs={"color": "goldenrod"},
    path_kwargs={"color": "goldenrod"},
)
fig_hex.layout.title = dict(text="Hexagonal (Mg)", x=0.5, y=0.97, font_size=20)
fig_hex.show()


# %% 5. Trigonal/Rhombohedral (Al₂O₃)
trig_lattice = Lattice.rhombohedral(5.0, 55.0)
trig_structure = Structure(
    trig_lattice,
    ["Al"] * 2 + ["O"] * 3,
    [[0, 0, 0], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7], [0.1, 0.1, 0.1]],
)
fig_trig = pmv.brillouin_zone_3d(
    trig_structure,
    surface_kwargs={"color": "lightpink", "opacity": 0.3},
    point_kwargs={"color": "deeppink"},
    path_kwargs={"color": "deeppink"},
)
fig_trig.layout.title = dict(text="Trigonal (Al₂O₃)", x=0.5, y=0.97, font_size=20)
fig_trig.show()


# %% 6. Monoclinic (CaSO₄)
mono_lattice = Lattice.monoclinic(6.2845, 15.1802, 5.6776, 90.0)
mono_structure = Structure(
    mono_lattice,
    ["Ca", "S", "O", "O", "O", "O"],
    [
        [0, 0, 0],
        [0.5, 0, 0.5],
        [0.3, 0.3, 0.3],
        [0.7, 0.7, 0.7],
        [0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8],
    ],
)
fig_mono = pmv.brillouin_zone_3d(
    mono_structure,
    surface_kwargs={"color": "plum", "opacity": 0.3},
    point_kwargs={"color": "purple"},
    path_kwargs={"color": "purple"},
)
fig_mono.layout.title = dict(text="Monoclinic (CaSO₄)", x=0.5, y=0.97, font_size=20)
fig_mono.show()


# %% 7. Triclinic (CuSO₄)
tri_lattice = Lattice.from_parameters(
    a=6.12, b=10.72, c=5.97, alpha=82.43, beta=107.43, gamma=102.67
)
tri_structure = Structure(
    tri_lattice,
    ["Cu", "S", "O", "O", "O", "O"],
    [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3],
        [0.7, 0.7, 0.7],
        [0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8],
    ],
)
fig_tri = pmv.brillouin_zone_3d(
    tri_structure,
    surface_kwargs={"color": "lightcyan", "opacity": 0.3},
    point_kwargs={"color": "teal"},
    path_kwargs={"color": "teal"},
)
fig_tri.layout.title = dict(text="Triclinic (CuSO₄)", x=0.5, y=0.97, font_size=20)
fig_tri.show()


# %% 8. Non-standard tetragonal structure (TiO₂ with c along x)
# c along x
# a along y
# b along z

non_std_structure = Structure(
    np.diag([6.0, 4.0, 4.0]),
    ["Ti", "O", "O"],
    [[0, 0, 0], [0.3, 0.3, 0], [0.7, 0.7, 0]],
)

# Plot BZ in original non-standard orientation
fig_non_std = pmv.brillouin_zone_3d(non_std_structure)
fig_non_std.layout.title = dict(
    text="Non-standard Tetragonal (TiO₂, c along x)", x=0.5, y=0.97, font_size=20
)
fig_non_std.show()


# Save all figures
# pmv.io.save_and_compress_svg(fig_bcc, "brillouin-zone-bcc")
# pmv.io.save_and_compress_svg(fig_tetra, "brillouin-zone-tetragonal")
# pmv.io.save_and_compress_svg(fig_ortho, "brillouin-zone-orthorhombic")
# pmv.io.save_and_compress_svg(fig_hex, "brillouin-zone-hexagonal")
# pmv.io.save_and_compress_svg(fig_trig, "brillouin-zone-trigonal")
# pmv.io.save_and_compress_svg(fig_mono, "brillouin-zone-monoclinic")
# pmv.io.save_and_compress_svg(fig_tri, "brillouin-zone-triclinic")
# pmv.io.save_and_compress_svg(fig_non_std, "brillouin-zone-non-standard-tetragonal")
# pmv.io.save_and_compress_svg(fig_std, "brillouin-zone-standardized-tetragonal")
