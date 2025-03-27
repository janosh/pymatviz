# /// script
# dependencies = [
#     "seekpath>=2.1",
# ]
# ///


# %%
from glob import glob

import numpy as np
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES


structures = {
    tuple(
        path.split("/")[-1].replace(".json.gz", "").rsplit("-", maxsplit=2)
    ): Structure.from_file(path)
    for path in glob(f"{TEST_FILES}/structures/mp-*-*-*.json.gz")
}


def simple_subplot_title(_struct: Structure, key: tuple[str, str, str]) -> str:
    """Custom subplot title function."""
    _mat_id, formula, system = key
    return f"{formula} {system.title()}"


def volume_subplot_title(struct: Structure, key: tuple[str, str, str]) -> str:
    """Custom subplot title function showing structure and BZ volumes."""
    _mat_id, formula, system = key
    struct_vol = struct.volume  # in Å³
    # BZ volume is (2π)³/struct_vol in Å⁻³
    bz_vol = (2 * np.pi) ** 3 / struct_vol
    return (
        f"{formula} {system.title()}<br>V<sub>struct</sub> = {struct_vol:.1f} Å³, "
        f"V<sub>BZ</sub> = {bz_vol:.1f} Å⁻³"
    )


def spacegroup_subplot_title(struct: Structure, key: tuple[str, str, str]) -> str:
    """Custom subplot title function showing spacegroup information."""
    _mat_id, formula, system = key
    sym_data = struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True)
    return f"{formula} {system.title()}<br>Space group: {sym_data.number}"


# %% render example Brillouin zone for each crystal system individually
for (mat_id, formula, system), struct in structures.items():
    fig = pmv.brillouin_zone_3d(struct)
    title = f"{formula} {system.title()} ({mat_id})"
    fig.layout.title = dict(text=title, x=0.5, y=0.97, font_size=20)
    fig.show()
    # pmv.io.save_and_compress_svg(fig, f"brillouin-{system.lower()}-{mat_id}")


# %% pass ase.Atoms
atoms = next(iter(structures.values()))
fig = pmv.brillouin_zone_3d(atoms)
fig.show()


# %% 2x4 grid
fig = pmv.brillouin_zone_3d(
    structures,
    n_cols=4,
    subplot_title=simple_subplot_title,  # type: ignore[arg-type]
    surface_kwargs=dict(opacity=0.4),
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "brillouin-2x4-grid")


# %% 3x3 grid with structure and BZ volumes
fig = pmv.brillouin_zone_3d(
    # only pass 6 structures to avoid crowding
    {k: v for idx, (k, v) in enumerate(structures.items()) if idx < 6},
    n_cols=3,
    subplot_title=volume_subplot_title,  # type: ignore[arg-type]
    surface_kwargs=dict(opacity=0.4),
)
fig.layout.title = dict(
    text="Brillouin zones with volumes", x=0.5, y=0.99, font_size=24
)
fig.layout.margin = dict(l=0, r=0, t=40, b=0)
fig.show()
# pmv.io.save_and_compress_svg(fig, "brillouin-volumes-3-cols")


# %% 4x2 grid with spacegroup information
fig = pmv.brillouin_zone_3d(
    structures,
    n_cols=2,
    subplot_title=spacegroup_subplot_title,  # type: ignore[arg-type]
    surface_kwargs=dict(opacity=0.4),
)
fig.layout.title = dict(
    text="Brillouin zones and spacegroups", x=0.5, y=0.99, font_size=24
)
fig.layout.margin = dict(l=0, r=0, t=40, b=0)
fig.show()
# pmv.io.save_and_compress_svg(fig, "brillouin-symmetries-4x2-grid")
