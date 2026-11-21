"""matbench_dielectric dataset.

Input: Pymatgen Structure of the material.
Target variable: refractive index.
Entries: 4,764

Matbench v0.1 test dataset for predicting refractive index from structure. Adapted from
Materials Project database. Removed entries having a formation energy (or energy above
the convex hull) more than 150meV and those having refractive indices less than 1 and
those containing noble gases. Retrieved April 2, 2019.

https://ml.materialsproject.org/projects/matbench_dielectric
"""

# %%
from typing import get_args

from matbench_discovery.structure import prototype
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.typing import CrystalSystem


crystal_systems = get_args(CrystalSystem)  # ordered triclinic to cubic


# %%
df_diel = load_dataset("matbench_dielectric")

df_diel[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info()
    for struct in tqdm(df_diel[Key.structure], desc="Getting spacegroups")
]

proto_label_key = f"{Key.protostructure}_moyo"
n_wyckoff_pos_key = f"{Key.n_wyckoff_pos}_moyo"
df_diel[proto_label_key] = [
    prototype.get_protostructure_label(struct)
    for struct in tqdm(df_diel[Key.structure], desc="Getting Wyckoff strings")
]
df_diel[n_wyckoff_pos_key] = df_diel[proto_label_key].map(
    prototype.count_wyckoff_positions
)

df_diel[Key.crystal_system] = df_diel[Key.spg_num].map(pmv.utils.spg_to_crystal_sys)

df_diel[Key.volume] = [x.volume for x in df_diel[Key.structure]]
df_diel[Key.formula] = [x.formula for x in df_diel[Key.structure]]


# %%
title = "Elements in Matbench Dielectric"
pmv.ptable_heatmap(df_diel[Key.formula], log=True, color_bar={"title": title}).show()


# %%
pmv.spacegroup_bar_plot(df_diel[Key.spg_num]).show()


# %%
widget = pmv.spacegroup_sunburst(df_diel[Key.spg_num], label_text="label+percent")
# widget.to_img("dielectric-spacegroup-sunburst.pdf")
widget.show()


# %% MatterViz has no rich-text tick labels, so per-group stats go into series labels
def violins_by_crystal_system(
    y_col: str, title: str, *, log_y: bool = False, show_mean: bool = False
) -> pmv.widgets.BoxPlotWidget:
    """Violin plot (with all points) of y_col per crystal system, labeled with each
    group's size (and optionally mean).
    """
    series = []
    for cry_sys, df_group in sorted(
        df_diel.groupby(Key.crystal_system), key=lambda x: crystal_systems.index(x[0])
    ):
        label = f"{cry_sys} {len(df_group):,} = {len(df_group) / len(df_diel):.0%}"
        if show_mean:
            label += f" mean={df_group[y_col].mean():.1f}"
        series += [{"y": df_group[y_col], "label": label}]
    return pmv.box_plot(
        series,
        kind="violin",
        points="strip",
        title=title,
        show_legend=False,
        y_axis={"label": y_col, "scale_type": "log" if log_y else "linear"},
    )


widget = violins_by_crystal_system(
    "n", "Refractive index distribution by crystal system"
)
# widget.to_img("dielectric-violin.pdf")
widget.show()


# %%
widget = violins_by_crystal_system(
    n_wyckoff_pos_key,
    "Matbench dielectric: Number of Wyckoff positions by crystal system",
    log_y=True,
    show_mean=True,
)
# widget.to_img("dielectric-violin-num-wyckoffs.pdf")
widget.show()


# %% Matbench Dielectric: Refractive Index vs. Volume (marker size = n)
df_vol = df_diel.query(f"{Key.volume} < 1500")  # exclude few large-volume outliers
widget = pmv.scatter_plot(
    series=[
        {
            "x": df_group[Key.volume],
            "y": df_group["n"],
            "size_values": df_group["n"],
            "label": cry_sys,
            "markers": "points",
        }
        for cry_sys, df_group in df_vol.groupby(Key.crystal_system)
    ],
    x_axis={"label": Key.volume.label},
    y_axis={"label": "Refractive index n"},
)
# widget.to_img("dielectric-scatter.pdf")
widget.show()
