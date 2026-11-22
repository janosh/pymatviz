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

# /// script
# dependencies = [
#     "matbench-discovery",
#     "matminer>=0.10.1",
#     "tqdm>=4.67",
# ]
# ///

# %%
from typing import get_args

import kemio
from matbench_discovery.structure import prototype
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.typing import CrystalSystem


crystal_systems = get_args(CrystalSystem)  # ordered triclinic to cubic


# %%
df_diel = load_dataset("matbench_dielectric")

df_diel[["spg_symbol", "spg_num"]] = [
    struct.get_space_group_info()
    for struct in tqdm(df_diel["structure"], desc="Getting spacegroups")
]

proto_label_key = "protostructure_moyo"
n_wyckoff_pos_key = "n_wyckoff_pos_moyo"
df_diel[proto_label_key] = [
    prototype.get_protostructure_label(struct)
    for struct in tqdm(df_diel["structure"], desc="Getting Wyckoff strings")
]
df_diel[n_wyckoff_pos_key] = df_diel[proto_label_key].map(
    prototype.count_wyckoff_positions
)

df_diel["crystal_system"] = df_diel["spg_num"].map(kemio.crystal_system)

df_diel["volume"] = [x.volume for x in df_diel["structure"]]
df_diel["formula"] = [x.formula for x in df_diel["structure"]]


# %%
title = "Elements in Matbench Dielectric"
pmv.ptable_heatmap(df_diel["formula"], log=True, color_bar={"title": title}).show()


# %%
pmv.spacegroup_bar_plot(df_diel["spg_num"]).show()


# %%
widget = pmv.spacegroup_sunburst(df_diel["spg_num"], label_text="label+percent")
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
        df_diel.groupby("crystal_system"), key=lambda x: crystal_systems.index(x[0])
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
df_vol = df_diel.query("volume < 1500")  # exclude few large-volume outliers
widget = pmv.scatter_plot(
    series=[
        {
            "x": df_group["volume"],
            "y": df_group["n"],
            "size_values": df_group["n"],
            "label": cry_sys,
            "markers": "points",
        }
        for cry_sys, df_group in df_vol.groupby("crystal_system")
    ],
    x_axis={"label": "Volume (Å³)"},
    y_axis={"label": "Refractive index n"},
)
# widget.to_img("dielectric-scatter.pdf")
widget.show()
