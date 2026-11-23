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
#     "matminer>=0.10.1",
# ]
# ///

import re
from typing import get_args

import kemio
from kemio.interop import to_structure
from kemio.options import CrystalSystem
from matminer.datasets import load_dataset

import pymatviz as pmv


crystal_systems = get_args(CrystalSystem)  # ordered triclinic to cubic


def count_wyckoff_positions(protostructure_label: str) -> int:
    """Number of occupied Wyckoff positions in an AFLOW-style protostructure label
    like ``A4B3_hP28_159_ab2c_2c:N-Si`` (a digit before a letter repeats that site).
    """
    wyckoff_part = protostructure_label.split(":", 1)[0].split("_", 3)[3]
    site_repeats = re.findall(r"(\d*)[a-z]", wyckoff_part)
    return sum(int(repeat or 1) for repeat in site_repeats)


# %%
df_diel = load_dataset("matbench_dielectric")
kemio_structs = [to_structure(struct) for struct in df_diel["structure"]]

sym_summary = kemio.symmetry_summary(kemio_structs, symprec=0.01)
df_diel["spg_symbol"] = sym_summary.hm_short
df_diel["spg_num"] = sym_summary.number

proto_label_key = "protostructure_moyo"
n_wyckoff_pos_key = "n_wyckoff_pos_moyo"
df_diel[proto_label_key] = kemio.protostructure_labels(kemio_structs)
df_diel[n_wyckoff_pos_key] = df_diel[proto_label_key].map(
    count_wyckoff_positions, na_action="ignore"
)

df_diel["crystal_system"] = df_diel["spg_num"].map(kemio.crystal_system)

df_diel["volume"] = [struct.volume for struct in kemio_structs]
df_diel["formula"] = [struct.composition.formula for struct in kemio_structs]


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
